"""
SafeQR — physical_analyzer.py
Physical Tamper Analyzer.

Input : Raw image bytes + QR bounding box (from qr_extractor)
Output: PhysicalAnalysis (status, tampered, confidence, evidence, checks)

Checks performed (all deterministic, no ML):
  1. Edge Anomaly    — Canny edges inside QR region, checks for unnatural sharp lines
  2. Overlay Patch   — Contour detection for foreign rectangles pasted over QR
  3. Obstruction     — Percentage of QR region that is too dark/light (covered)
  4. Contrast        — Local contrast variance inside QR vs surrounding area

Each check returns a score 0-100 and a finding string.
Final confidence = weighted average of triggered checks.
"""

import logging
import math
from typing import Tuple, Optional

import cv2
import numpy as np

from app.utils.image_utils import load_image_from_bytes, ImageLoadError, resize_if_needed
from app.models.response_models import PhysicalAnalysis, BoundingBox, TamperStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Thresholds — tuned for real QR tamper cases
# ─────────────────────────────────────────────

EDGE_ANOMALY_THRESHOLD      = 0.18   # Ratio of edge pixels inside QR that are suspicious
OVERLAY_CONTOUR_MIN_AREA    = 0.08   # Foreign contour must be >8% of QR area to flag
OBSTRUCTION_DARK_THRESHOLD  = 30     # Pixel value below this = "covered/dark"
OBSTRUCTION_LIGHT_THRESHOLD = 245    # Pixel value above this = "covered/bright patch"
OBSTRUCTION_MAX_RATIO       = 0.20   # >20% of QR blocked = suspicious
CONTRAST_DROP_RATIO         = 0.40   # Local contrast inside QR < 40% of surrounding = suspicious

# Verdict thresholds
SCORE_CLEAN      = 30
SCORE_SUSPICIOUS = 60


# ─────────────────────────────────────────────
# Individual Checks
# ─────────────────────────────────────────────

def _check_edge_anomaly(
    gray: np.ndarray,
    qr_region: np.ndarray,
    bbox: BoundingBox
) -> Tuple[float, str]:
    """
    Detect unnatural sharp edges inside the QR region.
    A real QR code has regular, grid-aligned edges.
    Tampered QRs often have diagonal cuts or soft-edged overlays.

    Returns (score 0-100, finding string)
    """
    if qr_region.size == 0:
        return 0.0, "Edge check skipped — empty region"

    # Canny edge detection on QR region
    edges = cv2.Canny(qr_region, threshold1=50, threshold2=150)

    total_pixels = edges.size
    edge_pixels  = np.count_nonzero(edges)
    edge_ratio   = edge_pixels / total_pixels if total_pixels > 0 else 0

    # Check for diagonal edges (non-axis-aligned lines) using Hough
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180,
        threshold=30, minLineLength=15, maxLineGap=5
    )

    diagonal_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                angle = 90.0
            else:
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            # Not horizontal (0°) or vertical (90°) → diagonal = suspicious
            if not (angle < 5 or abs(angle - 90) < 5 or abs(angle - 180) < 5):
                diagonal_count += 1

    diagonal_ratio = diagonal_count / max(len(lines), 1) if lines is not None else 0

    if edge_ratio > EDGE_ANOMALY_THRESHOLD and diagonal_ratio > 0.3:
        score = min(100, int((edge_ratio / EDGE_ANOMALY_THRESHOLD) * 50 + diagonal_ratio * 50))
        finding = f"Edge anomaly: {edge_ratio:.2%} edge density, {diagonal_count} diagonal lines detected"
    else:
        score = int(edge_ratio * 100)
        finding = f"Edge pattern normal: {edge_ratio:.2%} density, {diagonal_count} diagonal lines"

    return float(score), finding


def _check_overlay_patch(
    gray: np.ndarray,
    qr_region: np.ndarray,
    bbox: BoundingBox
) -> Tuple[float, str]:
    """
    Detect rectangular foreign objects pasted over the QR code.
    Sticker-over-QR attacks show up as clean-edged rectangles with
    different texture than surrounding QR modules.

    Returns (score 0-100, finding string)
    """
    if qr_region.size == 0:
        return 0.0, "Overlay check skipped — empty region"

    qr_area = bbox.width * bbox.height

    # Threshold to find distinct regions
    _, thresh = cv2.threshold(qr_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (qr_area * OVERLAY_CONTOUR_MIN_AREA):
            continue  # Too small to be a patch

        # Check how rectangular it is
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:  # Rectangular contour
            x, y, w, h = cv2.boundingRect(cnt)
            # Adjust coordinates to full image space
            abs_x = bbox.x + x
            abs_y = bbox.y + y
            suspicious_contours.append((abs_x, abs_y, w, h, area))

    if suspicious_contours:
        largest = max(suspicious_contours, key=lambda c: c[4])
        ax, ay, aw, ah, area = largest
        coverage = area / qr_area
        score = min(100, int(coverage * 200))
        finding = f"Overlay contour detected at ({ax},{ay}), size {aw}x{ah}px, covers {coverage:.1%} of QR"
    else:
        score = 0
        finding = "No overlay patches detected"

    return float(score), finding


def _check_obstruction(
    gray: np.ndarray,
    qr_region: np.ndarray,
    bbox: BoundingBox
) -> Tuple[float, str]:
    """
    Detect if a significant portion of the QR is covered/blocked.
    Checks for abnormally dark or bright uniform patches inside the QR.

    Returns (score 0-100, finding string)
    """
    if qr_region.size == 0:
        return 0.0, "Obstruction check skipped — empty region"

    total = qr_region.size

    dark_pixels  = np.sum(qr_region < OBSTRUCTION_DARK_THRESHOLD)
    light_pixels = np.sum(qr_region > OBSTRUCTION_LIGHT_THRESHOLD)

    dark_ratio  = dark_pixels / total
    light_ratio = light_pixels / total

    # A real QR has ~50% dark, ~50% light but with variation
    # Extreme imbalance suggests obstruction
    blocked_ratio = max(dark_ratio, light_ratio)

    if blocked_ratio > OBSTRUCTION_MAX_RATIO:
        score = min(100, int(blocked_ratio * 300))
        dominant = "dark" if dark_ratio > light_ratio else "bright"
        finding = f"Partial obstruction: {blocked_ratio:.1%} {dominant} pixels, possible physical cover"
    else:
        score = int(blocked_ratio * 100)
        finding = f"No obstruction: pixel distribution normal ({dark_ratio:.1%} dark, {light_ratio:.1%} bright)"

    return float(score), finding


def _check_contrast(
    gray: np.ndarray,
    qr_region: np.ndarray,
    bbox: BoundingBox
) -> Tuple[float, str]:
    """
    Compare local contrast inside QR vs surrounding area.
    Tampered/replaced QRs often have different contrast profile
    than the surrounding poster/image.

    Returns (score 0-100, finding string)
    """
    if qr_region.size == 0:
        return 0.0, "Contrast check skipped — empty region"

    h, w = gray.shape

    # QR region contrast
    qr_std = float(np.std(qr_region))

    # Surrounding region (expand bbox by 50%, clamped to image)
    pad = int(max(bbox.width, bbox.height) * 0.5)
    sx1 = max(0, bbox.x - pad)
    sy1 = max(0, bbox.y - pad)
    sx2 = min(w, bbox.x + bbox.width + pad)
    sy2 = min(h, bbox.y + bbox.height + pad)

    surrounding = gray[sy1:sy2, sx1:sx2]

    # Mask out the QR region itself from surrounding
    mask = np.ones(surrounding.shape, dtype=bool)
    rel_x = bbox.x - sx1
    rel_y = bbox.y - sy1
    mask[rel_y:rel_y+bbox.height, rel_x:rel_x+bbox.width] = False
    surrounding_only = surrounding[mask]

    if surrounding_only.size == 0:
        return 0.0, "Contrast check skipped — no surrounding area"

    surr_std = float(np.std(surrounding_only))

    if surr_std == 0:
        return 0.0, "Contrast check skipped — uniform surrounding"

    ratio = qr_std / surr_std

    if ratio < CONTRAST_DROP_RATIO:
        score = min(100, int((1 - ratio) * 100))
        finding = f"Contrast irregularity: QR contrast ({qr_std:.1f}) is {ratio:.1%} of surrounding ({surr_std:.1f})"
    elif ratio > (1 / CONTRAST_DROP_RATIO):
        score = min(100, int((ratio - 1) * 50))
        finding = f"Contrast spike: QR contrast ({qr_std:.1f}) unusually high vs surrounding ({surr_std:.1f})"
    else:
        score = 0
        finding = f"Contrast normal: QR ({qr_std:.1f}) vs surrounding ({surr_std:.1f}), ratio {ratio:.2f}"

    return float(score), finding


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def _compute_verdict(
    edge_score: float,
    overlay_score: float,
    obstruction_score: float,
    contrast_score: float
) -> Tuple[float, TamperStatus]:
    """
    Weighted combination of all check scores → final confidence + status.

    Weights reflect severity:
      overlay   = 0.40  (most reliable indicator)
      edge      = 0.25
      contrast  = 0.20
      obstruct  = 0.15
    """
    confidence = (
        overlay_score      * 0.40 +
        edge_score         * 0.25 +
        contrast_score     * 0.20 +
        obstruction_score  * 0.15
    )
    confidence = round(confidence, 1)

    if confidence >= SCORE_SUSPICIOUS:
        status = TamperStatus.TAMPERED
    elif confidence >= SCORE_CLEAN:
        status = TamperStatus.SUSPICIOUS
    else:
        status = TamperStatus.CLEAN

    return confidence, status


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def analyze_physical(
    image_bytes: bytes,
    bbox: Optional[BoundingBox]
) -> PhysicalAnalysis:
    """
    Main entry point.
    Accepts raw image bytes + QR bounding box from qr_extractor.
    Returns fully-typed PhysicalAnalysis.

    Never raises — all errors produce a CLEAN result with error note.
    """
    # ── Load image ────────────────────────────
    try:
        img = load_image_from_bytes(image_bytes)
        img = resize_if_needed(img)
    except ImageLoadError as e:
        logger.error(f"Physical analyzer image load failed: {e}")
        return _error_result(str(e))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── No bounding box — analyze full image ──
    if bbox is None:
        logger.warning("No bounding box provided — analyzing full image center")
        h, w = gray.shape
        pad = min(h, w) // 4
        bbox = BoundingBox(x=pad, y=pad, width=w - pad*2, height=h - pad*2)

    # ── Extract QR region ─────────────────────
    qr_region = gray[
        bbox.y : bbox.y + bbox.height,
        bbox.x : bbox.x + bbox.width
    ]

    # ── Run all 4 checks ──────────────────────
    edge_score,        edge_finding        = _check_edge_anomaly(gray, qr_region, bbox)
    overlay_score,     overlay_finding     = _check_overlay_patch(gray, qr_region, bbox)
    obstruction_score, obstruction_finding = _check_obstruction(gray, qr_region, bbox)
    contrast_score,    contrast_finding    = _check_contrast(gray, qr_region, bbox)

    logger.debug(f"Scores — edge:{edge_score:.1f} overlay:{overlay_score:.1f} "
                 f"obstruct:{obstruction_score:.1f} contrast:{contrast_score:.1f}")

    # ── Compute verdict ───────────────────────
    confidence, status = _compute_verdict(
        edge_score, overlay_score, obstruction_score, contrast_score
    )

    tampered = status in (TamperStatus.TAMPERED, TamperStatus.SUSPICIOUS)

    # ── Build evidence string ─────────────────
    # Lead with the highest-scoring finding
    findings = [
        (edge_score,        edge_finding),
        (overlay_score,     overlay_finding),
        (obstruction_score, obstruction_finding),
        (contrast_score,    contrast_finding),
    ]
    primary_finding = max(findings, key=lambda f: f[0])[1]

    logger.info(f"Physical analysis: {status} (confidence={confidence})")

    return PhysicalAnalysis(
        status=status,
        tampered=tampered,
        confidence=confidence,
        evidence=primary_finding,
        checks={
            "edge_anomaly":   {"score": edge_score,        "finding": edge_finding},
            "overlay_patch":  {"score": overlay_score,     "finding": overlay_finding},
            "obstruction":    {"score": obstruction_score, "finding": obstruction_finding},
            "contrast":       {"score": contrast_score,    "finding": contrast_finding},
        }
    )


def _error_result(error_msg: str) -> PhysicalAnalysis:
    return PhysicalAnalysis(
        status=TamperStatus.CLEAN,
        tampered=False,
        confidence=0.0,
        evidence=f"Analysis failed: {error_msg}",
        checks={}
    )


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python physical_analyzer.py <image_path>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        raw = f.read()

    # Run without bbox first (full image analysis)
    result = analyze_physical(raw, bbox=None)
    print(json.dumps(result.model_dump(), indent=2))
