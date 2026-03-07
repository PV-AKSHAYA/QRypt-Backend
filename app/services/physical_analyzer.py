"""
app/services/physical_analyzer.py
───────────────────────────────────
Physical Tamper Analyzer.

Input  : raw image bytes
Output : PhysicalLayerResult (tampered, confidence, evidence)

Detects:
  1. Double-edge signature   — sticker pasted over original QR
  2. Overlay patch           — foreign rectangular region inside QR zone
  3. Contrast irregularity   — brightness discontinuity at sticker boundary
  4. Partial obstruction     — QR finder patterns missing/blocked

All checks are deterministic (no ML, no randomness).
Same image always returns same result.
"""

import cv2
import numpy as np
import logging

from app.models.response_models import PhysicalLayerResult
from app.utils.image_utils import bytes_to_cv2

logger = logging.getLogger("safeqr.physical_analyzer")


# ══════════════════════════════════════════════════════════════
#  CONSTANTS — tuned for real-world QR photos
# ══════════════════════════════════════════════════════════════

# Contour area range that qualifies as "QR-sized"
QR_AREA_MIN = 5000
QR_AREA_MAX = 80_000

# How many suspicious rectangular layers trigger tamper flag
DOUBLE_EDGE_THRESHOLD = 5

# Brightness std-dev above which we flag contrast irregularity
CONTRAST_STD_THRESHOLD = 118.0

# Minimum confidence to call it tampered
TAMPER_CONFIDENCE_THRESHOLD = 45


# ══════════════════════════════════════════════════════════════
#  CHECK 1 — Double-Edge Signature
# ══════════════════════════════════════════════════════════════

def _check_double_edge(gray: np.ndarray) -> tuple[bool, int, str]:
    """
    Detects sticker overlay by counting overlapping rectangular contours
    in the QR region. A real QR has one clean boundary.
    A stickered QR has 2+ nested rectangular boundaries.

    Returns: (flagged, score 0-100, evidence_string)
    """
    edges    = cv2.Canny(gray, 30, 120)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    rectangular_layers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (QR_AREA_MIN < area < QR_AREA_MAX):
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:   # rectangular shape
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            # QR codes are roughly square — 0.7 to 1.3 aspect ratio
            if 0.7 < aspect < 1.3:
                rectangular_layers.append((x, y, w, h, area))

    count = len(rectangular_layers)

    if count >= DOUBLE_EDGE_THRESHOLD:
        # Report the two most prominent locations
        top2    = sorted(rectangular_layers, key=lambda c: c[4], reverse=True)[:2]
        coords  = ", ".join(f"({c[0]},{c[1]})" for c in top2)
        score   = min(count * 22, 60)
        return True, score, f"Double-edge signature: {count} rectangular layers at {coords}"

    return False, 0, ""


# ══════════════════════════════════════════════════════════════
#  CHECK 2 — Overlay Patch Detection
# ══════════════════════════════════════════════════════════════

def _check_overlay_patch(gray: np.ndarray) -> tuple[bool, int, str]:
    """
    Detects a foreign patch (sticker) by looking for a rectangular
    region with significantly different mean brightness from surroundings.

    Returns: (flagged, score 0-100, evidence_string)
    """
    h, w   = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Divide image into a 4x4 grid, compute mean brightness per cell
    cell_h = h // 4
    cell_w = w // 4
    means  = []

    for row in range(4):
        for col in range(4):
            y1 = row * cell_h
            y2 = y1 + cell_h
            x1 = col * cell_w
            x2 = x1 + cell_w
            cell = blurred[y1:y2, x1:x2]
            means.append(float(np.mean(cell)))

    if not means:
        return False, 0, ""

    overall_mean = np.mean(means)
    deviations   = [abs(m - overall_mean) for m in means]
    max_dev      = max(deviations)
    max_idx      = deviations.index(max_dev)

    # A sticker creates a bright/dark patch that stands out
    if max_dev > 65:
        row = max_idx // 4
        col = max_idx % 4
        score = min(int(max_dev * 0.8), 40)
        return (
            True, score,
            f"Brightness anomaly in grid cell ({col},{row}): "
            f"deviation={max_dev:.1f} from mean={overall_mean:.1f}"
        )

    return False, 0, ""


# ══════════════════════════════════════════════════════════════
#  CHECK 3 — Contrast Irregularity
# ══════════════════════════════════════════════════════════════

def _check_contrast_irregularity(gray: np.ndarray) -> tuple[bool, int, str]:
    """
    A sticker over a QR creates a localized region of very different
    texture/contrast compared to the rest of the image.
    Uses local standard deviation as a texture measure.

    Returns: (flagged, score 0-100, evidence_string)
    """
    # Local std-dev map using a sliding window
    mean_sq = cv2.blur(gray.astype(np.float32) ** 2, (15, 15))
    mean    = cv2.blur(gray.astype(np.float32),       (15, 15))
    local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    global_std = float(np.std(gray))
    max_local  = float(np.max(local_std))

    # High local std-dev AND high global std-dev = contrast irregularity
    if global_std > CONTRAST_STD_THRESHOLD and max_local > 90:
        score = min(int(global_std * 0.4), 30)
        return (
            True, score,
            f"Contrast irregularity: global_std={global_std:.1f}, "
            f"max_local_std={max_local:.1f}"
        )

    return False, 0, ""


# ══════════════════════════════════════════════════════════════
#  CHECK 4 — Finder Pattern Obstruction
# ══════════════════════════════════════════════════════════════

def _check_finder_obstruction(gray: np.ndarray) -> tuple[bool, int, str]:
    """
    QR codes have 3 finder patterns (the square corners).
    If a sticker covers part of the QR, we may find fewer than 3.
    Uses morphological operations to locate square marker regions.

    Returns: (flagged, score 0-100, evidence_string)
    """
    # Threshold → find dark square regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to merge nearby dark regions
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Look for nested square contours (finder pattern signature)
    finder_candidates = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if not (200 < area < 15_000):
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) == 4:
            finder_candidates += 1

    # Healthy QR: 3+ finder patterns visible
    # Obstructed QR: fewer than 3 found despite being a QR image
    if 0 < finder_candidates < 3:
        score = (3 - finder_candidates) * 15
        return (
            True, score,
            f"Only {finder_candidates}/3 finder patterns visible — "
            f"possible partial obstruction"
        )

    return False, 0, ""


# ══════════════════════════════════════════════════════════════
#  MAIN ANALYZE FUNCTION
# ══════════════════════════════════════════════════════════════

def analyze_physical(img_bytes: bytes) -> PhysicalLayerResult:
    """
    Run all 4 tamper checks and combine into a single verdict.

    Scoring:
      double_edge      → up to 60 pts
      overlay_patch    → up to 40 pts
      contrast         → up to 30 pts
      finder_obstruct  → up to 30 pts
      Total capped at 100

    Args:
        img_bytes: Raw image bytes

    Returns:
        PhysicalLayerResult with tampered, confidence, evidence
    """
    try:
        cv_img = bytes_to_cv2(img_bytes)
    except ValueError as e:
        logger.error(f"Physical analyzer image load failed: {e}")
        return PhysicalLayerResult(
            tampered   = False,
            confidence = 0,
            evidence   = f"Could not load image for physical analysis: {e}",
        )

    # Convert to grayscale — all checks work on grayscale
    if len(cv_img.shape) == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_img

    logger.info(f"Physical analyzer running on {gray.shape[1]}x{gray.shape[0]} image")

    # ── Run all 4 checks ─────────────────────────────────────
    checks = []

    flag1, score1, evidence1 = _check_double_edge(gray)
    checks.append((flag1, score1, evidence1, "DOUBLE_EDGE"))

    flag2, score2, evidence2 = _check_overlay_patch(gray)
    checks.append((flag2, score2, evidence2, "OVERLAY_PATCH"))

    flag3, score3, evidence3 = _check_contrast_irregularity(gray)
    checks.append((flag3, score3, evidence3, "CONTRAST"))

    flag4, score4, evidence4 = _check_finder_obstruction(gray)
    checks.append((flag4, score4, evidence4, "FINDER_OBSTRUCT"))

    # ── Aggregate ────────────────────────────────────────────
    flags_triggered = [c for c in checks if c[0]]
    total_score     = min(sum(c[1] for c in flags_triggered), 100)

    tampered   = total_score >= TAMPER_CONFIDENCE_THRESHOLD
    confidence = total_score

    # Build evidence string from all triggered checks
    if flags_triggered:
        evidence_parts = [c[2] for c in flags_triggered if c[2]]
        evidence = " | ".join(evidence_parts)
    else:
        evidence = "No tamper signals detected. Single clean boundary. Physical integrity intact."

    logger.info(
        f"Physical analysis complete: tampered={tampered} "
        f"confidence={confidence} flags={[c[3] for c in flags_triggered]}"
    )

    return PhysicalLayerResult(
        tampered   = tampered,
        confidence = confidence,
        evidence   = evidence,
    )