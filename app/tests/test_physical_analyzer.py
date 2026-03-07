"""
SafeQR — Tests for Physical Analyzer
Run with: pytest tests/test_physical_analyzer.py -v
"""

import io
import pytest
import numpy as np
import cv2
import qrcode

from app.services.physical_analyzer import analyze_physical
from app.models.response_models import BoundingBox, TamperStatus


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_clean_qr_bytes() -> tuple:
    """Returns (image_bytes, BoundingBox) for a clean QR."""
    qr = qrcode.QRCode(box_size=10, border=4)
    qr.add_data("https://example.com")
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    raw = buf.getvalue()

    # pil gives us white border, QR starts roughly at 40px
    bbox = BoundingBox(x=39, y=39, width=292, height=292)
    return raw, bbox


def make_tampered_qr_bytes() -> tuple:
    """
    Returns image bytes where a bright rectangle is pasted
    over the QR code — simulates a sticker attack.
    """
    raw, bbox = make_clean_qr_bytes()

    # Load and draw overlay patch
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Draw a filled white rectangle over 40% of the QR
    ox = bbox.x + 20
    oy = bbox.y + 20
    ow = int(bbox.width * 0.4)
    oh = int(bbox.height * 0.4)
    cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (255, 255, 255), -1)
    # Draw a border to make it look like a sticker
    cv2.rectangle(img, (ox, oy), (ox + ow, oy + oh), (180, 180, 180), 2)

    _, buf = cv2.imencode(".png", img)
    return buf.tobytes(), bbox


def make_blank_image_bytes() -> bytes:
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestPhysicalAnalyzer:

    def test_clean_qr_returns_clean_status(self):
        raw, bbox = make_clean_qr_bytes()
        result = analyze_physical(raw, bbox)
        # Clean QR should have low confidence score
        assert result.confidence < 60
        assert result.status in (TamperStatus.CLEAN, TamperStatus.SUSPICIOUS)

    def test_tampered_qr_detected(self):
        raw, bbox = make_tampered_qr_bytes()
        result = analyze_physical(raw, bbox)
        # Overlay patch should push score up
        assert result.checks["overlay_patch"]["score"] > 0
        assert result.evidence != ""

    def test_result_has_all_four_checks(self):
        raw, bbox = make_clean_qr_bytes()
        result = analyze_physical(raw, bbox)
        assert "edge_anomaly"  in result.checks
        assert "overlay_patch" in result.checks
        assert "obstruction"   in result.checks
        assert "contrast"      in result.checks

    def test_confidence_in_range(self):
        raw, bbox = make_clean_qr_bytes()
        result = analyze_physical(raw, bbox)
        assert 0 <= result.confidence <= 100

    def test_no_bbox_runs_without_crash(self):
        raw, _ = make_clean_qr_bytes()
        result = analyze_physical(raw, bbox=None)
        assert result is not None
        assert result.evidence != ""

    def test_corrupt_image_returns_gracefully(self):
        result = analyze_physical(b"not_an_image", bbox=None)
        assert result.tampered is False
        assert "failed" in result.evidence.lower()

    def test_empty_bytes_returns_gracefully(self):
        result = analyze_physical(b"", bbox=None)
        assert result.tampered is False

    def test_evidence_string_not_empty(self):
        raw, bbox = make_clean_qr_bytes()
        result = analyze_physical(raw, bbox)
        assert isinstance(result.evidence, str)
        assert len(result.evidence) > 0

    def test_checks_have_score_and_finding(self):
        raw, bbox = make_clean_qr_bytes()
        result = analyze_physical(raw, bbox)
        for check_name, check_data in result.checks.items():
            assert "score"   in check_data, f"{check_name} missing score"
            assert "finding" in check_data, f"{check_name} missing finding"
            assert 0 <= check_data["score"] <= 100