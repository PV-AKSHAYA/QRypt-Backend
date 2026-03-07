"""
tests/test_physical_analyzer.py
─────────────────────────────────
Run with: pytest app/tests/test_physical_analyzer.py -v

Tests:
  1. Clean QR image       → tampered=False, confidence low
  2. Synthetically tampered image → tampered=True, confidence high
  3. Corrupt bytes        → returns safe default, no crash
  4. Empty bytes          → returns safe default, no crash
  5. Output fields always present and valid types
  6. Confidence always 0-100
"""

import pytest
import qrcode
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.services.physical_analyzer import analyze_physical


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def make_clean_qr_bytes(url: str = "https://example.com") -> bytes:
    """Generate a plain QR code image — no tampering."""
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_tampered_qr_bytes() -> bytes:
    """
    Synthetically create a tampered QR image:
    - Generate a real QR code
    - Paste a bright white rectangle over part of it (simulates sticker)
    - Draw a second rectangle border (simulates double-edge)
    """
    # Base QR
    qr_pil = qrcode.make("https://legitimate-bank.com").resize((300, 300))
    canvas = Image.new("RGB", (400, 400), "white")
    canvas.paste(qr_pil, (50, 50))

    # Simulate sticker overlay — bright patch with its own border
    draw = ImageDraw.Draw(canvas)

    # Outer border of sticker (creates double-edge)
    draw.rectangle([100, 100, 260, 260], outline="black", width=4)
    # Sticker fill (brightness anomaly)
    draw.rectangle([104, 104, 256, 256], fill=(240, 240, 240))
    # Inner content of sticker (nested rectangle = double-edge signature)
    draw.rectangle([120, 120, 240, 240], outline="black", width=3)
    draw.rectangle([130, 130, 230, 230], fill=(200, 200, 200))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def make_noisy_qr_bytes() -> bytes:
    """QR with heavy noise — stress test for false positives."""
    qr_pil = qrcode.make("https://noisy-test.com").resize((300, 300))
    cv_img = np.array(qr_pil.convert("RGB"))

    # Add gaussian noise
    noise  = np.random.normal(0, 15, cv_img.shape).astype(np.int16)
    noisy  = np.clip(cv_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    _, buf = cv2.imencode(".png", noisy)
    return buf.tobytes()


# ══════════════════════════════════════════════════════════════
#  TESTS
# ══════════════════════════════════════════════════════════════

class TestPhysicalAnalyzer:

    def test_clean_qr_not_flagged(self):
        """A genuine QR code on white background should not be flagged."""
        result = analyze_physical(make_clean_qr_bytes())
        # Allow some tolerance — clean QR should be well below threshold
        assert result.confidence < 80, (
            f"Clean QR flagged with confidence {result.confidence}"
        )

    def test_tampered_qr_flagged(self):
        """Synthetically tampered image must be flagged as tampered."""
        result = analyze_physical(make_tampered_qr_bytes())
        assert result.tampered is True, "Tampered QR not detected"
        assert result.confidence >= 35, (
            f"Confidence too low for obvious tamper: {result.confidence}"
        )

    def test_tampered_evidence_not_empty(self):
        """Tampered result must include a non-empty evidence string."""
        result = analyze_physical(make_tampered_qr_bytes())
        assert result.evidence
        assert len(result.evidence) > 10

    def test_clean_evidence_says_intact(self):
        """Clean result evidence should mention no tampering found."""
        result = analyze_physical(make_clean_qr_bytes())
        assert "intact" in result.evidence.lower() or \
               "no tamper" in result.evidence.lower() or \
               "clean" in result.evidence.lower()

    def test_corrupt_bytes_no_crash(self):
        """Corrupt input must return a safe default — never raise."""
        result = analyze_physical(b"not an image")
        assert result.tampered   is False
        assert result.confidence == 0
        assert result.evidence   != ""

    def test_empty_bytes_no_crash(self):
        """Empty input must return safe default — never raise."""
        result = analyze_physical(b"")
        assert result.tampered   is False
        assert result.confidence == 0

    def test_confidence_always_in_range(self):
        """Confidence must always be 0-100."""
        for img_b in [
            make_clean_qr_bytes(),
            make_tampered_qr_bytes(),
            make_noisy_qr_bytes(),
        ]:
            result = analyze_physical(img_b)
            assert 0 <= result.confidence <= 100, (
                f"Confidence out of range: {result.confidence}"
            )

    def test_output_types_correct(self):
        """All output fields must be correct types."""
        result = analyze_physical(make_clean_qr_bytes())
        assert isinstance(result.tampered,   bool)
        assert isinstance(result.confidence, int)
        assert isinstance(result.evidence,   str)

    def test_deterministic(self):
        """Same image must always return same result."""
        img_b   = make_tampered_qr_bytes()
        result1 = analyze_physical(img_b)
        result2 = analyze_physical(img_b)
        assert result1.tampered   == result2.tampered
        assert result1.confidence == result2.confidence
        assert result1.evidence   == result2.evidence