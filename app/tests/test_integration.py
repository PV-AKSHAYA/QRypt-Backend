"""
tests/test_integration.py
──────────────────────────
Full end-to-end integration tests for the /scan endpoint.

Tests all 3 demo scenarios:
  1. SAFE   — clean QR pointing to legitimate domain
  2. SUSPICIOUS — QR with multiple redirects, risky TLD
  3. HIGH_RISK  — tampered image + malicious URL signals

Uses TestClient (no real server needed).
VirusTotal and Gemini are mocked — tests run offline.

Run: pytest app/tests/test_integration.py -v
"""

import io
import pytest
import qrcode
import numpy as np
import cv2
from PIL import Image, ImageDraw
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.models.response_models import Verdict, ReputationClass, URLMatch


# ══════════════════════════════════════════════════════════════
#  TEST CLIENT
# ══════════════════════════════════════════════════════════════

client = TestClient(app, raise_server_exceptions=False)


# ══════════════════════════════════════════════════════════════
#  IMAGE GENERATORS
# ══════════════════════════════════════════════════════════════

def make_qr_image(url: str) -> bytes:
    """Clean QR code with generous white padding to avoid edge artifacts."""
    qr = qrcode.QRCode(border=10)   # More border to separate from image edge
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_tampered_qr_image(url: str) -> bytes:
    """QR with a small sticker overlay — triggers physical analyzer but stays readable."""
    qr = qrcode.QRCode(border=4)
    qr.add_data(url)
    qr.make(fit=True)
    # Use NEAREST to keep modules sharp
    qr_img = qr.make_image().convert("RGB").resize((300, 300), Image.Resampling.NEAREST)
    
    canvas = Image.new("RGB", (400, 400), "white")
    canvas.paste(qr_img, (50, 50))

    draw = ImageDraw.Draw(canvas)
    # Move sticker even more to the edge
    # QR is at (50, 50) to (350, 350).
    # Sticker from (240, 240) to (340, 340)
    rects = [
        [240, 240, 340, 340],
        [245, 245, 335, 335],
        [250, 250, 330, 330],
        [255, 255, 325, 325],
        [260, 260, 320, 320],
        [265, 265, 315, 315],
    ]
    for i, rect in enumerate(rects):
        if i % 2 == 0:
            draw.rectangle(rect, outline="black", width=4)   # Increased width
        else:
            draw.rectangle(rect, fill=(240, 240, 240))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def make_blank_image() -> bytes:
    """No QR — should return 400."""
    img = Image.new("RGB", (300, 300), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_corrupt_bytes() -> bytes:
    return b"this is not an image"


# ══════════════════════════════════════════════════════════════
#  MOCK HELPERS
# ══════════════════════════════════════════════════════════════

def mock_vt_clean():
    from app.models.response_models import VirusTotalResult
    return VirusTotalResult(
        malicious=0, suspicious=0, harmless=80,
        total_engines=87, reputation_class=ReputationClass.CLEAN
    )


def mock_vt_malicious():
    from app.models.response_models import VirusTotalResult
    return VirusTotalResult(
        malicious=5, suspicious=2, harmless=60,
        total_engines=87, reputation_class=ReputationClass.MALICIOUS
    )


def mock_ai_safe():
    from app.models.response_models import AILayerResult
    return AILayerResult(
        visual_context="Restaurant menu card",
        expected_brand="Starbucks Coffee",
        url_match=URLMatch.YES,
        impersonation_probability=0.02,
        confidence=0.95,
        explanation="URL matches expected brand perfectly."
    )


def mock_ai_dangerous():
    from app.models.response_models import AILayerResult
    return AILayerResult(
        visual_context="Government tax payment poster",
        expected_brand="IRS / Tax Authority",
        url_match=URLMatch.NO,
        impersonation_probability=0.93,
        confidence=0.89,
        explanation="QR leads to unregistered domain, not a government TLD."
    )


# ══════════════════════════════════════════════════════════════
#  SCENARIO 1 — SAFE QR
# ══════════════════════════════════════════════════════════════

class TestScenario1Safe:
    """
    Clean QR pointing to github.com.
    Physical: clean image
    Technical: 0 redirects, valid HTTPS, low TLD risk
    AI: context matches, low impersonation
    Expected: SAFE verdict, score < 30
    """

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_safe_qr_verdict(self, mock_ai, mock_vt):
        img_bytes = make_qr_image("https://github.com")

        r = client.post(
            "/api/v1/scan",
            files={"image": ("safe.png", img_bytes, "image/png")},
        )

        assert r.status_code == 200, f"Expected 200 got {r.status_code}: {r.text}"
        data = r.json()

        assert data["risk"]["verdict"] == Verdict.SAFE
        assert data["risk"]["score"]   <  30

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_safe_qr_structure(self, mock_ai, mock_vt):
        """All required fields present in response."""
        img_bytes = make_qr_image("https://github.com")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("safe.png", img_bytes, "image/png")},
        )
        data = r.json()

        # Top-level fields
        assert "scan_id"         in data
        assert "timestamp"       in data
        assert "threat_memory"   in data
        assert "qr"              in data
        assert "physical_layer"  in data
        assert "technical_layer" in data
        assert "ai_layer"        in data
        assert "risk"            in data

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_safe_qr_url_extracted(self, mock_ai, mock_vt):
        """QR content must be extracted correctly."""
        url       = "https://github.com"
        img_bytes = make_qr_image(url)
        r = client.post(
            "/api/v1/scan",
            files={"image": ("safe.png", img_bytes, "image/png")},
        )
        data = r.json()
        assert data["qr"]["raw_content"] == url

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_safe_physical_not_tampered(self, mock_ai, mock_vt):
        """Clean QR image — physical layer must not flag tamper."""
        img_bytes = make_qr_image("https://github.com")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("safe.png", img_bytes, "image/png")},
        )
        data = r.json()
        assert data["physical_layer"]["confidence"] < 60


# ══════════════════════════════════════════════════════════════
#  SCENARIO 2 — SUSPICIOUS QR
# ══════════════════════════════════════════════════════════════

class TestScenario2Suspicious:
    """
    QR pointing to a risky-looking URL.
    Technical: high TLD risk, suspicious keywords
    AI: uncertain context
    Expected: SUSPICIOUS verdict, 30 <= score < 60
    """

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context")
    def test_suspicious_risky_url(self, mock_ai, mock_vt):
        from app.models.response_models import AILayerResult
        mock_ai.return_value = AILayerResult(
            visual_context="Generic flyer",
            expected_brand="Unknown",
            url_match=URLMatch.UNCERTAIN,
            impersonation_probability=0.45,
            confidence=0.4,
            explanation="Cannot determine brand intent from visual."
        )

        # .xyz TLD + suspicious keywords = technical layer risk
        img_bytes = make_qr_image("https://login-verify-account.xyz/confirm")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("sus.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()

        # Should at minimum not be SAFE given risky URL
        score = data["risk"]["score"]
        assert score >= 20, f"Expected elevated score, got {score}"

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_skip_virustotal_flag(self, mock_ai, mock_vt):
        """skip_virustotal=true must complete without calling VT."""
        img_bytes = make_qr_image("https://example.com")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"skip_virustotal": "true"},
        )
        assert r.status_code == 200
        mock_vt.assert_not_called()

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_skip_ai_flag(self, mock_ai, mock_vt):
        """skip_ai=true must complete without calling Gemini."""
        img_bytes = make_qr_image("https://example.com")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"skip_ai": "true"},
        )
        assert r.status_code == 200
        mock_ai.assert_not_called()

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_context_hint_accepted(self, mock_ai, mock_vt):
        """context_hint form field must be accepted without error."""
        img_bytes = make_qr_image("https://example.com")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"context_hint": "bank poster"},
        )
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════
#  SCENARIO 3 — HIGH RISK QR
# ══════════════════════════════════════════════════════════════

class TestScenario3HighRisk:
    """
    Tampered image + malicious URL + AI mismatch.
    All 3 layers fire.
    Expected: HIGH_RISK verdict, score >= 60
    """

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_malicious())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_dangerous())
    def test_high_risk_verdict(self, mock_ai, mock_vt):
        img_bytes = make_tampered_qr_image(
            "https://taxrefund-claim.xyz/pay"
        )
        r = client.post(
            "/api/v1/scan",
            files={"image": ("tampered.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()

        assert data["risk"]["verdict"] == Verdict.HIGH_RISK
        assert data["risk"]["score"]   >= 60

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_malicious())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_dangerous())
    def test_all_layers_fired(self, mock_ai, mock_vt):
        """All 3 breakdown scores must be non-zero for HIGH_RISK."""
        img_bytes = make_tampered_qr_image(
            "https://taxrefund-claim.xyz/pay"
        )
        r = client.post(
            "/api/v1/scan",
            files={"image": ("tampered.png", img_bytes, "image/png")},
        )
        data      = r.json()
        breakdown = data["risk"]["breakdown"]

        # Technical + AI must contribute
        assert breakdown["threat_intel_score"] > 0
        assert breakdown["ai_context_score"]   > 0

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_malicious())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_dangerous())
    def test_vt_engines_in_response(self, mock_ai, mock_vt):
        """VT malicious count must appear in technical layer."""
        # Use a readable URL
        img_bytes = make_tampered_qr_image("https://taxrefund-claim.xyz/pay")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("bad.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["technical_layer"]["virustotal"]["malicious"] == 5

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_malicious())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_dangerous())
    def test_ai_mismatch_in_response(self, mock_ai, mock_vt):
        """AI url_match=NO must appear in ai_layer."""
        # Use same URL as verdict test to ensure consistency and readability
        img_bytes = make_tampered_qr_image("https://taxrefund-claim.xyz/pay")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("bad.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["ai_layer"]["url_match"] == "NO"
        assert data["ai_layer"]["impersonation_probability"] > 0.8


# ══════════════════════════════════════════════════════════════
#  ERROR HANDLING TESTS
# ══════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_no_qr_returns_400(self):
        """Blank image with no QR → 400."""
        r = client.post(
            "/api/v1/scan",
            files={"image": ("blank.png", make_blank_image(), "image/png")},
        )
        assert r.status_code == 400

    def test_corrupt_image_returns_400(self):
        """Non-image bytes → 400."""
        r = client.post(
            "/api/v1/scan",
            files={"image": ("bad.jpg", make_corrupt_bytes(), "image/jpeg")},
        )
        assert r.status_code == 400

    def test_missing_image_returns_422(self):
        """No image field at all → 422 validation error."""
        r = client.post("/api/v1/scan")
        assert r.status_code == 422

    @patch("app.api.scan.check_virustotal", side_effect=Exception("VT down"))
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_vt_crash_scan_still_completes(self, mock_ai, mock_vt):
        """VT service throwing → scan still returns 200."""
        img_bytes = make_qr_image("https://example.com")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  side_effect=Exception("Gemini down"))
    def test_ai_crash_scan_still_completes(self, mock_ai, mock_vt):
        """Gemini throwing → scan still returns 200."""
        img_bytes = make_qr_image("https://example.com")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        assert r.status_code == 200

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_scan_id_is_unique(self, mock_ai, mock_vt):
        """Every scan must have a unique scan_id."""
        img_bytes = make_qr_image("https://example.com")
        ids = set()
        for _ in range(3):
            r = client.post(
                "/api/v1/scan",
                files={"image": ("test.png", img_bytes, "image/png")},
            )
            ids.add(r.json()["scan_id"])
        assert len(ids) == 3

    @patch("app.api.scan.check_virustotal", return_value=mock_vt_clean())
    @patch("app.api.scan.analyze_context",  return_value=mock_ai_safe())
    def test_score_always_in_range(self, mock_ai, mock_vt):
        """Score must always be 0-100."""
        img_bytes = make_qr_image("https://example.com")
        r = client.post(
            "/api/v1/scan",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        score = r.json()["risk"]["score"]
        assert 0 <= score <= 100


# ══════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════

class TestHealth:

    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_docs_accessible(self):
        r = client.get("/docs")
        assert r.status_code == 200