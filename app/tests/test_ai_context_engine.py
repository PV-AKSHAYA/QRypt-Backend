"""
SafeQR — Tests for AI Context Engine
Run offline: pytest tests/test_ai_context_engine.py -v -m "not network"
Run all:     pytest tests/test_ai_context_engine.py -v
"""

import pytest
from app.services.ai_context_engine import (
    _extract_json,
    _validate_and_build,
    _skipped_result,
    _error_result,
    _build_prompt,
)
from app.models.response_models import URLMatchStatus, AIContextResult


# ─────────────────────────────────────────────
# JSON Extraction Tests (no network)
# ─────────────────────────────────────────────

class TestExtractJSON:

    def test_clean_json(self):
        raw = '{"visual_context": "test", "expected_brand": "Google"}'
        result = _extract_json(raw)
        assert result is not None
        assert result["expected_brand"] == "Google"

    def test_json_in_markdown_fences(self):
        raw = '```json\n{"visual_context": "test", "expected_brand": "Apple"}\n```'
        result = _extract_json(raw)
        assert result is not None
        assert result["expected_brand"] == "Apple"

    def test_json_with_preamble(self):
        raw = 'Here is my analysis:\n{"visual_context": "poster", "expected_brand": "PayPal"}'
        result = _extract_json(raw)
        assert result is not None
        assert result["expected_brand"] == "PayPal"

    def test_invalid_json_returns_none(self):
        result = _extract_json("This is just plain text with no JSON")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _extract_json("")
        assert result is None


# ─────────────────────────────────────────────
# Validate and Build Tests (no network)
# ─────────────────────────────────────────────

class TestValidateAndBuild:

    def _sample_data(self):
        return {
            "visual_context": "A PayPal login page flyer",
            "expected_brand": "PayPal",
            "url_match": "MISMATCH",
            "impersonation_probability": 0.92,
            "confidence": 0.85,
            "explanation": "URL domain does not match PayPal official domain"
        }

    def test_valid_data_builds_correctly(self):
        result = _validate_and_build(self._sample_data())
        assert isinstance(result, AIContextResult)
        assert result.expected_brand == "PayPal"
        assert result.url_match == URLMatchStatus.MISMATCH
        assert result.impersonation_probability == 0.92

    def test_invalid_url_match_defaults_to_unknown(self):
        data = self._sample_data()
        data["url_match"] = "INVALID_VALUE"
        result = _validate_and_build(data)
        assert result.url_match == URLMatchStatus.UNKNOWN

    def test_impersonation_clamped_above_1(self):
        data = self._sample_data()
        data["impersonation_probability"] = 5.0
        result = _validate_and_build(data)
        assert result.impersonation_probability <= 1.0

    def test_impersonation_clamped_below_0(self):
        data = self._sample_data()
        data["impersonation_probability"] = -0.5
        result = _validate_and_build(data)
        assert result.impersonation_probability >= 0.0

    def test_confidence_clamped(self):
        data = self._sample_data()
        data["confidence"] = 99.9
        result = _validate_and_build(data)
        assert result.confidence <= 1.0

    def test_missing_fields_get_defaults(self):
        result = _validate_and_build({})
        assert result.visual_context != ""
        assert result.expected_brand != ""
        assert result.url_match == URLMatchStatus.UNKNOWN

    def test_url_match_case_insensitive(self):
        data = self._sample_data()
        data["url_match"] = "match"
        result = _validate_and_build(data)
        assert result.url_match == URLMatchStatus.MATCH


# ─────────────────────────────────────────────
# Helper Result Tests (no network)
# ─────────────────────────────────────────────

class TestHelperResults:

    def test_skipped_result_is_skipped(self):
        result = _skipped_result("No API key")
        assert result.skipped is True
        assert result.error is None

    def test_error_result_not_skipped(self):
        result = _error_result("API failed")
        assert result.skipped is False
        assert result.error == "API failed"

    def test_skipped_has_zero_confidence(self):
        result = _skipped_result("reason")
        assert result.confidence == 0.0

    def test_error_has_unknown_url_match(self):
        result = _error_result("reason")
        assert result.url_match == URLMatchStatus.UNKNOWN


# ─────────────────────────────────────────────
# Prompt Builder Tests (no network)
# ─────────────────────────────────────────────

class TestBuildPrompt:

    def test_prompt_contains_url(self):
        prompt = _build_prompt("https://fake-paypal.tk/login")
        assert "https://fake-paypal.tk/login" in prompt

    def test_prompt_contains_json_structure(self):
        prompt = _build_prompt("https://example.com")
        assert "visual_context" in prompt
        assert "expected_brand" in prompt
        assert "impersonation_probability" in prompt

    def test_prompt_with_ocr_text(self):
        prompt = _build_prompt("https://example.com", extracted_text="Scan to win a prize!")
        assert "Scan to win a prize!" in prompt

    def test_prompt_without_ocr_text(self):
        prompt = _build_prompt("https://example.com", extracted_text=None)
        assert isinstance(prompt, str)
        assert len(prompt) > 100


# ─────────────────────────────────────────────
# Live Gemini Test (requires network + API key)
# ─────────────────────────────────────────────

@pytest.mark.network
class TestLiveGemini:

    @pytest.mark.asyncio
    async def test_clean_qr_image(self):
        import os, io
        import qrcode
        from app.services.ai_context_engine import analyze_ai_context

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            pytest.skip("GEMINI_API_KEY not set")

        # Generate a simple QR image
        qr = qrcode.QRCode(box_size=10, border=4)
        qr.add_data("https://example.com")
        qr.make(fit=True)
        buf = io.BytesIO()
        qr.make_image().save(buf, format="PNG")
        image_bytes = buf.getvalue()

        result = await analyze_ai_context(
            image_bytes=image_bytes,
            final_url="https://example.com",
            api_key=api_key,
        )

        assert result.skipped is False
        assert result.visual_context != ""
        assert result.url_match in (URLMatchStatus.MATCH, URLMatchStatus.UNKNOWN)
        assert 0.0 <= result.impersonation_probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0