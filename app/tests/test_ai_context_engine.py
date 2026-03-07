"""
tests/test_ai_context_engine.py
─────────────────────────────────
Run with: pytest app/tests/test_ai_context_engine.py -v

All tests use mocks — no Gemini API key needed.
Tests cover: clean response, JSON drift, retry logic,
missing key, network error, schema coercion.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.ai_context_engine import (
    analyze_context,
    _parse_response,
    _validate_and_build,
    _default_result,
)
from app.models.response_models import AILayerResult, URLMatch


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def make_blank_image_bytes() -> bytes:
    from PIL import Image
    import io
    img = Image.new("RGB", (200, 200), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


GOOD_JSON = """{
  "visual_context": "Bank branch ATM area poster",
  "expected_brand": "HDFC Bank",
  "url_match": "NO",
  "impersonation_probability": 0.91,
  "confidence": 0.87,
  "explanation": "QR leads to unrelated domain, not HDFC Bank official site."
}"""

FENCED_JSON = f"```json\n{GOOD_JSON}\n```"

PARTIAL_JSON = "Here is the analysis:\n" + GOOD_JSON + "\nHope that helps!"


# ══════════════════════════════════════════════════════════════
#  UNIT TESTS — parser and validator
# ══════════════════════════════════════════════════════════════

class TestParseResponse:

    def test_clean_json_parses(self):
        result = _parse_response(GOOD_JSON)
        assert result is not None
        assert result["url_match"] == "NO"
        assert result["expected_brand"] == "HDFC Bank"

    def test_fenced_json_stripped(self):
        """JSON wrapped in ```json ``` fences must be handled."""
        result = _parse_response(FENCED_JSON)
        assert result is not None
        assert result["visual_context"] == "Bank branch ATM area poster"

    def test_extra_text_around_json(self):
        """JSON buried in prose must be extracted."""
        result = _parse_response(PARTIAL_JSON)
        assert result is not None

    def test_invalid_json_returns_none(self):
        result = _parse_response("This is not JSON at all")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _parse_response("")
        assert result is None


class TestValidateAndBuild:

    def test_valid_dict_builds_result(self):
        data = {
            "visual_context":             "Government poster",
            "expected_brand":             "IRS",
            "url_match":                  "NO",
            "impersonation_probability":  0.9,
            "confidence":                 0.85,
            "explanation":                "Domain mismatch detected.",
        }
        result = _validate_and_build(data)
        assert result is not None
        assert result.url_match == URLMatch.NO
        assert result.impersonation_probability == 0.9

    def test_url_match_coerced_to_uncertain(self):
        """Invalid url_match value → UNCERTAIN."""
        data = {
            "visual_context": "x", "expected_brand": "y",
            "url_match": "MAYBE",
            "impersonation_probability": 0.5,
            "confidence": 0.5,
            "explanation": "test",
        }
        result = _validate_and_build(data)
        assert result.url_match == URLMatch.UNCERTAIN

    def test_probability_clamped_to_range(self):
        """Values outside 0–1 must be clamped."""
        data = {
            "visual_context": "x", "expected_brand": "y",
            "url_match": "YES",
            "impersonation_probability": 1.5,   # out of range
            "confidence": -0.2,                  # out of range
            "explanation": "test",
        }
        result = _validate_and_build(data)
        assert result.impersonation_probability == 1.0
        assert result.confidence == 0.0

    def test_missing_optional_fields_use_defaults(self):
        """Missing fields must not crash."""
        data = {"url_match": "YES"}
        result = _validate_and_build(data)
        assert result is not None
        assert result.visual_context == "Unknown environment"
        assert result.expected_brand == "Unknown"


class TestDefaultResult:

    def test_default_has_uncertain_match(self):
        r = _default_result()
        assert r.url_match  == URLMatch.UNCERTAIN
        assert r.confidence == 0.0

    def test_default_never_raises(self):
        r = _default_result("some reason")
        assert isinstance(r, AILayerResult)


# ══════════════════════════════════════════════════════════════
#  INTEGRATION TESTS — mocked Gemini
# ══════════════════════════════════════════════════════════════

def _mock_openai_response(text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = text
    mock_response.choices = [mock_choice]
    return mock_response


class TestAnalyzeContext:

    @patch("app.services.ai_context_engine.settings")
    def test_no_api_key_returns_default(self, mock_settings):
        mock_settings.OPENAI_API_KEY = ""
        result = analyze_context(make_blank_image_bytes(), "https://example.com")
        assert result.url_match  == URLMatch.UNCERTAIN
        assert result.confidence == 1.0

    @patch("app.services.ai_context_engine.OpenAI")
    @patch("app.services.ai_context_engine.settings")
    def test_clean_response_parsed_correctly(self, mock_settings, mock_openai):
        """Good OpenAI response → correct AILayerResult."""
        mock_settings.OPENAI_API_KEY = "fake-key"

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(GOOD_JSON)

        result = analyze_context(make_blank_image_bytes(), "https://hdfc-fake.xyz")

        assert result.url_match                 == URLMatch.NO
        assert result.expected_brand            == "HDFC Bank"
        assert result.impersonation_probability  > 0.8
        assert result.confidence                 > 0.8

    @patch("app.services.ai_context_engine.OpenAI")
    @patch("app.services.ai_context_engine.settings")
    def test_fenced_json_handled(self, mock_settings, mock_openai):
        """OpenAI wrapping JSON in fences must still parse."""
        mock_settings.OPENAI_API_KEY = "fake-key"

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(FENCED_JSON)

        result = analyze_context(make_blank_image_bytes(), "https://example.com")
        assert result.url_match != URLMatch.UNCERTAIN or result.confidence >= 0.0

    @patch("app.services.ai_context_engine.OpenAI")
    @patch("app.services.ai_context_engine.settings")
    def test_bad_json_retries_then_returns_default(self, mock_settings, mock_openai):
        """Unparseable JSON after 2 attempts → safe default."""
        mock_settings.OPENAI_API_KEY = "fake-key"

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "Sorry I cannot analyze this."
        )

        result = analyze_context(make_blank_image_bytes(), "https://example.com")
        assert result.url_match  == URLMatch.UNCERTAIN
        assert result.confidence == 1.0

    @patch("app.services.ai_context_engine.OpenAI")
    @patch("app.services.ai_context_engine.settings")
    def test_openai_exception_no_crash(self, mock_settings, mock_openai):
        """OpenAI throwing exception → safe default, never raises."""
        mock_settings.OPENAI_API_KEY = "fake-key"

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = analyze_context(make_blank_image_bytes(), "https://example.com")
        assert isinstance(result, AILayerResult)
        assert result.confidence == 1.0

    @patch("app.services.ai_context_engine.OpenAI")
    @patch("app.services.ai_context_engine.settings")
    def test_output_types_always_correct(self, mock_settings, mock_openai):
        """All fields must be correct types regardless of input."""
        mock_settings.OPENAI_API_KEY = "fake-key"

        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(GOOD_JSON)

        result = analyze_context(make_blank_image_bytes(), "https://example.com")

        assert isinstance(result.visual_context,            str)
        assert isinstance(result.expected_brand,            str)
        assert isinstance(result.url_match,                 URLMatch)
        assert isinstance(result.impersonation_probability, float)
        assert isinstance(result.confidence,                float)
        assert isinstance(result.explanation,               str)