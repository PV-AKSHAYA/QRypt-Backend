"""
tests/test_threat_intel.py
───────────────────────────
Run with: pytest app/tests/test_threat_intel.py -v

All tests use mocks — no real VT API calls made.
This means tests run instantly and work without an API key.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.threat_intel import (
    check_virustotal,
    _derive_reputation,
    _unknown_result,
)
from app.models.response_models import ReputationClass


# ══════════════════════════════════════════════════════════════
#  UNIT TESTS — pure logic, no mocks needed
# ══════════════════════════════════════════════════════════════

class TestDeriveReputation:

    def test_clean_when_zero_malicious(self):
        assert _derive_reputation(0, 0) == ReputationClass.CLEAN

    def test_suspicious_at_one_malicious(self):
        assert _derive_reputation(1, 0) == ReputationClass.SUSPICIOUS

    def test_suspicious_at_two_malicious(self):
        assert _derive_reputation(2, 0) == ReputationClass.SUSPICIOUS

    def test_malicious_at_three_engines(self):
        assert _derive_reputation(3, 0) == ReputationClass.MALICIOUS

    def test_malicious_at_many_engines(self):
        assert _derive_reputation(15, 5) == ReputationClass.MALICIOUS

    def test_suspicious_from_suspicious_count(self):
        # 0 malicious but 3+ suspicious engines → SUSPICIOUS
        assert _derive_reputation(0, 3) == ReputationClass.SUSPICIOUS

    def test_clean_with_low_suspicious(self):
        # 0 malicious, 2 suspicious → still CLEAN
        assert _derive_reputation(0, 2) == ReputationClass.CLEAN


class TestUnknownResult:

    def test_returns_unknown_class(self):
        r = _unknown_result()
        assert r.reputation_class == ReputationClass.UNKNOWN

    def test_returns_zero_counts(self):
        r = _unknown_result("test")
        assert r.malicious     == 0
        assert r.suspicious    == 0
        assert r.total_engines == 0


# ══════════════════════════════════════════════════════════════
#  INTEGRATION TESTS — mocked HTTP responses
# ══════════════════════════════════════════════════════════════

def _mock_submit_response(analysis_id: str = "test-id-123") -> MagicMock:
    """Mock a successful VT URL submission."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "data": {"id": analysis_id}
    }
    return mock


def _mock_poll_response(malicious: int, suspicious: int,
                        harmless: int = 70, status: str = "completed") -> MagicMock:
    """Mock a VT analysis poll response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "data": {
            "attributes": {
                "status": status,
                "stats": {
                    "malicious":  malicious,
                    "suspicious": suspicious,
                    "harmless":   harmless,
                    "undetected": 87 - malicious - suspicious - harmless,
                }
            }
        }
    }
    return mock


class TestCheckVirusTotal:

    @patch("app.services.threat_intel.settings")
    def test_no_api_key_returns_unknown(self, mock_settings):
        """Missing API key → UNKNOWN result, no crash."""
        mock_settings.VIRUSTOTAL_API_KEY = ""
        result = check_virustotal("https://example.com")
        assert result.reputation_class == ReputationClass.UNKNOWN

    @patch("app.services.threat_intel.settings")
    def test_invalid_url_returns_unknown(self, mock_settings):
        """Non-HTTP URL → UNKNOWN result."""
        mock_settings.VIRUSTOTAL_API_KEY = "fake-key"
        result = check_virustotal("not-a-url")
        assert result.reputation_class == ReputationClass.UNKNOWN

    @patch("app.services.threat_intel.requests.post")
    @patch("app.services.threat_intel.requests.get")
    @patch("app.services.threat_intel.settings")
    @patch("app.services.threat_intel.time.sleep")
    def test_clean_url_returns_clean(
        self, mock_sleep, mock_settings, mock_get, mock_post
    ):
        """0 malicious engines → CLEAN."""
        mock_settings.VIRUSTOTAL_API_KEY = "fake-key"
        mock_settings.HTTP_TIMEOUT       = 8

        mock_post.return_value = _mock_submit_response("id-clean")
        mock_get.return_value  = _mock_poll_response(malicious=0, suspicious=0)

        result = check_virustotal("https://example.com")

        assert result.reputation_class == ReputationClass.CLEAN
        assert result.malicious        == 0
        assert result.total_engines    > 0

    @patch("app.services.threat_intel.requests.post")
    @patch("app.services.threat_intel.requests.get")
    @patch("app.services.threat_intel.settings")
    @patch("app.services.threat_intel.time.sleep")
    def test_malicious_url_flagged(
        self, mock_sleep, mock_settings, mock_get, mock_post
    ):
        """4 malicious engines → MALICIOUS."""
        mock_settings.VIRUSTOTAL_API_KEY = "fake-key"
        mock_settings.HTTP_TIMEOUT       = 8

        mock_post.return_value = _mock_submit_response("id-bad")
        mock_get.return_value  = _mock_poll_response(malicious=4, suspicious=2)

        result = check_virustotal("https://phishing-site.xyz")

        assert result.reputation_class == ReputationClass.MALICIOUS
        assert result.malicious        == 4
        assert result.suspicious       == 2

    @patch("app.services.threat_intel.requests.post")
    @patch("app.services.threat_intel.settings")
    def test_rate_limit_returns_unknown(self, mock_settings, mock_post):
        """HTTP 429 rate limit → UNKNOWN, no crash."""
        mock_settings.VIRUSTOTAL_API_KEY = "fake-key"
        mock_settings.HTTP_TIMEOUT       = 8

        mock = MagicMock()
        mock.status_code = 429
        mock_post.return_value = mock

        result = check_virustotal("https://example.com")
        assert result.reputation_class == ReputationClass.UNKNOWN

    @patch("app.services.threat_intel.requests.post")
    @patch("app.services.threat_intel.settings")
    def test_network_error_no_crash(self, mock_settings, mock_post):
        """Network exception → UNKNOWN, never raises."""
        mock_settings.VIRUSTOTAL_API_KEY = "fake-key"
        mock_settings.HTTP_TIMEOUT       = 8
        mock_post.side_effect            = Exception("Network error")

        result = check_virustotal("https://example.com")
        assert result.reputation_class == ReputationClass.UNKNOWN

    @patch("app.services.threat_intel.requests.post")
    @patch("app.services.threat_intel.requests.get")
    @patch("app.services.threat_intel.settings")
    @patch("app.services.threat_intel.time.sleep")
    def test_analysis_still_queued_returns_unknown(
        self, mock_sleep, mock_settings, mock_get, mock_post
    ):
        """If analysis never completes → UNKNOWN."""
        mock_settings.VIRUSTOTAL_API_KEY = "fake-key"
        mock_settings.HTTP_TIMEOUT       = 8

        mock_post.return_value = _mock_submit_response("id-slow")
        # Always return queued status
        mock_get.return_value  = _mock_poll_response(
            malicious=0, suspicious=0, status="queued"
        )

        result = check_virustotal("https://slow-site.com")
        assert result.reputation_class == ReputationClass.UNKNOWN

    @patch("app.services.threat_intel.requests.post")
    @patch("app.services.threat_intel.requests.get")
    @patch("app.services.threat_intel.settings")
    @patch("app.services.threat_intel.time.sleep")
    def test_result_fields_always_valid_types(
        self, mock_sleep, mock_settings, mock_get, mock_post
    ):
        """All output fields must be correct types."""
        mock_settings.VIRUSTOTAL_API_KEY = "fake-key"
        mock_settings.HTTP_TIMEOUT       = 8
        mock_post.return_value = _mock_submit_response()
        mock_get.return_value  = _mock_poll_response(malicious=1, suspicious=0)

        result = check_virustotal("https://example.com")

        assert isinstance(result.malicious,        int)
        assert isinstance(result.suspicious,       int)
        assert isinstance(result.harmless,         int)
        assert isinstance(result.total_engines,    int)
        assert isinstance(result.reputation_class, ReputationClass)