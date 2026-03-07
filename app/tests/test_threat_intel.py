"""
SafeQR — Tests for Threat Intel
Run offline tests: pytest tests/test_threat_intel.py -v -m "not network"
Run all tests:     pytest tests/test_threat_intel.py -v
"""

import pytest
from app.services.threat_intel import (
    classify_reputation,
    reputation_to_score,
    _url_to_vt_id,
    _error_result,
)
from app.models.response_models import ReputationClass


# ─────────────────────────────────────────────
# Reputation Classifier Tests (no network)
# ─────────────────────────────────────────────

class TestClassifyReputation:

    def test_two_malicious_is_malicious(self):
        assert classify_reputation(2, 0, 70) == ReputationClass.MALICIOUS

    def test_one_malicious_is_suspicious(self):
        assert classify_reputation(1, 0, 70) == ReputationClass.SUSPICIOUS

    def test_three_suspicious_is_suspicious(self):
        assert classify_reputation(0, 3, 70) == ReputationClass.SUSPICIOUS

    def test_clean_result(self):
        assert classify_reputation(0, 0, 70) == ReputationClass.CLEAN

    def test_zero_engines_is_unknown(self):
        assert classify_reputation(0, 0, 0) == ReputationClass.UNKNOWN

    def test_high_malicious_count(self):
        assert classify_reputation(45, 5, 72) == ReputationClass.MALICIOUS

    def test_two_suspicious_still_clean(self):
        # 2 suspicious is below threshold of 3
        assert classify_reputation(0, 2, 70) == ReputationClass.CLEAN


# ─────────────────────────────────────────────
# Reputation Score Tests (no network)
# ─────────────────────────────────────────────

class TestReputationToScore:

    def test_clean_is_low_score(self):
        assert reputation_to_score(ReputationClass.CLEAN) <= 10

    def test_malicious_is_high_score(self):
        assert reputation_to_score(ReputationClass.MALICIOUS) >= 90

    def test_suspicious_is_medium(self):
        score = reputation_to_score(ReputationClass.SUSPICIOUS)
        assert 50 <= score <= 80

    def test_unknown_is_moderate(self):
        score = reputation_to_score(ReputationClass.UNKNOWN)
        assert 20 <= score <= 50

    def test_all_classes_in_range(self):
        for cls in ReputationClass:
            score = reputation_to_score(cls)
            assert 0 <= score <= 100, f"{cls} score out of range: {score}"


# ─────────────────────────────────────────────
# URL ID Encoding Tests (no network)
# ─────────────────────────────────────────────

class TestURLToVTId:

    def test_returns_string(self):
        result = _url_to_vt_id("https://example.com")
        assert isinstance(result, str)

    def test_no_padding(self):
        result = _url_to_vt_id("https://example.com")
        assert "=" not in result

    def test_deterministic(self):
        url = "https://example.com/test"
        assert _url_to_vt_id(url) == _url_to_vt_id(url)

    def test_different_urls_different_ids(self):
        id1 = _url_to_vt_id("https://example.com")
        id2 = _url_to_vt_id("https://malicious.tk")
        assert id1 != id2


# ─────────────────────────────────────────────
# Error Result Tests (no network)
# ─────────────────────────────────────────────

class TestErrorResult:

    def test_error_result_not_queried(self):
        result = _error_result("test error")
        assert result.queried is False

    def test_error_result_unknown_reputation(self):
        result = _error_result("test error")
        assert result.reputation_class == ReputationClass.UNKNOWN

    def test_error_result_preserves_flag_count(self):
        result = _error_result("test error", previous_flag_count=3)
        assert result.previous_flag_count == 3

    def test_error_result_has_error_message(self):
        result = _error_result("something went wrong")
        assert result.error == "something went wrong"


# ─────────────────────────────────────────────
# Live VT API Test (requires network + API key)
# ─────────────────────────────────────────────

@pytest.mark.network
class TestLiveVT:

    @pytest.mark.asyncio
    async def test_clean_url(self):
        import os
        from app.services.threat_intel import check_threat_intel
        api_key = os.getenv("VT_API_KEY", "")
        if not api_key:
            pytest.skip("VT_API_KEY not set")

        result = await check_threat_intel("https://example.com", api_key)
        assert result.queried is True
        assert result.total_engines > 0
        assert result.reputation_class in (ReputationClass.CLEAN, ReputationClass.UNKNOWN)