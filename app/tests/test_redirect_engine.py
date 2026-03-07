"""
SafeQR — Tests for Redirect Engine
Run with: pytest tests/test_redirect_engine.py -v

Note: Tests marked with @pytest.mark.network require internet access.
      Run offline tests only with: pytest -m "not network"
"""

import pytest
import asyncio

from app.services.redirect_engine import (
    analyze_url,
    is_shortened_url,
    score_tld_risk,
    score_suspicious_keywords,
    compute_domain_entropy,
)


# ─────────────────────────────────────────────
# Shortener Detection Tests (no network)
# ─────────────────────────────────────────────

class TestShortenerDetection:

    def test_bitly_detected(self):
        assert is_shortened_url("https://bit.ly/abc123") is True

    def test_tinyurl_detected(self):
        assert is_shortened_url("https://tinyurl.com/xyz") is True

    def test_normal_url_not_shortened(self):
        assert is_shortened_url("https://example.com/page") is False

    def test_google_not_shortened(self):
        assert is_shortened_url("https://google.com") is False


# ─────────────────────────────────────────────
# TLD Risk Scoring Tests (no network)
# ─────────────────────────────────────────────

class TestTLDRiskScoring:

    def test_com_is_low_risk(self):
        assert score_tld_risk("com") < 20

    def test_tk_is_high_risk(self):
        assert score_tld_risk("tk") > 80

    def test_xyz_is_medium_high(self):
        assert score_tld_risk("xyz") > 50

    def test_unknown_tld_returns_default(self):
        score = score_tld_risk("unknowntld")
        assert score == 30.0

    def test_gov_is_very_low(self):
        assert score_tld_risk("gov") <= 5


# ─────────────────────────────────────────────
# Keyword Scoring Tests (no network)
# ─────────────────────────────────────────────

class TestKeywordScoring:

    def test_login_detected(self):
        score, keywords = score_suspicious_keywords("https://paypal-login.tk/verify")
        assert "login" in keywords or "verify" in keywords
        assert score > 0

    def test_clean_url_no_keywords(self):
        score, keywords = score_suspicious_keywords("https://example.com/products/shoes")
        assert score == 0
        assert keywords == []

    def test_multiple_keywords_higher_score(self):
        score1, _ = score_suspicious_keywords("https://example.com/login")
        score2, _ = score_suspicious_keywords("https://example.com/login/verify/account/secure")
        assert score2 >= score1

    def test_score_capped_at_100(self):
        # Many keywords — should not exceed 100
        url = "https://login-verify-account-secure-password-banking-paypal-update.tk"
        score, _ = score_suspicious_keywords(url)
        assert score <= 100


# ─────────────────────────────────────────────
# Domain Entropy Tests (no network)
# ─────────────────────────────────────────────

class TestDomainEntropy:

    def test_simple_domain_low_entropy(self):
        # "google" — simple, low entropy
        entropy = compute_domain_entropy("google")
        assert entropy < 3.5

    def test_random_domain_high_entropy(self):
        # Random-looking DGA domain
        entropy = compute_domain_entropy("xk2p9mzqvf8r")
        assert entropy > 3.0

    def test_empty_domain_returns_zero(self):
        assert compute_domain_entropy("") == 0.0

    def test_entropy_is_float(self):
        assert isinstance(compute_domain_entropy("example"), float)


# ─────────────────────────────────────────────
# Full URL Analysis Tests (require network)
# ─────────────────────────────────────────────

@pytest.mark.network
class TestURLAnalysis:

    @pytest.mark.asyncio
    async def test_clean_url_analysis(self):
        result = await analyze_url("https://example.com")
        assert result.final_url != ""
        assert result.domain != ""
        assert result.tld == "com"
        assert result.tld_risk_score < 20
        assert result.error is None or result.error == ""

    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self):
        result = await analyze_url("")
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_url_without_scheme(self):
        result = await analyze_url("example.com")
        assert "example.com" in result.original_url or "example.com" in result.final_url

    @pytest.mark.asyncio
    async def test_result_structure_complete(self):
        result = await analyze_url("https://example.com")
        assert hasattr(result, "original_url")
        assert hasattr(result, "final_url")
        assert hasattr(result, "redirect_chain")
        assert hasattr(result, "redirect_count")
        assert hasattr(result, "is_shortened")
        assert hasattr(result, "domain")
        assert hasattr(result, "tld")
        assert hasattr(result, "tld_risk_score")
        assert hasattr(result, "keyword_score")
        assert hasattr(result, "domain_entropy")
        assert hasattr(result, "suspicious_keywords")
        assert 0 <= result.tld_risk_score <= 100
        assert 0 <= result.keyword_score <= 100