"""
tests/test_redirect_engine.py
───────────────────────────────
Run with: pytest app/tests/test_redirect_engine.py -v

Tests split into two groups:
  - Unit tests (no network — test pure functions from url_utils)
  - Integration tests (real network — marked, can be skipped offline)
"""

import pytest
from app.utils.url_utils import (
    compute_entropy,
    get_tld_risk,
    find_suspicious_keywords,
    is_shortener,
    check_ssl,
    parse_domain,
)
from app.services.redirect_engine import analyze_url


# ══════════════════════════════════════════════════════════════
#  UNIT TESTS — no network needed
# ══════════════════════════════════════════════════════════════

class TestUrlUtils:

    # ── Entropy ──────────────────────────────────────────────

    def test_entropy_low_for_real_words(self):
        """Human-readable domain names have low entropy."""
        assert compute_entropy("google") < 3.0
        assert compute_entropy("amazon") < 3.0
        assert compute_entropy("starbucks") < 3.5

    def test_entropy_high_for_random(self):
        """Random-looking strings have high entropy."""
        assert compute_entropy("xkq7z2mf9p") > 3.0
        assert compute_entropy("a1b2c3d4e5") > 3.0

    def test_entropy_zero_for_single_char(self):
        """Single repeated character = zero entropy."""
        assert compute_entropy("aaaaaaa") == 0.0

    def test_entropy_empty_string(self):
        """Empty string returns 0.0, no crash."""
        assert compute_entropy("") == 0.0

    # ── TLD Risk ─────────────────────────────────────────────

    def test_safe_tld_low_risk(self):
        assert get_tld_risk("https://example.com")  < 0.15
        assert get_tld_risk("https://example.org")  < 0.15
        assert get_tld_risk("https://example.gov")  < 0.05

    def test_risky_tld_high_risk(self):
        assert get_tld_risk("https://example.xyz")  > 0.70
        assert get_tld_risk("https://example.click") > 0.80
        assert get_tld_risk("https://example.loan")  > 0.85

    def test_unknown_tld_returns_default(self):
        score = get_tld_risk("https://example.unknowntld999")
        assert 0.0 <= score <= 1.0

    # ── Suspicious Keywords ───────────────────────────────────

    def test_phishing_keywords_detected(self):
        url  = "https://paypal-login-verify.xyz/account/confirm"
        kws  = find_suspicious_keywords(url)
        assert "login" in kws or "verify" in kws or "confirm" in kws

    def test_clean_url_no_keywords(self):
        url = "https://starbucks.com/menu/drinks"
        kws = find_suspicious_keywords(url)
        assert len(kws) == 0

    def test_tax_scam_keywords(self):
        url = "https://irs-tax-refund-claim.xyz/pay"
        kws = find_suspicious_keywords(url)
        assert "refund" in kws or "claim" in kws

    # ── Shortener Detection ───────────────────────────────────

    def test_known_shorteners_detected(self):
        assert is_shortener("https://bit.ly/3xQfT9")     is True
        assert is_shortener("https://tinyurl.com/xyz")   is True
        assert is_shortener("https://t.co/abc")          is True
        assert is_shortener("https://cutt.ly/example")   is True

    def test_real_domain_not_shortener(self):
        assert is_shortener("https://google.com")        is False
        assert is_shortener("https://starbucks.com/menu") is False

    # ── SSL Check ────────────────────────────────────────────

    def test_https_url_passes_scheme_check(self):
        assert check_ssl("https://example.com") is True

    def test_http_url_fails_scheme_check(self):
        assert check_ssl("http://example.com")  is False

    # ── Domain Parsing ────────────────────────────────────────

    def test_parse_domain_extracts_components(self):
        result = parse_domain("https://mail.google.com/path?q=1")
        assert result["domain"]      == "google"
        assert result["suffix"]      == "com"
        assert result["subdomain"]   == "mail"
        assert result["full_domain"] == "google.com"

    def test_parse_domain_no_subdomain(self):
        result = parse_domain("https://example.com/path")
        assert result["domain"] == "example"
        assert result["subdomain"] == ""


# ══════════════════════════════════════════════════════════════
#  INTEGRATION TESTS — require network
#  Skip with: pytest -m "not network"
# ══════════════════════════════════════════════════════════════

@pytest.mark.network
class TestRedirectEngineIntegration:

    def test_direct_url_no_hops(self):
        """A direct URL with no redirects → chain length 1."""
        result = analyze_url("https://httpbin.org/get")
        assert result.hop_count    == 0
        assert result.ssl_valid    is True
        assert result.final_url    == "https://httpbin.org/get"
        assert len(result.redirect_chain) == 1

    def test_output_structure_complete(self):
        """All required fields must be present."""
        result = analyze_url("https://example.com")
        assert result.original_url        is not None
        assert result.final_url           is not None
        assert result.redirect_chain      is not None
        assert isinstance(result.hop_count,           int)
        assert isinstance(result.ssl_valid,           bool)
        assert isinstance(result.is_shortener,        bool)
        assert isinstance(result.domain_entropy,      float)
        assert isinstance(result.tld_risk_score,      float)
        assert isinstance(result.suspicious_keywords, list)

    def test_safe_domain_low_scores(self):
        """A clean real-world domain should score low on risk indicators."""
        result = analyze_url("https://github.com")
        assert result.ssl_valid         is True
        assert result.tld_risk_score    < 0.15
        assert result.suspicious_keywords == []

    def test_network_error_no_crash(self):
        """Unreachable URL must not crash — return best-effort result."""
        result = analyze_url("https://this-domain-definitely-does-not-exist-12345.xyz")
        assert result.original_url is not None
        assert isinstance(result.hop_count, int)
        assert isinstance(result.redirect_chain, list)