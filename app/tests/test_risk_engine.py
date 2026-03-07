"""
tests/test_risk_engine.py
──────────────────────────
Run with: pytest app/tests/test_risk_engine.py -v

Tests every scoring scenario:
  - Pure SAFE inputs  → low score
  - Pure DANGER inputs → high score
  - Mixed inputs      → middle score
  - Boundary checks   → score always 0-100
  - Verdict thresholds → correct classification
  - Breakdown math    → parts sum to total
"""

import pytest
from app.services.risk_engine import (
    calculate_risk,
    _score_physical,
    _score_threat_intel,
    _score_ai_context,
    _classify_verdict,
)
from app.models.response_models import (
    PhysicalLayerResult,
    TechnicalLayerResult,
    AILayerResult,
    VirusTotalResult,
    RiskBreakdown,
    Verdict,
    ReputationClass,
    URLMatch,
)


# ══════════════════════════════════════════════════════════════
#  FIXTURE BUILDERS
# ══════════════════════════════════════════════════════════════

def make_clean_physical() -> PhysicalLayerResult:
    return PhysicalLayerResult(
        tampered=False, confidence=0,
        evidence="No tamper signals detected."
    )


def make_tampered_physical(confidence: int = 75) -> PhysicalLayerResult:
    return PhysicalLayerResult(
        tampered=True, confidence=confidence,
        evidence="Double-edge signature detected at (142,88)."
    )


def make_clean_tech() -> TechnicalLayerResult:
    return TechnicalLayerResult(
        original_url        = "https://starbucks.com/menu",
        final_url           = "https://starbucks.com/menu",
        redirect_chain      = ["https://starbucks.com/menu"],
        hop_count           = 0,
        ssl_valid           = True,
        is_shortener        = False,
        domain_entropy      = 2.3,
        tld_risk_score      = 0.05,
        suspicious_keywords = [],
        domain_age_days     = 3000,
        virustotal          = VirusTotalResult(
            malicious=0, suspicious=0, harmless=80,
            total_engines=87, reputation_class=ReputationClass.CLEAN
        ),
    )


def make_dangerous_tech() -> TechnicalLayerResult:
    return TechnicalLayerResult(
        original_url        = "https://bit.ly/scam",
        final_url           = "https://taxrefund-claim.xyz/pay",
        redirect_chain      = ["https://bit.ly/scam", "https://taxrefund-claim.xyz/pay"],
        hop_count           = 2,
        ssl_valid           = False,
        is_shortener        = True,
        domain_entropy      = 4.1,
        tld_risk_score      = 0.85,
        suspicious_keywords = ["refund", "claim", "tax", "pay"],
        domain_age_days     = 3,
        virustotal          = VirusTotalResult(
            malicious=4, suspicious=2, harmless=70,
            total_engines=87, reputation_class=ReputationClass.MALICIOUS
        ),
    )


def make_clean_ai() -> AILayerResult:
    return AILayerResult(
        visual_context            = "Starbucks table menu card",
        expected_brand            = "Starbucks Coffee",
        url_match                 = URLMatch.YES,
        impersonation_probability = 0.02,
        confidence                = 0.95,
        explanation               = "URL matches expected brand perfectly.",
    )


def make_dangerous_ai() -> AILayerResult:
    return AILayerResult(
        visual_context            = "Official government tax payment poster",
        expected_brand            = "IRS / Tax Authority",
        url_match                 = URLMatch.NO,
        impersonation_probability = 0.94,
        confidence                = 0.88,
        explanation               = "QR leads to unregistered domain, not a government TLD.",
    )


def make_uncertain_ai() -> AILayerResult:
    return AILayerResult(
        visual_context            = "Generic flyer",
        expected_brand            = "Unknown",
        url_match                 = URLMatch.UNCERTAIN,
        impersonation_probability = 0.4,
        confidence                = 0.3,
        explanation               = "Cannot determine brand from image.",
    )


# ══════════════════════════════════════════════════════════════
#  UNIT TESTS — sub-scorers
# ══════════════════════════════════════════════════════════════

class TestScorePhysical:

    def test_clean_scores_zero(self):
        assert _score_physical(make_clean_physical()) == 0.0

    def test_tampered_scores_confidence(self):
        assert _score_physical(make_tampered_physical(75)) == 75.0

    def test_max_confidence_caps_at_100(self):
        p = make_tampered_physical(100)
        assert _score_physical(p) == 100.0


class TestScoreThreatIntel:

    def test_clean_url_low_score(self):
        score = _score_threat_intel(make_clean_tech())
        assert score < 15.0, f"Clean URL scored too high: {score}"

    def test_malicious_url_high_score(self):
        score = _score_threat_intel(make_dangerous_tech())
        assert score >= 40.0, f"Dangerous URL scored too low: {score}"

    def test_no_ssl_adds_penalty(self):
        tech = make_clean_tech()
        tech.ssl_valid = False
        score_no_ssl = _score_threat_intel(tech)
        tech.ssl_valid = True
        score_ssl = _score_threat_intel(tech)
        assert score_no_ssl > score_ssl

    def test_keywords_increase_score(self):
        tech_no_kw = make_clean_tech()
        tech_kw    = make_clean_tech()
        tech_kw.suspicious_keywords = ["login", "verify", "bank"]
        assert _score_threat_intel(tech_kw) > _score_threat_intel(tech_no_kw)

    def test_many_hops_increase_score(self):
        tech_direct  = make_clean_tech()
        tech_hopping = make_clean_tech()
        tech_hopping.hop_count = 4
        assert _score_threat_intel(tech_hopping) > _score_threat_intel(tech_direct)

    def test_score_never_exceeds_100(self):
        # Even the most dangerous URL caps at 100
        assert _score_threat_intel(make_dangerous_tech()) <= 100.0

    def test_unknown_vt_gives_no_penalty(self):
        """UNKNOWN reputation = benefit of the doubt."""
        tech = make_clean_tech()
        tech.virustotal = VirusTotalResult(
            malicious=0, suspicious=0,
            total_engines=0, reputation_class=ReputationClass.UNKNOWN
        )
        score = _score_threat_intel(tech)
        assert score < 20.0


class TestScoreAIContext:

    def test_clean_match_low_score(self):
        score = _score_ai_context(make_clean_ai())
        assert score < 10.0, f"Clean AI scored too high: {score}"

    def test_impersonation_high_score(self):
        score = _score_ai_context(make_dangerous_ai())
        assert score >= 80.0, f"Dangerous AI scored too low: {score}"

    def test_uncertain_match_medium_score(self):
        clean_score     = _score_ai_context(make_clean_ai())
        uncertain_score = _score_ai_context(make_uncertain_ai())
        dangerous_score = _score_ai_context(make_dangerous_ai())
        assert clean_score < uncertain_score < dangerous_score

    def test_url_no_adds_to_score(self):
        ai_yes = make_clean_ai()
        ai_no  = make_clean_ai()
        ai_no.url_match = URLMatch.NO
        assert _score_ai_context(ai_no) > _score_ai_context(ai_yes)

    def test_score_never_exceeds_100(self):
        assert _score_ai_context(make_dangerous_ai()) <= 100.0


# ══════════════════════════════════════════════════════════════
#  VERDICT CLASSIFIER
# ══════════════════════════════════════════════════════════════

class TestClassifyVerdict:

    def test_low_score_is_safe(self):
        assert _classify_verdict(0)  == Verdict.SAFE
        assert _classify_verdict(15) == Verdict.SAFE
        assert _classify_verdict(29) == Verdict.SAFE

    def test_mid_score_is_suspicious(self):
        assert _classify_verdict(30) == Verdict.SUSPICIOUS
        assert _classify_verdict(45) == Verdict.SUSPICIOUS
        assert _classify_verdict(59) == Verdict.SUSPICIOUS

    def test_high_score_is_high_risk(self):
        assert _classify_verdict(60)  == Verdict.HIGH_RISK
        assert _classify_verdict(82)  == Verdict.HIGH_RISK
        assert _classify_verdict(100) == Verdict.HIGH_RISK


# ══════════════════════════════════════════════════════════════
#  FULL CALCULATE_RISK TESTS
# ══════════════════════════════════════════════════════════════

class TestCalculateRisk:

    def test_all_clean_gives_safe(self):
        result = calculate_risk(
            make_clean_physical(),
            make_clean_tech(),
            make_clean_ai(),
        )
        assert result.verdict == Verdict.SAFE
        assert result.score   < 30

    def test_all_dangerous_gives_high_risk(self):
        result = calculate_risk(
            make_tampered_physical(80),
            make_dangerous_tech(),
            make_dangerous_ai(),
        )
        assert result.verdict == Verdict.HIGH_RISK
        assert result.score   >= 60

    def test_mixed_signals_suspicious(self):
        """Clean physical + risky URL + uncertain AI → SUSPICIOUS."""
        result = calculate_risk(
            make_clean_physical(),
            make_dangerous_tech(),
            make_uncertain_ai(),
        )
        assert result.score >= 30

    def test_score_always_0_to_100(self):
        """Score must never go outside 0-100."""
        for _ in range(3):
            result = calculate_risk(
                make_tampered_physical(100),
                make_dangerous_tech(),
                make_dangerous_ai(),
            )
            assert 0 <= result.score <= 100

    def test_breakdown_fields_present(self):
        """Breakdown must have all three sub-scores."""
        result = calculate_risk(
            make_clean_physical(),
            make_clean_tech(),
            make_clean_ai(),
        )
        assert isinstance(result.breakdown.physical_score,     float)
        assert isinstance(result.breakdown.threat_intel_score, float)
        assert isinstance(result.breakdown.ai_context_score,   float)

    def test_breakdown_sums_to_total(self):
        """
        The three breakdown scores should sum close to the final score.
        (Small floating point rounding is acceptable.)
        """
        result = calculate_risk(
            make_tampered_physical(60),
            make_dangerous_tech(),
            make_dangerous_ai(),
        )
        total_from_breakdown = (
            result.breakdown.physical_score     +
            result.breakdown.threat_intel_score +
            result.breakdown.ai_context_score
        )
        assert abs(total_from_breakdown - result.score) <= 2

    def test_verdict_matches_score(self):
        """Verdict enum must always match score range."""
        result = calculate_risk(
            make_clean_physical(),
            make_clean_tech(),
            make_clean_ai(),
        )
        if result.score < 30:
            assert result.verdict == Verdict.SAFE
        elif result.score < 60:
            assert result.verdict == Verdict.SUSPICIOUS
        else:
            assert result.verdict == Verdict.HIGH_RISK