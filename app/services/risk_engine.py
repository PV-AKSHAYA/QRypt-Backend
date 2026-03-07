"""
app/services/risk_engine.py
────────────────────────────
Risk Scoring Engine.

Input  : PhysicalLayerResult + TechnicalLayerResult + AILayerResult
Output : RiskResult (score 0-100, verdict, breakdown)

Formula:
    Risk = (Physical * 0.30) + (ThreatIntel * 0.30) + (AIContext * 0.40)

Each layer produces a normalised sub-score 0-100.
Weights are loaded from config so judges can see them clearly.

Sub-score logic:
  Physical   → confidence directly (already 0-100)
  ThreatIntel→ weighted combo of VT engines + URL signals
  AIContext  → weighted combo of impersonation_prob + url_match + tld_risk
"""

import logging
from app.models.response_models import (
    PhysicalLayerResult,
    TechnicalLayerResult,
    AILayerResult,
    RiskResult,
    RiskBreakdown,
    Verdict,
    ReputationClass,
    URLMatch,
)
from app.core.config import settings

logger = logging.getLogger("safeqr.risk_engine")


# ══════════════════════════════════════════════════════════════
#  LAYER SUB-SCORERS  (each returns 0.0 – 100.0)
# ══════════════════════════════════════════════════════════════

def _score_physical(physical: PhysicalLayerResult) -> float:
    """
    Physical sub-score = tamper confidence directly.
    Already normalised 0-100 by physical_analyzer.
    """
    return float(physical.confidence)


def _score_threat_intel(tech: TechnicalLayerResult) -> float:
    """
    Threat intel sub-score combines:
      - VirusTotal malicious engine count  (up to 50 pts)
      - VirusTotal suspicious engine count (up to 10 pts)
      - TLD risk score                     (up to 15 pts)
      - Suspicious keyword count           (up to 10 pts)
      - Hop count penalty                  (up to 10 pts)
      - No SSL penalty                     (up to  5 pts)
    Total capped at 100.
    """
    score = 0.0
    vt    = tech.virustotal

    # ── VirusTotal signals ────────────────────────────────────
    if vt.reputation_class == ReputationClass.MALICIOUS:
        # Scale malicious count — 10 engines = full 50 pts
        score += min(vt.malicious * 5.0, 50.0)
    elif vt.reputation_class == ReputationClass.SUSPICIOUS:
        score += min(vt.malicious * 5.0, 50.0)
        score += min(vt.suspicious * 2.0, 10.0)
    # UNKNOWN → 0 pts (benefit of the doubt)

    # ── TLD risk ──────────────────────────────────────────────
    score += tech.tld_risk_score * 15.0

    # ── Suspicious keywords ───────────────────────────────────
    keyword_count = len(tech.suspicious_keywords)
    score        += min(keyword_count * 2.5, 10.0)

    # ── Redirect hop penalty ──────────────────────────────────
    # 1 hop = normal, 2+ = suspicious, 4+ = max penalty
    if tech.hop_count >= 4:
        score += 10.0
    elif tech.hop_count >= 2:
        score += tech.hop_count * 2.0

    # ── SSL penalty ───────────────────────────────────────────
    if not tech.ssl_valid:
        score += 5.0

    return min(score, 100.0)


def _score_ai_context(ai: AILayerResult) -> float:
    """
    AI context sub-score combines:
      - Impersonation probability  (up to 60 pts)
      - URL match verdict          (up to 30 pts)
      - Low confidence penalty     (up to 10 pts)
    Total capped at 100.
    """
    score = 0.0

    # ── Impersonation probability (core signal) ───────────────
    score += ai.impersonation_probability * 60.0

    # ── URL match verdict ─────────────────────────────────────
    if ai.url_match == URLMatch.NO:
        score += 30.0
    elif ai.url_match == URLMatch.UNCERTAIN:
        score += 10.0
    # YES → 0 pts

    # ── Low confidence means AI is unsure → mild penalty ──────
    # If AI has low confidence, nudge score up slightly (conservative)
    if ai.confidence < 0.4:
        score += 10.0 * (1.0 - ai.confidence)

    return min(score, 100.0)


# ══════════════════════════════════════════════════════════════
#  VERDICT CLASSIFIER
# ══════════════════════════════════════════════════════════════

def _classify_verdict(score: float) -> Verdict:
    """
    Map final score to verdict using config thresholds.
    Default: >= 60 → HIGH_RISK, >= 30 → SUSPICIOUS, else SAFE
    """
    if score >= settings.THRESHOLD_HIGH_RISK:
        return Verdict.HIGH_RISK
    if score >= settings.THRESHOLD_SUSPICIOUS:
        return Verdict.SUSPICIOUS
    return Verdict.SAFE


# ══════════════════════════════════════════════════════════════
#  MAIN SCORE FUNCTION
# ══════════════════════════════════════════════════════════════

def calculate_risk(
    physical: PhysicalLayerResult,
    tech:     TechnicalLayerResult,
    ai:       AILayerResult,
) -> RiskResult:
    """
    Calculate final risk score from all three layers.

    Args:
        physical : Result from physical_analyzer
        tech     : Result from redirect_engine + threat_intel combined
        ai       : Result from ai_context_engine

    Returns:
        RiskResult with score, verdict, and per-layer breakdown
    """

    # ── Sub-scores ────────────────────────────────────────────
    physical_sub     = _score_physical(physical)
    threat_intel_sub = _score_threat_intel(tech)
    ai_context_sub   = _score_ai_context(ai)

    # ── Weighted total ────────────────────────────────────────
    w_physical     = settings.WEIGHT_PHYSICAL      # 0.30
    w_threat_intel = settings.WEIGHT_THREAT_INTEL  # 0.30
    w_ai_context   = settings.WEIGHT_AI_CONTEXT    # 0.40

    raw_score = (
        (physical_sub     * w_physical)     +
        (threat_intel_sub * w_threat_intel) +
        (ai_context_sub   * w_ai_context)
    )

    final_score = min(round(raw_score), 100)

    # ── Breakdown — what contributed how much ─────────────────
    breakdown = RiskBreakdown(
        physical_score     = round(physical_sub     * w_physical,     2),
        threat_intel_score = round(threat_intel_sub * w_threat_intel, 2),
        ai_context_score   = round(ai_context_sub   * w_ai_context,   2),
    )

    verdict = _classify_verdict(final_score)

    logger.info(
        f"Risk score: {final_score}/100 | verdict={verdict} | "
        f"physical={breakdown.physical_score} "
        f"threat={breakdown.threat_intel_score} "
        f"ai={breakdown.ai_context_score}"
    )

    return RiskResult(
        score     = final_score,
        verdict   = verdict,
        breakdown = breakdown,
    )