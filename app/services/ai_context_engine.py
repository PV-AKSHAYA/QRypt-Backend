"""
app/services/ai_context_engine.py
───────────────────────────────────
AI Context Engine — Gemini 1.5 Flash multi-modal analysis.

Input  : image bytes + final URL + optional context hint
Output : AILayerResult (forced structured JSON — no free-text chaos)

Flow:
  1. Build a strict prompt that forces JSON output
  2. Send image + URL to Gemini 1.5 Flash
  3. Parse and validate response against AILayerResult schema
  4. Retry once if JSON parse fails
  5. Return safe default if Gemini is unavailable

Key design decision:
  The prompt explicitly forbids free text and demands a JSON object
  matching our exact schema. If Gemini drifts, we catch it and retry.
"""

import json
import logging
import re
from PIL import Image

import google.generativeai as genai

from app.models.response_models import AILayerResult, URLMatch
from app.utils.image_utils import bytes_to_pil
from app.core.config import settings

logger = logging.getLogger("safeqr.ai_context_engine")


# ══════════════════════════════════════════════════════════════
#  PROMPT — strict JSON contract
# ══════════════════════════════════════════════════════════════

def _build_prompt(final_url: str, context_hint: str = "") -> str:
    hint_line = f"\nContext hint from user: {context_hint}" if context_hint else ""

    return f"""You are a QR code security forensics analyst.
Analyze the image provided. It contains a QR code in a physical environment.

The QR code leads to this URL: {final_url}{hint_line}

Your task: Determine if this QR code is legitimate or a phishing/quishing attack.

Respond ONLY with a valid JSON object. No explanation before or after.
No markdown. No code fences. Just the raw JSON object.

Use exactly this schema:
{{
  "visual_context": "<describe what you see in the image: poster, flyer, menu, etc>",
  "expected_brand": "<what brand or organization does the visual suggest>",
  "url_match": "<YES if URL matches expected brand, NO if mismatch, UNCERTAIN if unclear>",
  "impersonation_probability": <float 0.0 to 1.0 — likelihood this is impersonating a brand>,
  "confidence": <float 0.0 to 1.0 — your confidence in this analysis>,
  "explanation": "<one clear sentence summarizing the forensic verdict>"
}}

Rules:
- url_match must be exactly one of: YES, NO, UNCERTAIN
- impersonation_probability and confidence must be numbers between 0.0 and 1.0
- explanation must be a single sentence under 200 characters
- If you cannot determine the visual context, set visual_context to "Unknown environment"
- If no clear brand is visible, set expected_brand to "Unknown"
"""


# ══════════════════════════════════════════════════════════════
#  JSON PARSER — handles Gemini drift
# ══════════════════════════════════════════════════════════════

def _parse_gemini_response(raw_text: str) -> dict | None:
    """
    Parse Gemini's response into a dict.
    Handles common drift patterns:
      - JSON wrapped in ```json ... ``` fences
      - Leading/trailing whitespace
      - Extra text before/after the JSON object
    Returns None if parsing fails completely.
    """
    text = raw_text.strip()

    # Strip markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    # Extract just the JSON object if there's extra text around it
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e} | Raw: {raw_text[:200]}")
        return None


def _validate_and_build(data: dict) -> AILayerResult | None:
    """
    Validate parsed dict against AILayerResult schema.
    Coerces types where possible, returns None if critical fields missing.
    """
    try:
        # Coerce url_match to enum
        url_match_raw = str(data.get("url_match", "UNCERTAIN")).upper().strip()
        if url_match_raw not in ("YES", "NO", "UNCERTAIN"):
            url_match_raw = "UNCERTAIN"
        url_match = URLMatch(url_match_raw)

        # Clamp floats to 0.0–1.0
        imp_prob   = max(0.0, min(1.0, float(data.get("impersonation_probability", 0.5))))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))

        return AILayerResult(
            visual_context            = str(data.get("visual_context", "Unknown environment"))[:300],
            expected_brand            = str(data.get("expected_brand", "Unknown"))[:100],
            url_match                 = url_match,
            impersonation_probability = round(imp_prob,   2),
            confidence                = round(confidence, 2),
            explanation               = str(data.get("explanation", "AI analysis completed"))[:200],
        )

    except Exception as e:
        logger.warning(f"AILayerResult build failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  SAFE DEFAULT
# ══════════════════════════════════════════════════════════════

def _default_result(reason: str = "") -> AILayerResult:
    """
    Return a neutral result when Gemini is unavailable.
    Scan continues — AI is one layer of three.
    """
    if reason:
        logger.info(f"AI context engine returning default: {reason}")
    return AILayerResult(
        visual_context            = "AI analysis unavailable",
        expected_brand            = "Unknown",
        url_match                 = URLMatch.UNCERTAIN,
        impersonation_probability = 0.5,
        confidence                = 0.0,
        explanation               = reason or "Gemini AI context analysis was not available.",
    )


# ══════════════════════════════════════════════════════════════
#  MAIN ANALYZE FUNCTION
# ══════════════════════════════════════════════════════════════

def analyze_context(
    img_bytes:    bytes,
    final_url:    str,
    context_hint: str = "",
) -> AILayerResult:
    """
    Send image + URL to Gemini 1.5 Flash for multi-modal forensic analysis.

    Args:
        img_bytes:    Raw image bytes of the QR environment photo
        final_url:    Final destination URL after all redirects
        context_hint: Optional human hint (e.g. "bank poster")

    Returns:
        AILayerResult — always returns, never raises
    """

    # ── Guard: API key ────────────────────────────────────────
    if not settings.GEMINI_API_KEY:
        return _default_result("Gemini API key not configured")

    # ── Configure Gemini ──────────────────────────────────────
    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        return _default_result(f"Gemini init failed: {e}")

    # ── Load image for Gemini ─────────────────────────────────
    try:
        pil_image = bytes_to_pil(img_bytes)
    except Exception as e:
        return _default_result(f"Image load failed: {e}")

    # ── Build prompt ──────────────────────────────────────────
    prompt = _build_prompt(final_url, context_hint)

    # ── Call Gemini with one retry ────────────────────────────
    for attempt in range(1, 3):
        try:
            logger.info(f"Gemini request attempt {attempt}: {final_url[:60]}")

            response  = model.generate_content(
                [prompt, pil_image],
                generation_config=genai.GenerationConfig(
                    temperature      = 0.1,   # low temp = consistent outputs
                    max_output_tokens= 512,
                ),
            )

            raw_text = response.text
            logger.debug(f"Gemini raw response: {raw_text[:300]}")

            # ── Parse ─────────────────────────────────────────
            parsed = _parse_gemini_response(raw_text)
            if parsed is None:
                logger.warning(f"Attempt {attempt}: JSON parse failed, retrying")
                continue

            # ── Validate ──────────────────────────────────────
            result = _validate_and_build(parsed)
            if result is None:
                logger.warning(f"Attempt {attempt}: Schema validation failed, retrying")
                continue

            logger.info(
                f"Gemini analysis complete: "
                f"url_match={result.url_match} "
                f"impersonation={result.impersonation_probability} "
                f"confidence={result.confidence}"
            )
            return result

        except Exception as e:
            logger.error(f"Gemini attempt {attempt} failed: {e}")
            if attempt == 2:
                return _default_result(f"Gemini error: {str(e)[:100]}")

    return _default_result("Gemini returned unparseable response after 2 attempts")