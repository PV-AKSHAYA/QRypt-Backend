"""
app/services/ai_context_engine.py
Works with: google-genai (new SDK)
Model: gemini-1.5-flash-8b
"""

import io
import json
import logging
import re

from google import genai
from google.genai import types as genai_types

from app.models.response_models import AILayerResult, URLMatch
from app.utils.image_utils import bytes_to_pil
from app.core.config import settings

logger = logging.getLogger("safeqr.ai_context_engine")

GEMINI_MODEL = "gemini-1.5-flash-8b"


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
  "impersonation_probability": <float 0.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "explanation": "<one clear sentence summarizing the forensic verdict>"
}}

Rules:
- url_match must be exactly one of: YES, NO, UNCERTAIN
- impersonation_probability and confidence must be numbers between 0.0 and 1.0
- explanation must be a single sentence under 200 characters
- If you cannot determine the visual context, set visual_context to "Unknown environment"
- If no clear brand is visible, set expected_brand to "Unknown"
"""


def _parse_gemini_response(raw_text: str) -> dict | None:
    text = raw_text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e} | Raw: {raw_text[:200]}")
        return None


def _validate_and_build(data: dict) -> AILayerResult | None:
    try:
        url_match_raw = str(data.get("url_match", "UNCERTAIN")).upper().strip()
        if url_match_raw not in ("YES", "NO", "UNCERTAIN"):
            url_match_raw = "UNCERTAIN"
        url_match  = URLMatch(url_match_raw)
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


def _default_result(reason: str = "") -> AILayerResult:
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


def analyze_context(
    img_bytes:    bytes,
    final_url:    str,
    context_hint: str = "",
) -> AILayerResult:
    """
    Send image + URL to Gemini 1.5 Flash-8B for multi-modal forensic analysis.
    Uses google-genai (new SDK). Always returns, never raises.
    """

    if not settings.GEMINI_API_KEY:
        return _default_result("Gemini API key not configured")

    try:
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
    except Exception as e:
        return _default_result(f"Gemini init failed: {e}")

    try:
        pil_image = bytes_to_pil(img_bytes)
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        img_bytes_png = buf.getvalue()
    except Exception as e:
        return _default_result(f"Image load failed: {e}")

    prompt = _build_prompt(final_url, context_hint)

    for attempt in range(1, 3):
        try:
            logger.info(f"Gemini request attempt {attempt}: {final_url[:60]}")

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    genai_types.Part.from_bytes(data=img_bytes_png, mime_type="image/png"),
                    genai_types.Part.from_text(text=prompt),
                ],
                config=genai_types.GenerateContentConfig(
                    temperature       = 0.1,
                    max_output_tokens = 512,
                ),
            )

            raw_text = response.text
            logger.debug(f"Gemini raw response: {raw_text[:300]}")

            parsed = _parse_gemini_response(raw_text)
            if parsed is None:
                logger.warning(f"Attempt {attempt}: JSON parse failed, retrying")
                continue

            result = _validate_and_build(parsed)
            if result is None:
                logger.warning(f"Attempt {attempt}: Schema validation failed, retrying")
                continue

            logger.info(
                f"Gemini analysis complete: url_match={result.url_match} "
                f"impersonation={result.impersonation_probability} "
                f"confidence={result.confidence}"
            )
            return result

        except Exception as e:
            logger.error(f"Gemini attempt {attempt} failed: {e}")
            if attempt == 2:
                return _default_result(f"Gemini error: {str(e)[:100]}")

    return _default_result("Gemini returned unparseable response after 2 attempts")