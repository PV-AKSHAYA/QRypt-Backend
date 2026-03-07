
import json
import logging
import re
from typing import Optional

from groq import Groq
from app.models.response_models import AILayerResult, URLMatch
from app.core.config import settings

logger = logging.getLogger("safeqr.ai_context")

client = Groq(api_key=settings.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a cybersecurity expert specialising in phishing and QR code fraud detection.

Analyse the provided URL and context. Return ONLY a valid JSON object — no preamble, no markdown, no explanation.

Required JSON schema:
{
  "visual_context": "<what the QR likely leads to based on URL analysis>",
  "expected_brand": "<brand name if URL implies one, or 'Unknown'>",
  "url_match": "<one of: YES | NO | UNCERTAIN>",
  "impersonation_probability": <float 0.0 to 1.0>,
  "confidence": <float 0.0 to 1.0>,
  "explanation": "<one concise sentence explaining the verdict>"
}

Rules:
- url_match = YES if URL domain legitimately matches the expected brand
- url_match = NO if URL uses typosquatting, character substitution, or wrong domain
- url_match = UNCERTAIN if brand cannot be determined
- impersonation_probability: 0.0 = definitely legitimate, 1.0 = definite phishing
- Output ONLY the JSON. Any extra text will break the system."""


def _parse_response(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No JSON found in response")


def analyze_context(
    img_bytes: bytes,
    final_url: str,
    context_hint: str = "",
) -> AILayerResult:
    """
    Analyse URL context using Kimi K2 via Groq.
    img_bytes accepted for API compatibility but not sent (text model).
    """
    prompt_parts = [f"URL to analyse: {final_url}"]
    if context_hint:
        prompt_parts.append(f"Context hint: {context_hint}")
    prompt_parts.append("Perform phishing/impersonation analysis. Return JSON only.")

    try:
        response = client.chat.completions.create(
            model="moonshotai/kimi-k2-instruct-0905",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "\n".join(prompt_parts)},
            ],
            max_tokens=512,
            temperature=0.0,
        )

        raw = response.choices[0].message.content
        data = _parse_response(raw)

        url_match_raw = data.get("url_match", "UNCERTAIN").upper()
        if url_match_raw == "YES":
            url_match = URLMatch.YES
        elif url_match_raw == "NO":
            url_match = URLMatch.NO
        else:
            url_match = URLMatch.UNCERTAIN

        return AILayerResult(
            visual_context           = str(data.get("visual_context", ""))[:500],
            expected_brand           = str(data.get("expected_brand", "Unknown"))[:100],
            url_match                = url_match,
            impersonation_probability= max(0.0, min(1.0, float(data.get("impersonation_probability", 0.0)))),
            confidence               = max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
            explanation              = str(data.get("explanation", ""))[:300],
        )

    except Exception as e:
        logger.error(f"Groq AI context error: {e}")
        return AILayerResult(
            visual_context="AI analysis unavailable",
            expected_brand="Unknown",
            url_match=URLMatch.UNCERTAIN,
            impersonation_probability=0.5,
            confidence=0.0,
            explanation=f"Groq error: {str(e)[:80]}",
        )