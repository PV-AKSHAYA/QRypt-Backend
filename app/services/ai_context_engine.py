"""
SafeQR — ai_context_engine.py
AI Context Engine.

Input : Image bytes + Final URL + Optional extracted text
Output: AIContextResult (structured — no free text chaos)

Uses:
  - Google Gemini Vision (gemini-1.5-flash)
  - Sends poster image + URL + prompt
  - Forces structured JSON output via strict prompt engineering
  - Parses and validates response into AIContextResult

Output contract (enforced):
{
  "visual_context": "",
  "expected_brand": "",
  "url_match": "MATCH|MISMATCH|UNKNOWN",
  "impersonation_probability": 0.0,
  "confidence": 0.0,
  "explanation": ""
}

Never returns free text. If Gemini deviates, we catch and re-parse.
"""

import asyncio
import base64
import json
import logging
import re
from typing import Optional

import httpx

from app.models.response_models import AIContextResult, URLMatchStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
REQUEST_TIMEOUT = 30.0


# ─────────────────────────────────────────────
# Prompt Engineering
# ─────────────────────────────────────────────

def _build_prompt(final_url: str, extracted_text: Optional[str] = None) -> str:
    """
    Strict prompt that forces Gemini to return ONLY valid JSON.
    No preamble. No explanation outside JSON. No markdown fences.
    """
    text_section = ""
    if extracted_text and extracted_text.strip():
        text_section = f"\nText extracted from image (OCR):\n{extracted_text.strip()}\n"

    return f"""You are a QR code security forensics AI. Analyze the provided image and URL for impersonation or phishing.

Final URL decoded from QR code: {final_url}
{text_section}
Your task:
1. Identify what brand, organization, or service the image appears to represent
2. Determine if the URL matches that brand's legitimate domain
3. Assess the probability of impersonation or phishing

Respond with ONLY a JSON object. No explanation outside JSON. No markdown. No preamble.
Use exactly this structure:

{{
  "visual_context": "brief description of what the image shows (poster, flyer, QR code context)",
  "expected_brand": "the brand or organization the image represents, or UNKNOWN",
  "url_match": "MATCH if URL belongs to expected brand, MISMATCH if suspicious, UNKNOWN if cannot determine",
  "impersonation_probability": 0.0,
  "confidence": 0.0,
  "explanation": "one sentence forensic reasoning"
}}

Rules:
- impersonation_probability: float 0.0 to 1.0 (1.0 = definitely impersonating)
- confidence: float 0.0 to 1.0 (how confident you are in this assessment)
- url_match must be exactly one of: MATCH, MISMATCH, UNKNOWN
- Keep explanation under 100 words
- If image is unclear or generic, set expected_brand to UNKNOWN and confidence low"""


# ─────────────────────────────────────────────
# JSON Extraction
# ─────────────────────────────────────────────

def _extract_json(raw_text: str) -> Optional[dict]:
    """
    Extract JSON from Gemini response even if it wraps it
    in markdown fences or adds surrounding text.
    """
    # Try direct parse first
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find JSON object in response
    match = re.search(r'\{[\s\S]*\}', raw_text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _validate_and_build(data: dict) -> AIContextResult:
    """
    Validate parsed JSON and build AIContextResult.
    Applies defaults for missing/invalid fields.
    """
    # url_match validation
    url_match_raw = str(data.get("url_match", "UNKNOWN")).upper().strip()
    try:
        url_match = URLMatchStatus(url_match_raw)
    except ValueError:
        url_match = URLMatchStatus.UNKNOWN

    # Clamp floats to 0.0-1.0
    def clamp(val, default=0.5):
        try:
            return max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            return default

    return AIContextResult(
        visual_context=str(data.get("visual_context", "Unable to determine"))[:500],
        expected_brand=str(data.get("expected_brand", "UNKNOWN"))[:100],
        url_match=url_match,
        impersonation_probability=clamp(data.get("impersonation_probability"), 0.5),
        confidence=clamp(data.get("confidence"), 0.3),
        explanation=str(data.get("explanation", "No explanation provided"))[:500],
        skipped=False,
        error=None,
    )


# ─────────────────────────────────────────────
# Gemini API Call
# ─────────────────────────────────────────────

async def _call_gemini(
    image_bytes: bytes,
    prompt: str,
    api_key: str,
) -> str:
    """
    Send image + prompt to Gemini Vision.
    Returns raw text response.
    """
    # Encode image to base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Detect mime type from magic bytes
    if image_bytes[:4] == b'\x89PNG':
        mime_type = "image/png"
    elif image_bytes[:2] == b'\xff\xd8':
        mime_type = "image/jpeg"
    elif image_bytes[:4] == b'RIFF':
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"  # default fallback

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64,
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,      # Low temp = deterministic output
            "maxOutputTokens": 512,
            "topP": 0.8,
        }
    }

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(
            f"{GEMINI_API_URL}?key={api_key}",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            data = response.json()
            # Extract text from Gemini response structure
            try:
                text = (
                    data["candidates"][0]
                       ["content"]
                       ["parts"][0]
                       ["text"]
                )
                return text
            except (KeyError, IndexError) as e:
                raise ValueError(f"Unexpected Gemini response structure: {e}")

        elif response.status_code == 429:
            raise RuntimeError("Gemini rate limit hit")

        elif response.status_code == 400:
            detail = response.json().get("error", {}).get("message", "")
            raise ValueError(f"Gemini bad request: {detail}")

        else:
            raise RuntimeError(
                f"Gemini API error: HTTP {response.status_code} — {response.text[:200]}"
            )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

async def analyze_ai_context(
    image_bytes: bytes,
    final_url: str,
    api_key: str,
    extracted_text: Optional[str] = None,
) -> AIContextResult:
    """
    Main entry point. Fully async.

    Args:
        image_bytes    : Raw image bytes of the scanned poster/QR
        final_url      : Final URL after redirect unrolling
        api_key        : Gemini API key
        extracted_text : Optional OCR text from image

    Returns AIContextResult. Never raises.
    """
    if not api_key:
        return _skipped_result("No Gemini API key configured")

    if not image_bytes:
        return _skipped_result("No image provided")

    if not final_url:
        return _skipped_result("No URL to analyze")

    prompt = _build_prompt(final_url, extracted_text)

    try:
        raw_response = await _call_gemini(image_bytes, prompt, api_key)
        logger.debug(f"Gemini raw response: {raw_response[:300]}")

    except RuntimeError as e:
        logger.warning(f"Gemini call failed: {e}")
        return _error_result(str(e))

    except ValueError as e:
        logger.warning(f"Gemini response error: {e}")
        return _error_result(str(e))

    except Exception as e:
        logger.error(f"Gemini unexpected error: {e}")
        return _error_result(f"Unexpected error: {str(e)}")

    # Parse JSON from response
    parsed = _extract_json(raw_response)

    if not parsed:
        logger.warning(f"Could not extract JSON from Gemini response: {raw_response[:200]}")
        return _error_result("Gemini returned non-JSON response — could not parse")

    try:
        result = _validate_and_build(parsed)
        logger.info(
            f"AI context: brand={result.expected_brand}, "
            f"url_match={result.url_match}, "
            f"impersonation={result.impersonation_probability:.2f}"
        )
        return result

    except Exception as e:
        logger.error(f"AIContextResult build failed: {e}")
        return _error_result(f"Result validation failed: {str(e)}")


def _skipped_result(reason: str) -> AIContextResult:
    return AIContextResult(
        visual_context="",
        expected_brand="",
        url_match=URLMatchStatus.UNKNOWN,
        impersonation_probability=0.0,
        confidence=0.0,
        explanation=reason,
        skipped=True,
        error=None,
    )


def _error_result(error: str) -> AIContextResult:
    return AIContextResult(
        visual_context="",
        expected_brand="",
        url_match=URLMatchStatus.UNKNOWN,
        impersonation_probability=0.5,
        confidence=0.0,
        explanation="AI analysis failed",
        skipped=False,
        error=error,
    )


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python ai_context_engine.py <image_path> <final_url> [gemini_api_key]")
        sys.exit(1)

    image_path = sys.argv[1]
    final_url  = sys.argv[2]
    api_key    = sys.argv[3] if len(sys.argv) > 3 else os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        print("Error: Gemini API key required as 3rd argument or GEMINI_API_KEY env var")
        sys.exit(1)

    import json

    async def main():
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        result = await analyze_ai_context(image_bytes, final_url, api_key)
        print(json.dumps(result.model_dump(), indent=2))

    asyncio.run(main())