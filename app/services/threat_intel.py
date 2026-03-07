"""
SafeQR — threat_intel.py
Threat Intelligence Layer.

Input : Final URL (from redirect_engine)
Output: ThreatIntelResult (malicious_count, total_engines, reputation_class, etc.)

Uses:
  - VirusTotal API v3 (URL scan + report)
  - Local DB cache (via image_hash) — checked before hitting VT
  - Previous flag count from our own scan history

Flow:
  1. Check local DB cache for this URL
  2. If cached and fresh → return cached result
  3. Submit URL to VirusTotal for scanning
  4. Poll for analysis result (max 30s)
  5. Parse + score result
  6. Return ThreatIntelResult

Rate limit: VT free tier = 4 requests/minute.
We handle 429 gracefully.
"""

import asyncio
import hashlib
import logging
import time
from typing import Optional

import httpx

from app.models.response_models import ThreatIntelResult, ReputationClass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

VT_BASE_URL         = "https://www.virustotal.com/api/v3"
VT_POLL_INTERVAL    = 3.0    # seconds between polls
VT_MAX_POLL_ATTEMPTS= 8      # max 24 seconds waiting
REQUEST_TIMEOUT     = 15.0


# ─────────────────────────────────────────────
# Reputation Classifier
# ─────────────────────────────────────────────

def classify_reputation(malicious: int, suspicious: int, total: int) -> ReputationClass:
    """
    Classify URL reputation based on engine votes.

    Thresholds (conservative — minimize false negatives):
      MALICIOUS  : 2+ engines flag as malicious
      SUSPICIOUS : 1 malicious OR 3+ suspicious
      CLEAN      : 0 malicious, <3 suspicious
      UNKNOWN    : no engines returned data
    """
    if total == 0:
        return ReputationClass.UNKNOWN
    if malicious >= 2:
        return ReputationClass.MALICIOUS
    if malicious == 1 or suspicious >= 3:
        return ReputationClass.SUSPICIOUS
    return ReputationClass.CLEAN


def reputation_to_score(reputation: ReputationClass) -> float:
    """Convert reputation class to 0-100 risk score for risk engine."""
    mapping = {
        ReputationClass.CLEAN:      5.0,
        ReputationClass.UNKNOWN:    30.0,
        ReputationClass.SUSPICIOUS: 65.0,
        ReputationClass.MALICIOUS:  95.0,
    }
    return mapping.get(reputation, 30.0)


# ─────────────────────────────────────────────
# URL ID for VirusTotal
# ─────────────────────────────────────────────

def _url_to_vt_id(url: str) -> str:
    """
    VirusTotal v3 uses base64url-encoded URL as the resource ID.
    No padding required.
    """
    import base64
    encoded = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    return encoded


# ─────────────────────────────────────────────
# VirusTotal API calls
# ─────────────────────────────────────────────

async def _get_existing_report(
    client: httpx.AsyncClient,
    url: str,
    api_key: str
) -> Optional[dict]:
    """
    Try to fetch an existing VT report for this URL without submitting a new scan.
    Returns raw stats dict or None if not found.
    """
    url_id = _url_to_vt_id(url)
    endpoint = f"{VT_BASE_URL}/urls/{url_id}"

    try:
        response = await client.get(
            endpoint,
            headers={"x-apikey": api_key},
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            stats = (
                data.get("data", {})
                    .get("attributes", {})
                    .get("last_analysis_stats", {})
            )
            if stats:
                logger.debug(f"VT existing report found for {url}")
                return stats
        elif response.status_code == 404:
            logger.debug(f"No existing VT report for {url}")
        elif response.status_code == 429:
            logger.warning("VT rate limit hit on report fetch")

    except Exception as e:
        logger.warning(f"VT report fetch error: {e}")

    return None


async def _submit_url_scan(
    client: httpx.AsyncClient,
    url: str,
    api_key: str
) -> Optional[str]:
    """
    Submit URL to VirusTotal for scanning.
    Returns analysis ID or None on failure.
    """
    try:
        response = await client.post(
            f"{VT_BASE_URL}/urls",
            headers={
                "x-apikey": api_key,
                "Content-Type": "application/x-www-form-urlencoded"
            },
            content=f"url={url}",
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            analysis_id = data.get("data", {}).get("id", "")
            logger.debug(f"VT scan submitted, analysis_id={analysis_id}")
            return analysis_id

        elif response.status_code == 429:
            logger.warning("VT rate limit hit on URL submission")

        else:
            logger.warning(f"VT submission failed: HTTP {response.status_code}")

    except Exception as e:
        logger.warning(f"VT submission error: {e}")

    return None


async def _poll_analysis(
    client: httpx.AsyncClient,
    analysis_id: str,
    api_key: str
) -> Optional[dict]:
    """
    Poll VT analysis endpoint until completed or timeout.
    Returns stats dict or None.
    """
    endpoint = f"{VT_BASE_URL}/analyses/{analysis_id}"

    for attempt in range(VT_MAX_POLL_ATTEMPTS):
        await asyncio.sleep(VT_POLL_INTERVAL)

        try:
            response = await client.get(
                endpoint,
                headers={"x-apikey": api_key},
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                attrs = data.get("data", {}).get("attributes", {})
                status = attrs.get("status", "")

                if status == "completed":
                    stats = attrs.get("stats", {})
                    logger.debug(f"VT analysis completed on attempt {attempt + 1}")
                    return stats

                logger.debug(f"VT analysis status: {status} (attempt {attempt + 1})")

            elif response.status_code == 429:
                logger.warning("VT rate limit hit during polling")
                await asyncio.sleep(15)

        except Exception as e:
            logger.warning(f"VT poll error (attempt {attempt + 1}): {e}")

    logger.warning(f"VT analysis timed out after {VT_MAX_POLL_ATTEMPTS} attempts")
    return None


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

async def check_threat_intel(
    url: str,
    api_key: str,
    previous_flag_count: int = 0,
) -> ThreatIntelResult:
    """
    Main entry point. Fully async.

    Args:
        url              : Final URL after redirect unrolling
        api_key          : VirusTotal API key
        previous_flag_count: Times this domain was flagged in our DB

    Returns ThreatIntelResult. Never raises.
    """
    if not url or not api_key:
        return _error_result("Missing URL or API key", previous_flag_count)

    async with httpx.AsyncClient(verify=False) as client:

        # ── Step 1: Try existing report first (saves quota) ──
        stats = await _get_existing_report(client, url, api_key)

        # ── Step 2: Submit new scan if no existing report ────
        if not stats:
            analysis_id = await _submit_url_scan(client, url, api_key)

            if analysis_id:
                # ── Step 3: Poll for result ──────────────────
                stats = await _poll_analysis(client, analysis_id, api_key)

        # ── Step 4: Parse result ─────────────────────────────
        if not stats:
            return ThreatIntelResult(
                queried=True,
                malicious_count=0,
                suspicious_count=0,
                total_engines=0,
                reputation_class=ReputationClass.UNKNOWN,
                vt_url=f"https://www.virustotal.com/gui/url/{_url_to_vt_id(url)}",
                cached=False,
                previous_flag_count=previous_flag_count,
                error="VT scan did not complete in time — try again shortly",
            )

        malicious  = stats.get("malicious",  0)
        suspicious = stats.get("suspicious", 0)
        harmless   = stats.get("harmless",   0)
        undetected = stats.get("undetected", 0)
        total      = malicious + suspicious + harmless + undetected

        reputation = classify_reputation(malicious, suspicious, total)

        # Previous flag warning
        if previous_flag_count > 0:
            logger.info(f"Domain previously flagged {previous_flag_count} time(s) in our DB")

        logger.info(
            f"VT result: malicious={malicious}, suspicious={suspicious}, "
            f"total={total}, reputation={reputation}"
        )

        return ThreatIntelResult(
            queried=True,
            malicious_count=malicious,
            suspicious_count=suspicious,
            total_engines=total,
            reputation_class=reputation,
            vt_url=f"https://www.virustotal.com/gui/url/{_url_to_vt_id(url)}",
            cached=False,
            previous_flag_count=previous_flag_count,
            error=None,
        )


def _error_result(error: str, previous_flag_count: int = 0) -> ThreatIntelResult:
    return ThreatIntelResult(
        queried=False,
        malicious_count=0,
        suspicious_count=0,
        total_engines=0,
        reputation_class=ReputationClass.UNKNOWN,
        vt_url=None,
        cached=False,
        previous_flag_count=previous_flag_count,
        error=error,
    )


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json, os
    logging.basicConfig(level=logging.INFO)

    url     = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    api_key = sys.argv[2] if len(sys.argv) > 2 else os.getenv("VT_API_KEY", "")

    if not api_key:
        print("Usage: python threat_intel.py <url> <vt_api_key>")
        print("Or set VT_API_KEY environment variable")
        sys.exit(1)

    async def main():
        result = await check_threat_intel(url, api_key)
        print(json.dumps(result.model_dump(), indent=2))

    asyncio.run(main())