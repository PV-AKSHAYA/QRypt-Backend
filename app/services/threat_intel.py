"""
app/services/threat_intel.py
─────────────────────────────
Threat Intelligence Layer — VirusTotal API.

Input  : final URL (after redirect unrolling)
Output : VirusTotalResult + updates TechnicalLayerResult.virustotal

Flow:
  1. Submit URL to VT for scanning
  2. Poll analysis results (max 3 attempts)
  3. Parse engine verdicts → malicious / suspicious / harmless counts
  4. Derive reputation_class from counts
  5. Return structured VirusTotalResult

Fail-safe rules:
  - API key missing      → return UNKNOWN, log warning, no crash
  - Rate limit hit       → return UNKNOWN with note
  - Network timeout      → return UNKNOWN with note
  - Any exception        → return UNKNOWN, never crash the scan
"""

import time
import logging
import requests

from app.models.response_models import VirusTotalResult, ReputationClass
from app.core.config import settings

logger = logging.getLogger("safeqr.threat_intel")


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

VT_BASE_URL     = "https://www.virustotal.com/api/v3"
VT_SUBMIT_URL   = f"{VT_BASE_URL}/urls"
VT_ANALYSIS_URL = f"{VT_BASE_URL}/analyses"

MAX_POLL_ATTEMPTS = 3
POLL_WAIT_SECONDS = 3     # wait between polls

# Thresholds for reputation class
MALICIOUS_THRESHOLD  = 3   # 3+ engines → MALICIOUS
SUSPICIOUS_THRESHOLD = 1   # 1-2 engines → SUSPICIOUS


# ══════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════

def _get_headers() -> dict:
    return {
        "x-apikey":     settings.VIRUSTOTAL_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept":       "application/json",
    }


def _derive_reputation(malicious: int, suspicious: int) -> ReputationClass:
    """
    Derive reputation class from engine counts.
    MALICIOUS  → 3+ engines flagged malicious
    SUSPICIOUS → 1-2 engines flagged malicious OR 3+ flagged suspicious
    CLEAN      → 0 malicious, <3 suspicious
    """
    if malicious >= MALICIOUS_THRESHOLD:
        return ReputationClass.MALICIOUS
    if malicious >= SUSPICIOUS_THRESHOLD or suspicious >= 3:
        return ReputationClass.SUSPICIOUS
    return ReputationClass.CLEAN


def _submit_url(url: str) -> str | None:
    """
    Submit URL to VirusTotal for analysis.
    Returns analysis ID or None on failure.
    """
    try:
        response = requests.post(
            VT_SUBMIT_URL,
            headers=_get_headers(),
            data=f"url={requests.utils.quote(url, safe='')}",
            timeout=settings.HTTP_TIMEOUT,
        )

        if response.status_code == 401:
            logger.error("VirusTotal: Invalid API key")
            return None

        if response.status_code == 429:
            logger.warning("VirusTotal: Rate limit hit")
            return None

        if response.status_code not in (200, 201):
            logger.warning(f"VirusTotal submit failed: HTTP {response.status_code}")
            return None

        data        = response.json()
        analysis_id = data.get("data", {}).get("id")

        if not analysis_id:
            logger.warning("VirusTotal: No analysis ID in response")
            return None

        logger.info(f"VirusTotal: URL submitted, analysis_id={analysis_id[:20]}...")
        return analysis_id

    except requests.exceptions.Timeout:
        logger.warning("VirusTotal: Submit request timed out")
        return None
    except Exception as e:
        logger.error(f"VirusTotal submit error: {e}")
        return None


def _poll_results(analysis_id: str) -> dict | None:
    """
    Poll VirusTotal for analysis results.
    Retries up to MAX_POLL_ATTEMPTS with POLL_WAIT_SECONDS between each.
    Returns raw stats dict or None.
    """
    headers = {
        "x-apikey": settings.VIRUSTOTAL_API_KEY,
        "Accept":   "application/json",
    }

    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        try:
            time.sleep(POLL_WAIT_SECONDS)

            response = requests.get(
                f"{VT_ANALYSIS_URL}/{analysis_id}",
                headers=headers,
                timeout=settings.HTTP_TIMEOUT,
            )

            if response.status_code == 429:
                logger.warning(f"VirusTotal: Rate limit on poll attempt {attempt}")
                break

            if response.status_code != 200:
                logger.warning(
                    f"VirusTotal poll attempt {attempt}: HTTP {response.status_code}"
                )
                continue

            data   = response.json()
            status = data.get("data", {}).get("attributes", {}).get("status", "")

            if status == "completed":
                stats = data["data"]["attributes"].get("stats", {})
                logger.info(f"VirusTotal analysis complete: {stats}")
                return stats

            elif status in ("queued", "in-progress"):
                logger.debug(
                    f"VirusTotal poll {attempt}/{MAX_POLL_ATTEMPTS}: status={status}"
                )
                continue

            else:
                logger.warning(f"VirusTotal unexpected status: {status}")
                break

        except requests.exceptions.Timeout:
            logger.warning(f"VirusTotal poll attempt {attempt} timed out")
        except Exception as e:
            logger.error(f"VirusTotal poll error (attempt {attempt}): {e}")
            break

    logger.warning("VirusTotal: Analysis did not complete in time")
    return None


# ══════════════════════════════════════════════════════════════
#  SAFE DEFAULT
# ══════════════════════════════════════════════════════════════

def _unknown_result(reason: str = "") -> VirusTotalResult:
    """
    Return a safe UNKNOWN result when VT is unavailable.
    Scan continues — VT is one layer of three.
    """
    if reason:
        logger.info(f"VirusTotal returning UNKNOWN: {reason}")
    return VirusTotalResult(
        malicious        = 0,
        suspicious       = 0,
        harmless         = 0,
        total_engines    = 0,
        reputation_class = ReputationClass.UNKNOWN,
    )


# ══════════════════════════════════════════════════════════════
#  MAIN CHECK FUNCTION
# ══════════════════════════════════════════════════════════════

def check_virustotal(url: str) -> VirusTotalResult:
    """
    Run full VirusTotal analysis on a URL.

    Args:
        url: The final URL (after redirect unrolling)

    Returns:
        VirusTotalResult — always returns, never raises
    """

    # ── Guard: API key must be configured ─────────────────────
    if not settings.VIRUSTOTAL_API_KEY:
        return _unknown_result("API key not configured")

    if not url or not url.startswith(("http://", "https://")):
        return _unknown_result(f"Invalid URL: {url[:50]}")

    logger.info(f"VirusTotal checking: {url[:80]}")

    # ── Step 1: Submit URL ────────────────────────────────────
    analysis_id = _submit_url(url)
    if not analysis_id:
        return _unknown_result("Submit failed")

    # ── Step 2: Poll for results ──────────────────────────────
    stats = _poll_results(analysis_id)
    if not stats:
        return _unknown_result("Analysis timed out")

    # ── Step 3: Parse stats ───────────────────────────────────
    malicious   = int(stats.get("malicious",   0))
    suspicious  = int(stats.get("suspicious",  0))
    harmless    = int(stats.get("harmless",    0))
    undetected  = int(stats.get("undetected",  0))
    total       = malicious + suspicious + harmless + undetected

    reputation  = _derive_reputation(malicious, suspicious)

    logger.info(
        f"VirusTotal result: malicious={malicious} suspicious={suspicious} "
        f"harmless={harmless} total={total} class={reputation}"
    )

    return VirusTotalResult(
        malicious        = malicious,
        suspicious       = suspicious,
        harmless         = harmless,
        total_engines    = total,
        reputation_class = reputation,
    )