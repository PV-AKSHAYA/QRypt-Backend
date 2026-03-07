"""
app/services/redirect_engine.py
─────────────────────────────────
URL Intelligence Engine.

Input  : original URL from QR code
Output : TechnicalLayerResult (minus virustotal — added by threat_intel.py)

Performs:
  1. Redirect chain unrolling   — follows all hops to final destination
  2. SSL validity check         — verifies HTTPS cert during request
  3. Domain entropy scoring     — detects randomly generated domains
  4. TLD risk scoring           — flags high-abuse TLDs
  5. Suspicious keyword scoring — detects phishing vocabulary in URL
  6. URL shortener detection    — flags known shortener services
  7. Domain parsing             — extracts clean domain components

Fail-safe: every step has a fallback.
If network is unreachable, returns best-effort result with what we know.
"""

import ssl
import socket
import logging
import requests
from urllib.parse import urlparse

from app.models.response_models import TechnicalLayerResult, VirusTotalResult, ReputationClass
from app.utils.url_utils import (
    compute_entropy,
    get_tld_risk,
    find_suspicious_keywords,
    is_shortener,
    check_ssl,
    parse_domain,
)
from app.core.config import settings

logger = logging.getLogger("safeqr.redirect_engine")


# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

MAX_HOPS = 10

REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ══════════════════════════════════════════════════════════════
#  STEP 1 — REDIRECT CHAIN UNROLLER
# ══════════════════════════════════════════════════════════════

def unroll_redirects(url: str) -> tuple[list[str], bool]:
    """
    Follow all HTTP redirects and return the full chain.

    Returns:
        chain      : list of URLs from original to final
        ssl_valid  : True if final URL has valid HTTPS cert

    Strategy:
        - Use HEAD requests (fast, no body download)
        - Follow Location headers manually (allow_redirects=False)
        - Detect redirect loops
        - Verify SSL on final hop
    """
    chain     = [url]
    ssl_valid = url.startswith("https://")
    visited   = {url}

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    current = url

    for hop in range(MAX_HOPS):
        try:
            response = session.head(
                current,
                allow_redirects=False,
                timeout=settings.HTTP_TIMEOUT,
                verify=True,           # raises SSLError if cert invalid
            )

            # Track SSL validity on each hop
            if current.startswith("https://"):
                ssl_valid = True       # cert verified (verify=True above)

            # Check for redirect
            if response.status_code in REDIRECT_STATUS_CODES:
                location = response.headers.get("Location", "")
                if not location:
                    break

                # Handle relative redirects
                if location.startswith("/"):
                    parsed  = urlparse(current)
                    location = f"{parsed.scheme}://{parsed.netloc}{location}"
                elif not location.startswith("http"):
                    parsed  = urlparse(current)
                    location = f"{parsed.scheme}://{parsed.netloc}/{location}"

                # Detect redirect loop
                if location in visited:
                    logger.warning(f"Redirect loop detected at {location}")
                    break

                visited.add(location)
                chain.append(location)
                current = location

            else:
                # Non-redirect response = we've reached the final URL
                break

        except requests.exceptions.SSLError:
            logger.warning(f"SSL cert invalid at {current}")
            ssl_valid = False
            # Continue — still report the chain
            break

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout at hop {hop}: {current}")
            break

        except requests.exceptions.TooManyRedirects:
            logger.warning("Too many redirects")
            break

        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error at {current}: {e}")
            break

        except Exception as e:
            logger.error(f"Unexpected error at hop {hop}: {e}")
            break

    logger.info(f"Redirect chain: {len(chain)} hops — final: {chain[-1][:60]}")
    return chain, ssl_valid


# ══════════════════════════════════════════════════════════════
#  STEP 2 — SSL DEEP CHECK
# ══════════════════════════════════════════════════════════════

def verify_ssl_cert(url: str) -> bool:
    """
    Perform an actual SSL handshake to verify certificate validity.
    More thorough than just checking scheme.
    Returns True if cert is valid and not expired.
    """
    if not url.startswith("https://"):
        return False

    parsed = urlparse(url)
    host   = parsed.netloc.split(":")[0]
    port   = 443

    try:
        ctx  = ssl.create_default_context()
        conn = ctx.wrap_socket(
            socket.create_connection((host, port), timeout=5),
            server_hostname=host,
        )
        conn.close()
        return True
    except ssl.SSLCertVerificationError:
        logger.warning(f"SSL cert verification failed for {host}")
        return False
    except ssl.SSLError as e:
        logger.warning(f"SSL error for {host}: {e}")
        return False
    except (socket.timeout, ConnectionRefusedError, OSError):
        # Network issue — don't penalise for connectivity
        return url.startswith("https://")
    except Exception as e:
        logger.debug(f"SSL check error for {host}: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  MAIN ANALYZE FUNCTION
# ══════════════════════════════════════════════════════════════

def analyze_url(original_url: str) -> TechnicalLayerResult:
    """
    Full URL intelligence analysis.

    Args:
        original_url: URL decoded from QR code

    Returns:
        TechnicalLayerResult with all fields populated.
        VirusTotal fields left at defaults (filled by threat_intel.py).
    """
    logger.info(f"URL analysis starting: {original_url[:80]}")

    # ── Step 1: Unroll redirects ──────────────────────────────
    try:
        redirect_chain, ssl_from_requests = unroll_redirects(original_url)
    except Exception as e:
        logger.error(f"Redirect unroll failed: {e}")
        redirect_chain     = [original_url]
        ssl_from_requests  = original_url.startswith("https://")

    final_url  = redirect_chain[-1]
    hop_count  = len(redirect_chain) - 1

    # ── Step 2: SSL verification ──────────────────────────────
    try:
        ssl_valid = verify_ssl_cert(final_url)
    except Exception:
        ssl_valid = ssl_from_requests

    # ── Step 3: Domain analysis ───────────────────────────────
    try:
        domain_info = parse_domain(final_url)
        domain_name = domain_info.get("domain", "")
    except Exception:
        domain_name = ""

    # ── Step 4: Entropy (on domain name only, not full URL) ───
    try:
        domain_entropy = round(compute_entropy(domain_name), 3)
    except Exception:
        domain_entropy = 0.0

    # ── Step 5: TLD risk ──────────────────────────────────────
    try:
        tld_risk = get_tld_risk(final_url)
    except Exception:
        tld_risk = 0.30

    # ── Step 6: Suspicious keywords ───────────────────────────
    try:
        # Check both original and final URL for keywords
        keywords = list(set(
            find_suspicious_keywords(original_url) +
            find_suspicious_keywords(final_url)
        ))
    except Exception:
        keywords = []

    # ── Step 7: Shortener detection ───────────────────────────
    try:
        shortener = is_shortener(original_url)
    except Exception:
        shortener = False

    logger.info(
        f"URL analysis complete | hops={hop_count} ssl={ssl_valid} "
        f"entropy={domain_entropy} tld_risk={tld_risk} "
        f"keywords={keywords} shortener={shortener}"
    )

    return TechnicalLayerResult(
        original_url        = original_url,
        final_url           = final_url,
        redirect_chain      = redirect_chain,
        hop_count           = hop_count,
        ssl_valid           = ssl_valid,
        is_shortener        = shortener,
        domain_entropy      = domain_entropy,
        tld_risk_score      = tld_risk,
        suspicious_keywords = keywords,
        domain_age_days     = None,       # not implemented — needs WHOIS
        virustotal          = VirusTotalResult(
            reputation_class=ReputationClass.UNKNOWN
        ),
    )