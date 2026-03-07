"""
SafeQR — redirect_engine.py
URL Intelligence Engine.

Input : Raw URL string (from qr_extractor)
Output: URLAnalysis (full structured JSON)

Performs:
  1. Shortener detection
  2. Redirect unrolling — follows all hops to final URL
  3. Domain + TLD parsing
  4. TLD risk scoring
  5. Suspicious keyword scoring
  6. SSL validity check
  7. Domain entropy (Shannon)
  8. Structured JSON output

All I/O is async. Uses httpx with strict timeouts.
Never crashes — all errors captured into URLAnalysis.error
"""

import asyncio
import logging
import math
import re
import ssl
import socket
from collections import Counter
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import httpx
import tldextract

from app.models.response_models import URLAnalysis, RedirectHop

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MAX_REDIRECTS   = 10
REQUEST_TIMEOUT = 8.0   # seconds per hop
USER_AGENT      = "Mozilla/5.0 (SafeQR-Scanner/1.0)"


# ─────────────────────────────────────────────
# Data: Known URL Shorteners
# ─────────────────────────────────────────────

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "short.link", "tiny.cc", "is.gd", "buff.ly", "rebrand.ly",
    "cutt.ly", "shorturl.at", "bl.ink", "snip.ly", "clicky.me",
    "budurl.com", "bc.vc", "adf.ly", "sh.st", "link.tl",
    "mcaf.ee", "po.st", "qr.ae", "s.id", "zws.im",
}


# ─────────────────────────────────────────────
# Data: TLD Risk Scores (0-100)
# Higher = riskier. Based on abuse statistics.
# ─────────────────────────────────────────────

TLD_RISK_SCORES = {
    # Very high risk
    "tk": 95, "ml": 92, "ga": 90, "cf": 90, "gq": 88,
    "xyz": 75, "top": 72, "click": 70, "loan": 85,
    "download": 80, "racing": 78, "win": 72, "stream": 70,
    "gdn": 68, "bid": 75, "trade": 70, "webcam": 80,
    # Medium risk
    "info": 40, "biz": 38, "cc": 45, "pw": 55, "ws": 42,
    "in": 25, "co": 20, "io": 15, "me": 15, "tv": 20,
    # Low risk (trusted)
    "com": 5, "org": 5, "net": 8, "edu": 2, "gov": 1,
    "uk": 5, "de": 5, "fr": 5, "jp": 5, "au": 5,
    "ca": 5, "nl": 5, "se": 5, "no": 5, "fi": 5,
}

DEFAULT_TLD_RISK = 30  # For unknown TLDs


# ─────────────────────────────────────────────
# Data: Suspicious Keywords
# ─────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    # Phishing patterns
    "login", "signin", "sign-in", "account", "verify", "verification",
    "secure", "security", "update", "confirm", "banking", "bank",
    "paypal", "apple", "google", "microsoft", "amazon", "netflix",
    "facebook", "instagram", "whatsapp", "support", "helpdesk",
    # Urgency patterns
    "urgent", "alert", "warning", "suspended", "locked", "limited",
    "expire", "expired", "immediate", "action-required",
    # Reward/scam patterns
    "free", "winner", "won", "prize", "reward", "claim", "gift",
    "bonus", "offer", "deal", "discount", "coupon",
    # Credential harvesting
    "password", "credential", "auth", "oauth", "token", "reset",
    "recover", "unlock", "reactivate",
    # Payment scams
    "payment", "invoice", "billing", "checkout", "wallet", "crypto",
    "bitcoin", "transfer", "refund",
]


# ─────────────────────────────────────────────
# Shortener Detection
# ─────────────────────────────────────────────

def is_shortened_url(url: str) -> bool:
    """Check if URL uses a known shortener domain."""
    try:
        ext = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}".lower()
        return domain in SHORTENER_DOMAINS
    except Exception:
        return False


# ─────────────────────────────────────────────
# TLD Risk Scoring
# ─────────────────────────────────────────────

def score_tld_risk(tld: str) -> float:
    """Return risk score 0-100 for a given TLD."""
    tld = tld.lower().lstrip(".")
    return float(TLD_RISK_SCORES.get(tld, DEFAULT_TLD_RISK))


# ─────────────────────────────────────────────
# Suspicious Keyword Scoring
# ─────────────────────────────────────────────

def score_suspicious_keywords(url: str) -> Tuple[float, List[str]]:
    """
    Scan URL for suspicious keywords.
    Returns (score 0-100, list of matched keywords).
    """
    url_lower = url.lower()
    matched = []

    for keyword in SUSPICIOUS_KEYWORDS:
        # Match as word boundary or path segment
        pattern = re.compile(r'[\W_]' + re.escape(keyword) + r'[\W_]|' +
                             re.escape(keyword), re.IGNORECASE)
        if pattern.search(url_lower):
            matched.append(keyword)

    # Score: each keyword adds weight, capped at 100
    score = min(100.0, len(matched) * 15.0)
    return score, matched


# ─────────────────────────────────────────────
# Domain Entropy
# ─────────────────────────────────────────────

def compute_domain_entropy(domain: str) -> float:
    """
    Shannon entropy of domain name characters.
    High entropy (>4.0) suggests randomly generated domains
    used in DGA (domain generation algorithm) malware.

    Legitimate domains: 2.5-3.8
    DGA domains: 4.0+
    """
    if not domain:
        return 0.0

    domain = domain.lower()
    counts = Counter(domain)
    length = len(domain)
    entropy = -sum(
        (c / length) * math.log2(c / length)
        for c in counts.values()
        if c > 0
    )
    return round(entropy, 3)


# ─────────────────────────────────────────────
# SSL Validity
# ─────────────────────────────────────────────

def check_ssl(domain: str) -> Optional[bool]:
    """
    Check if domain has a valid SSL certificate.
    Returns True/False/None (None = could not check).
    """
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                return cert is not None
    except ssl.SSLCertVerificationError:
        return False
    except (socket.timeout, socket.gaierror, ConnectionRefusedError, OSError):
        return None  # Could not connect — not necessarily invalid


# ─────────────────────────────────────────────
# Redirect Unrolling
# ─────────────────────────────────────────────

async def unroll_redirects(url: str) -> Tuple[List[RedirectHop], str, Optional[str]]:
    """
    Follow redirect chain from original URL to final destination.
    Returns (hops, final_url, error_or_None).

    Uses HEAD requests first (faster), falls back to GET.
    """
    hops: List[RedirectHop] = []
    current_url = url
    error = None

    headers = {"User-Agent": USER_AGENT}

    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=REQUEST_TIMEOUT,
            headers=headers,
            verify=False,  # We check SSL separately
        ) as client:

            for step in range(1, MAX_REDIRECTS + 1):
                hops.append(RedirectHop(step=step, url=current_url, status_code=None))

                try:
                    # Try HEAD first
                    response = await client.head(current_url)
                except httpx.RequestError:
                    # Fallback to GET
                    try:
                        response = await client.get(current_url)
                    except httpx.RequestError as e:
                        error = f"Request failed at hop {step}: {str(e)}"
                        break

                # Update hop with status code
                hops[-1] = RedirectHop(
                    step=step,
                    url=current_url,
                    status_code=response.status_code
                )

                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location", "")
                    if not location:
                        break
                    # Handle relative redirects
                    if location.startswith("/"):
                        parsed = urlparse(current_url)
                        location = f"{parsed.scheme}://{parsed.netloc}{location}"
                    current_url = location
                else:
                    # Not a redirect — we've reached the final URL
                    break

            else:
                error = f"Redirect chain exceeded {MAX_REDIRECTS} hops — possible loop"

    except Exception as e:
        error = f"Redirect unrolling failed: {str(e)}"

    return hops, current_url, error


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

async def analyze_url(url: str) -> URLAnalysis:
    """
    Main entry point. Fully async.
    Accepts raw URL string from QR code.
    Returns fully-typed URLAnalysis.

    Never raises.
    """
    if not url or not url.strip():
        return _error_result(url, "Empty URL")

    # Ensure URL has a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    original_url = url

    # ── Shortener detection ───────────────────
    shortened = is_shortened_url(url)

    # ── Redirect unrolling ────────────────────
    try:
        hops, final_url, redirect_error = await unroll_redirects(url)
    except Exception as e:
        return _error_result(url, f"Redirect engine crashed: {e}")

    # ── Parse final domain ────────────────────
    try:
        ext = tldextract.extract(final_url)
        domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
        tld    = ext.suffix or ""
    except Exception as e:
        return _error_result(url, f"Domain parsing failed: {e}")

    # ── Scores ────────────────────────────────
    tld_risk              = score_tld_risk(tld)
    keyword_score, matched_keywords = score_suspicious_keywords(final_url)
    entropy               = compute_domain_entropy(ext.domain)

    # ── SSL check (sync — run in thread) ─────
    ssl_valid: Optional[bool] = None
    if domain:
        try:
            ssl_valid = await asyncio.get_event_loop().run_in_executor(
                None, check_ssl, domain
            )
        except Exception:
            ssl_valid = None

    logger.info(
        f"URL analysis complete: {original_url} → {final_url} "
        f"({len(hops)} hops, tld_risk={tld_risk}, keywords={len(matched_keywords)})"
    )

    return URLAnalysis(
        original_url=original_url,
        final_url=final_url,
        redirect_chain=hops,
        redirect_count=max(0, len(hops) - 1),
        is_shortened=shortened,
        domain=domain,
        tld=tld,
        tld_risk_score=tld_risk,
        keyword_score=keyword_score,
        ssl_valid=ssl_valid,
        domain_entropy=entropy,
        suspicious_keywords=matched_keywords,
        error=redirect_error,
    )


def _error_result(url: str, error: str) -> URLAnalysis:
    return URLAnalysis(
        original_url=url,
        final_url=url,
        redirect_chain=[],
        redirect_count=0,
        is_shortened=False,
        domain="",
        tld="",
        tld_risk_score=0.0,
        keyword_score=0.0,
        ssl_valid=None,
        domain_entropy=0.0,
        suspicious_keywords=[],
        error=error,
    )


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO)

    url = sys.argv[1] if len(sys.argv) > 1 else "https://bit.ly/3example"

    async def main():
        result = await analyze_url(url)
        print(json.dumps(result.model_dump(), indent=2))

    asyncio.run(main())