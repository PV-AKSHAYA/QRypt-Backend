"""
app/utils/url_utils.py
───────────────────────
URL parsing, entropy, TLD risk scoring, keyword detection.
All pure functions — no network calls here.
Used by redirect_engine.py.
"""

import re
import math
import logging
from urllib.parse import urlparse

import tldextract

logger = logging.getLogger("safeqr.url_utils")


# ══════════════════════════════════════════════════════════════
#  KNOWN URL SHORTENERS
# ══════════════════════════════════════════════════════════════

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "short.link", "cutt.ly", "rebrand.ly", "bl.ink", "tiny.cc",
    "is.gd", "buff.ly", "adf.ly", "shorte.st", "bc.vc",
    "lnkd.in", "fb.me", "youtu.be", "amzn.to", "page.link",
    "smarturl.it", "linktr.ee", "qr.ae", "s.id", "rb.gy",
}

# ══════════════════════════════════════════════════════════════
#  HIGH-RISK TLDs
# ══════════════════════════════════════════════════════════════
# Scores based on abuse frequency (0.0 = safe, 1.0 = very risky)

TLD_RISK_MAP = {
    # Extremely abused
    ".xyz":     0.85, ".top":     0.85, ".club":    0.80,
    ".online":  0.80, ".site":    0.78, ".live":    0.75,
    ".click":   0.90, ".link":    0.75, ".work":    0.72,
    ".loan":    0.92, ".win":     0.88, ".download":0.90,
    ".stream":  0.82, ".gq":      0.95, ".ml":      0.90,
    ".cf":      0.90, ".ga":      0.88, ".tk":      0.88,

    # Moderately risky
    ".info":    0.45, ".biz":     0.40, ".cc":      0.55,
    ".pw":      0.65, ".su":      0.60, ".ws":      0.50,
    ".to":      0.35, ".in":      0.25, ".co":      0.20,

    # Generally safe
    ".com":     0.05, ".org":     0.05, ".net":     0.08,
    ".edu":     0.01, ".gov":     0.01, ".io":      0.10,
    ".app":     0.08, ".dev":     0.06, ".tech":    0.15,
    ".uk":      0.05, ".de":      0.05, ".fr":      0.05,
    ".au":      0.05, ".ca":      0.05, ".jp":      0.05,
    ".in":      0.10,
}

DEFAULT_TLD_RISK = 0.30   # for unknown TLDs


# ══════════════════════════════════════════════════════════════
#  SUSPICIOUS KEYWORDS
# ══════════════════════════════════════════════════════════════

SUSPICIOUS_KEYWORDS = [
    # Financial fraud
    "refund", "claim", "reward", "prize", "winner", "bonus",
    "cashback", "free-money", "gift-card", "voucher",

    # Credential theft
    "login", "signin", "sign-in", "verify", "verification",
    "confirm", "account", "update", "secure", "security",
    "authenticate", "password", "credential",

    # Impersonation
    "paypal", "amazon", "apple", "google", "microsoft",
    "netflix", "bank", "irs", "gov", "tax", "invoice",
    "support", "helpdesk", "customer-service",

    # Urgency / scam patterns
    "urgent", "alert", "suspend", "locked", "limited",
    "expire", "click-here", "act-now", "immediate",
]


# ══════════════════════════════════════════════════════════════
#  FUNCTIONS
# ══════════════════════════════════════════════════════════════

def parse_domain(url: str) -> dict:
    """
    Extract domain components from a URL.
    Returns dict with: subdomain, domain, suffix, full_domain
    """
    extracted = tldextract.extract(url)
    return {
        "subdomain":   extracted.subdomain,
        "domain":      extracted.domain,
        "suffix":      extracted.suffix,
        "full_domain": extracted.registered_domain,  # e.g. 'example.com'
    }


def compute_entropy(text: str) -> float:
    """
    Shannon entropy of a string.
    High entropy (>3.5) = random-looking = likely generated domain.

    Formula: H = -sum(p * log2(p)) for each character frequency.

    Examples:
      'google'           → ~2.25  (low, human-readable)
      'xkq7z2mf9p'      → ~3.32  (high, random-looking)
      'aaaaaa'           → 0.0    (minimum)
    """
    if not text:
        return 0.0
    freq   = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def get_tld_risk(url: str) -> float:
    """
    Return TLD risk score (0.0–1.0) for the given URL.
    Higher = riskier TLD.
    """
    extracted = tldextract.extract(url)
    tld       = f".{extracted.suffix}" if extracted.suffix else ""

    # Check full TLD first (e.g. .co.uk), then just last part
    risk = TLD_RISK_MAP.get(tld)
    if risk is None:
        last = f".{tld.split('.')[-1]}" if "." in tld else tld
        risk = TLD_RISK_MAP.get(last, DEFAULT_TLD_RISK)

    return round(risk, 2)


def find_suspicious_keywords(url: str) -> list[str]:
    """
    Return list of suspicious keywords found in the URL.
    Checks both path and domain components.
    """
    url_lower = url.lower()
    found     = []
    for kw in SUSPICIOUS_KEYWORDS:
        # Use word-boundary style matching (separated by -, /, ., _)
        pattern = re.compile(r"(^|[\-\/\._=?&])" + re.escape(kw) + r"($|[\-\/\._=?&])")
        if pattern.search(url_lower):
            found.append(kw)
    return found


def is_shortener(url: str) -> bool:
    """Return True if URL uses a known URL shortener domain."""
    extracted = tldextract.extract(url)
    domain    = f"{extracted.domain}.{extracted.suffix}".lower()
    return domain in SHORTENER_DOMAINS


def check_ssl(url: str) -> bool:
    """
    Return True if URL uses HTTPS scheme.
    Note: Does NOT verify the certificate — just checks scheme.
    Full cert verification happens in redirect_engine.py during request.
    """
    return url.strip().lower().startswith("https://")


def get_url_components(url: str) -> dict:
    """
    Full breakdown of a URL into its components.
    Useful for logging and debugging.
    """
    parsed = urlparse(url)
    return {
        "scheme":   parsed.scheme,
        "netloc":   parsed.netloc,
        "path":     parsed.path,
        "query":    parsed.query,
        "fragment": parsed.fragment,
    }