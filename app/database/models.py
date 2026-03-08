"""
app/database/models.py
───────────────────────
MongoDB document schemas for SafeQR.

Collections:
  scans         — full scan result per image
  threat_memory — aggregated domain reputation over time
"""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
#  SCAN DOCUMENT — one per scan request
# ══════════════════════════════════════════════════════════════

class ScanDocument(BaseModel):
    """
    Stored in MongoDB `scans` collection.
    One document per scan — full audit trail.
    """
    scan_id:       str
    timestamp:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    image_hash:    str                      # MD5 of raw image bytes
    image_size:    int                      # bytes

    # QR
    raw_content:   str                      # decoded QR content
    original_url:  str
    final_url:     str
    final_domain:  str                      # e.g. "github.com"
    hop_count:     int

    # Layer results
    tampered:      bool
    tamper_confidence: int
    vt_malicious:  int
    vt_total:      int
    reputation:    str                      # CLEAN/SUSPICIOUS/MALICIOUS/UNKNOWN
    url_match:     str                      # YES/NO/UNCERTAIN
    impersonation: float

    # Final verdict
    risk_score:    int
    verdict:       str                      # SAFE/SUSPICIOUS/HIGH_RISK

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ══════════════════════════════════════════════════════════════
#  THREAT MEMORY DOCUMENT — one per unique domain
# ══════════════════════════════════════════════════════════════

class ThreatMemoryDocument(BaseModel):
    """
    Stored in MongoDB `threat_memory` collection.
    Aggregated reputation per domain — updated on every scan.
    """
    domain:             str                 # primary key — e.g. "taxrefund-claim.xyz"
    first_seen:         datetime
    last_seen:          datetime
    scan_count:         int  = 0
    high_risk_count:    int  = 0
    suspicious_count:   int  = 0
    safe_count:         int  = 0
    total_vt_malicious: int  = 0           # cumulative VirusTotal malicious hits
    last_verdict:       str  = "UNKNOWN"
    flagged:            bool = False        # True if ever HIGH_RISK

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ══════════════════════════════════════════════════════════════
#  HELPER — build ScanDocument from scan pipeline output
# ══════════════════════════════════════════════════════════════

def build_scan_document(
    scan_id:       str,
    image_hash:    str,
    image_size:    int,
    qr_result,
    physical,
    technical,
    ai_layer,
    risk,
) -> dict:
    """
    Build a MongoDB-ready dict from the scan pipeline results.
    Called at the end of scan.py before saving.
    """
    from app.utils.url_utils import parse_domain

    final_domain = parse_domain(technical.final_url) or technical.final_url

    return ScanDocument(
        scan_id           = scan_id,
        image_hash        = image_hash,
        image_size        = image_size,
        raw_content       = qr_result.raw_content,
        original_url      = qr_result.raw_content,
        final_url         = technical.final_url,
        final_domain      = final_domain,
        hop_count         = technical.hop_count,
        tampered          = physical.tampered,
        tamper_confidence = physical.confidence,
        vt_malicious      = technical.virustotal.malicious,
        vt_total          = technical.virustotal.total_engines,
        reputation        = technical.virustotal.reputation_class.value,
        url_match         = ai_layer.url_match.value,
        impersonation     = ai_layer.impersonation_probability,
        risk_score        = risk.score,
        verdict           = risk.verdict.value,
    ).model_dump()