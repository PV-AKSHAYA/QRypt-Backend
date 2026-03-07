"""
app/models/response_models.py
──────────────────────────────
All Pydantic response models for the /scan endpoint.
These are the EXACT shapes every service must return.
Lock this early — frontend and backend both depend on it.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


#  ENUMS
class Verdict(str, Enum):
    SAFE        = "SAFE"
    SUSPICIOUS  = "SUSPICIOUS"
    HIGH_RISK   = "HIGH_RISK"


class ReputationClass(str, Enum):
    CLEAN       = "CLEAN"
    SUSPICIOUS  = "SUSPICIOUS"
    MALICIOUS   = "MALICIOUS"
    UNKNOWN     = "UNKNOWN"


class URLMatch(str, Enum):
    YES         = "YES"
    NO          = "NO"
    UNCERTAIN   = "UNCERTAIN"


#  LAYER 0 — QR EXTRACTION

class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    w: int = Field(..., description="Width in pixels")
    h: int = Field(..., description="Height in pixels")


class QRResult(BaseModel):
    raw_content:  str             = Field(..., description="Decoded QR string")
    bounding_box: BoundingBox     = Field(..., description="QR location in image")
    qr_count:     int             = Field(..., description="Total QR codes found in image")


#  LAYER 1 — PHYSICAL TAMPER ANALYSIS

class PhysicalLayerResult(BaseModel):
    tampered:   bool  = Field(..., description="True if tampering detected")
    confidence: int   = Field(..., ge=0, le=100, description="Confidence 0-100")
    evidence:   str   = Field(..., description="Human-readable evidence string")

    # Example:
    # {
    #   "tampered": true,
    #   "confidence": 78,
    #   "evidence": "Overlay contour detected at (142, 88). Double-edge signature found."
    # }


#  LAYER 2 — URL / TECHNICAL INTELLIGENCE

class VirusTotalResult(BaseModel):
    malicious:        int              = Field(0,  description="Engines flagging as malicious")
    suspicious:       int              = Field(0,  description="Engines flagging as suspicious")
    harmless:         int              = Field(0,  description="Engines flagging as harmless")
    total_engines:    int              = Field(0,  description="Total engines queried")
    reputation_class: ReputationClass  = Field(ReputationClass.UNKNOWN)


class TechnicalLayerResult(BaseModel):
    original_url:        str               = Field(..., description="URL decoded from QR")
    final_url:           str               = Field(..., description="URL after all redirects")
    redirect_chain:      list[str]         = Field(..., description="Full hop-by-hop chain")
    hop_count:           int               = Field(..., description="Number of redirects")
    ssl_valid:           bool              = Field(..., description="HTTPS with valid cert")
    is_shortener:        bool              = Field(..., description="Known URL shortener used")
    domain_entropy:      float             = Field(..., description="Shannon entropy of domain")
    tld_risk_score:      float             = Field(..., ge=0.0, le=1.0, description="TLD risk 0-1")
    suspicious_keywords: list[str]         = Field(..., description="Risky words found in URL")
    domain_age_days:     Optional[int]     = Field(None, description="Domain age, None if unknown")
    virustotal:          VirusTotalResult  = Field(...)


#  LAYER 3 — AI CONTEXT ANALYSIS

class AILayerResult(BaseModel):
    visual_context:           str      = Field(..., description="What AI sees in the image")
    expected_brand:           str      = Field(..., description="Brand/org suggested by visual")
    url_match:                URLMatch = Field(..., description="Does URL match expected brand?")
    impersonation_probability: float   = Field(..., ge=0.0, le=1.0, description="0.0 to 1.0")
    confidence:               float    = Field(..., ge=0.0, le=1.0, description="AI confidence")
    explanation:              str      = Field(..., description="One-line forensic verdict")

    # Example:
    # {
    #   "visual_context": "Official government tax payment poster",
    #   "expected_brand": "IRS / Tax Authority",
    #   "url_match": "NO",
    #   "impersonation_probability": 0.92,
    #   "confidence": 0.88,
    #   "explanation": "QR leads to unregistered domain, not a government TLD"
    # }


#  RISK SCORING

class RiskBreakdown(BaseModel):
    physical_score:     float = Field(..., description="Physical layer contribution (max 30)")
    threat_intel_score: float = Field(..., description="Threat intel contribution (max 30)")
    ai_context_score:   float = Field(..., description="AI context contribution (max 40)")


class RiskResult(BaseModel):
    score:     int           = Field(..., ge=0, le=100, description="Final risk score 0-100")
    verdict:   Verdict       = Field(..., description="SAFE / SUSPICIOUS / HIGH_RISK")
    breakdown: RiskBreakdown = Field(...)


#  THREAT MEMORY (MongoDB lookup)

class ThreatMemory(BaseModel):
    seen_before:         bool          = Field(False)
    previous_scan_count: int           = Field(0)
    first_seen:          Optional[str] = Field(None, description="ISO timestamp")
    last_verdict:        Optional[str] = Field(None)


#  FINAL /scan RESPONSE

class ScanResponse(BaseModel):
    scan_id:        str                  = Field(..., description="Unique scan UUID")
    timestamp:      str                  = Field(..., description="ISO 8601 timestamp")
    threat_memory:  ThreatMemory         = Field(...)
    qr:             QRResult             = Field(...)
    physical_layer: PhysicalLayerResult  = Field(...)
    technical_layer: TechnicalLayerResult = Field(...)
    ai_layer:       AILayerResult        = Field(...)
    risk:           RiskResult           = Field(...)

#  ERROR RESPONSE

class ErrorResponse(BaseModel):
    error:   str           = Field(..., description="Short error code")
    detail:  str           = Field(..., description="Human-readable message")
    scan_id: Optional[str] = Field(None)