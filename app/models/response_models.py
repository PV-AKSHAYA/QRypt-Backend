"""
SafeQR — Response Models
Complete JSON output contract. Every field typed. Nothing implicit.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────

class Verdict(str, Enum):
    SAFE        = "SAFE"
    LOW_RISK    = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK   = "HIGH_RISK"
    CRITICAL    = "CRITICAL"


class TamperStatus(str, Enum):
    CLEAN      = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    TAMPERED   = "TAMPERED"


class URLMatchStatus(str, Enum):
    MATCH    = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN  = "UNKNOWN"


class ReputationClass(str, Enum):
    CLEAN      = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS  = "MALICIOUS"
    UNKNOWN    = "UNKNOWN"


# ─────────────────────────────────────────────
# SUB-MODELS
# ─────────────────────────────────────────────

class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left x pixel coordinate")
    y: int = Field(..., description="Top-left y pixel coordinate")
    width: int  = Field(..., description="Width in pixels")
    height: int = Field(..., description="Height in pixels")


class QRResult(BaseModel):
    found: bool            = Field(..., description="Was a QR code detected?")
    count: int             = Field(..., description="Number of QR codes found in image")
    raw_content: str       = Field(..., description="Decoded QR string (primary/first)")
    bounding_box: Optional[BoundingBox] = Field(None, description="Pixel location of QR in image")
    all_qr_contents: List[str]          = Field(default_factory=list, description="All decoded QRs if multiple")
    decode_error: Optional[str]         = Field(None, description="Error message if decode failed")


class PhysicalAnalysis(BaseModel):
    status: TamperStatus   = Field(..., description="Physical integrity verdict")
    tampered: bool         = Field(..., description="True if tampering detected")
    confidence: float      = Field(..., ge=0, le=100, description="Confidence 0–100")
    evidence: str          = Field(..., description="Human-readable finding")
    checks: dict           = Field(
        default_factory=dict,
        description="Individual check results: edge_anomaly, overlay_patch, obstruction, contrast"
    )


class RedirectHop(BaseModel):
    step: int              = Field(..., description="Hop number starting at 1")
    url: str               = Field(..., description="URL at this hop")
    status_code: Optional[int] = Field(None, description="HTTP status code")


class URLAnalysis(BaseModel):
    original_url: str      = Field(..., description="URL decoded directly from QR")
    final_url: str         = Field(..., description="URL after all redirects resolved")
    redirect_chain: List[RedirectHop] = Field(default_factory=list)
    redirect_count: int    = Field(..., description="Number of hops")
    is_shortened: bool     = Field(..., description="Was original URL a shortener?")
    domain: str            = Field(..., description="Parsed final domain")
    tld: str               = Field(..., description="Top-level domain")
    tld_risk_score: float  = Field(..., ge=0, le=100, description="TLD risk score")
    keyword_score: float   = Field(..., ge=0, le=100, description="Suspicious keyword score")
    ssl_valid: Optional[bool] = Field(None, description="SSL cert valid for final domain")
    domain_entropy: float  = Field(..., description="Shannon entropy of domain name")
    suspicious_keywords: List[str] = Field(default_factory=list, description="Matched suspicious keywords")
    error: Optional[str]   = Field(None, description="Error if URL analysis failed")


class ThreatIntelResult(BaseModel):
    queried: bool             = Field(..., description="Was VirusTotal queried?")
    malicious_count: int      = Field(default=0, description="Engines flagging as malicious")
    suspicious_count: int     = Field(default=0, description="Engines flagging as suspicious")
    total_engines: int        = Field(default=0, description="Total engines that scanned")
    reputation_class: ReputationClass = Field(default=ReputationClass.UNKNOWN)
    vt_url: Optional[str]     = Field(None, description="VirusTotal report link")
    cached: bool              = Field(default=False, description="Result from DB cache?")
    previous_flag_count: int  = Field(default=0, description="Times this domain was flagged in our DB")
    error: Optional[str]      = Field(None)


class AIContextResult(BaseModel):
    visual_context: str           = Field(..., description="What the poster/image appears to be")
    expected_brand: str           = Field(..., description="Brand or entity the image represents")
    url_match: URLMatchStatus     = Field(..., description="Does URL match expected brand?")
    impersonation_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float             = Field(..., ge=0.0, le=1.0)
    explanation: str              = Field(..., description="AI forensic reasoning")
    skipped: bool                 = Field(default=False, description="True if AI engine was skipped (QUICK mode)")
    error: Optional[str]          = Field(None)


class RiskBreakdown(BaseModel):
    physical_score: float     = Field(..., ge=0, le=100, description="Physical tamper score (weight 0.3)")
    threat_intel_score: float = Field(..., ge=0, le=100, description="Threat intel score (weight 0.3)")
    ai_context_score: float   = Field(..., ge=0, le=100, description="AI context score (weight 0.4)")
    physical_weight: float    = Field(default=0.3)
    threat_intel_weight: float= Field(default=0.3)
    ai_context_weight: float  = Field(default=0.4)


class RiskScore(BaseModel):
    score: int             = Field(..., ge=0, le=100, description="Final composite risk score")
    verdict: Verdict       = Field(..., description="Human verdict label")
    breakdown: RiskBreakdown
    suggested_action: str  = Field(..., description="What the user should do")


# ─────────────────────────────────────────────
# TOP-LEVEL RESPONSE
# ─────────────────────────────────────────────

class ScanResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    scan_id: str           = Field(..., description="UUID for this scan session")
    image_hash: str        = Field(..., description="SHA-256 of uploaded image")
    timestamp: str         = Field(..., description="ISO-8601 UTC timestamp")
    mode: str              = Field(..., description="Scan mode used")

    qr: QRResult
    physical: PhysicalAnalysis
    url: URLAnalysis
    threat_intel: ThreatIntelResult
    ai_context: AIContextResult
    risk: RiskScore


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    scan_id: Optional[str] = None
