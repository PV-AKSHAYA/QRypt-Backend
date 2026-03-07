"""
SafeQR — Request Models
All inbound API contracts. Nothing changes after this.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from enum import Enum


class ScanMode(str, Enum):
    FULL = "full"          # All engines (default)
    QUICK = "quick"        # Skip AI context, fast verdict
    FORENSIC = "forensic"  # Deep analysis, slower


class ScanRequest(BaseModel):
    """
    Primary /scan endpoint request.
    Image arrives as multipart/form-data (handled by FastAPI directly).
    This model carries optional metadata alongside the upload.
    """
    mode: ScanMode = Field(
        default=ScanMode.FULL,
        description="Analysis depth. FULL=all engines, QUICK=no AI, FORENSIC=deep"
    )
    include_raw_qr: bool = Field(
        default=False,
        description="Include raw decoded QR bytes in response"
    )
    caller_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional client identifier for rate limiting / audit"
    )

    class Config:
        use_enum_values = True


class ReScanRequest(BaseModel):
    """
    Re-run analysis on a previously scanned hash (DB lookup).
    Useful for re-checking flagged QR codes with fresh threat intel.
    """
    image_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of the original image"
    )
    refresh_threat_intel: bool = Field(
        default=True,
        description="Re-query VirusTotal even if cached"
    )

    @validator("image_hash")
    def must_be_hex(cls, v):
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("image_hash must be a valid hex string")
        return v.lower()


class FeedbackRequest(BaseModel):
    """
    User-submitted correction on a verdict.
    Feeds back into accuracy tracking.
    """
    scan_id: str = Field(..., description="UUID of the original scan result")
    user_verdict: str = Field(
        ...,
        pattern="^(SAFE|LOW_RISK|MEDIUM_RISK|HIGH_RISK|CRITICAL)$",
        description="User's corrected verdict"
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional explanation from user"
    )