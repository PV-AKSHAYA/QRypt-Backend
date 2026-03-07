"""
app/models/request_models.py
─────────────────────────────
Request validation models for /scan endpoint.
FastAPI uses these to validate incoming data before
it reaches any service.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ScanRequest(BaseModel):
    """
    Metadata that can optionally accompany the image upload.
    The image itself is sent as multipart/form-data (UploadFile).
    These are optional query params or form fields.
    """

    # Optional hint from client about what kind of QR this might be
    # Helps AI context engine narrow down its analysis
    context_hint: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional hint e.g. 'bank poster', 'restaurant menu'",
        examples=["government tax poster", "restaurant menu", "parking meter"],
    )

    # Skip VirusTotal if caller wants a fast result (demo mode)
    skip_virustotal: bool = Field(
        False,
        description="Skip VirusTotal check for faster response (reduces accuracy)",
    )

    # Skip AI context engine
    skip_ai: bool = Field(
        False,
        description="Skip AI context analysis",
    )