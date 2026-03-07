"""
app/api/scan.py
────────────────
The /scan endpoint — orchestrates all 6 services in order.

Flow:
  1. Validate image
  2. Extract QR
  3. Run physical analyzer
  4. Run redirect + URL engine
  5. Run VirusTotal (skippable)
  6. Run AI context (skippable)
  7. Calculate risk score
  8. Check threat memory (MongoDB)
  9. Save scan to DB
 10. Return full ScanResponse

Each service is wrapped in try/except.
If one service fails, scan continues with a safe default.
The endpoint NEVER returns 500 unless image is completely invalid.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from app.models.response_models import (
    ScanResponse,
    ErrorResponse,
    ThreatMemory,
    QRResult,
    BoundingBox,
)
from app.models.request_models import ScanRequest
from app.services.qr_extractor    import extract_qr, NoQRFoundError, InvalidImageError
from app.services.physical_analyzer import analyze_physical
from app.services.redirect_engine   import analyze_url
from app.services.threat_intel      import check_virustotal
from app.services.ai_context_engine import analyze_context
from app.services.risk_engine       import calculate_risk
from app.utils.image_utils          import validate_image_bytes, compute_image_hash
from app.utils.validators           import is_valid_url, normalise_url
from app.core.config                import settings

logger = logging.getLogger("safeqr.scan")

router = APIRouter()


# ══════════════════════════════════════════════════════════════
#  THREAT MEMORY — MongoDB lookup/save
#  Wrapped separately so DB failure never kills the scan
# ══════════════════════════════════════════════════════════════

async def _check_threat_memory(domain: str) -> ThreatMemory:
    """Look up domain in MongoDB. Returns empty ThreatMemory if DB unavailable."""
    try:
        from app.database.db import get_db
        db         = get_db()
        collection = db["scans"]
        doc        = await collection.find_one(
            {"final_domain": domain, "verdict": {"$in": ["HIGH_RISK", "SUSPICIOUS"]}}
        )
        if doc:
            count = await collection.count_documents({"final_domain": domain})
            return ThreatMemory(
                seen_before         = True,
                previous_scan_count = count,
                first_seen          = doc.get("timestamp", ""),
                last_verdict        = doc.get("verdict", ""),
            )
    except Exception as e:
        logger.debug(f"Threat memory lookup skipped: {e}")
    return ThreatMemory()


async def _save_scan(scan_id: str, img_hash: str, result: ScanResponse) -> None:
    """Save scan result to MongoDB. Silently skips if DB unavailable."""
    try:
        from app.database.db import get_db
        from urllib.parse import urlparse
        db         = get_db()
        collection = db["scans"]

        parsed       = urlparse(result.technical_layer.final_url)
        final_domain = parsed.netloc or result.technical_layer.final_url

        await collection.insert_one({
            "scan_id":      scan_id,
            "image_hash":   img_hash,
            "original_url": result.qr.raw_content,
            "final_url":    result.technical_layer.final_url,
            "final_domain": final_domain,
            "risk_score":   result.risk.score,
            "verdict":      result.risk.verdict.value,
            "timestamp":    result.timestamp,
        })
        logger.info(f"Scan saved to DB: {scan_id}")
    except Exception as e:
        logger.debug(f"DB save skipped: {e}")


# ══════════════════════════════════════════════════════════════
#  MAIN ENDPOINT
# ══════════════════════════════════════════════════════════════

@router.post(
    "/scan",
    response_model      = ScanResponse,
    responses           = {
        400: {"model": ErrorResponse, "description": "No QR found / invalid image"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
    summary             = "Forensic QR scan",
    description         = "Upload a QR code image for full 3-layer forensic analysis.",
)
async def scan_qr(
    image:           UploadFile        = File(..., description="Image file containing QR code"),
    context_hint:    Optional[str]     = Form(None,  description="Optional hint e.g. 'bank poster'"),
    skip_virustotal: bool              = Form(False, description="Skip VirusTotal check"),
    skip_ai:         bool              = Form(False, description="Skip AI context analysis"),
):
    scan_id   = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info(f"[{scan_id}] Scan started — file={image.filename}")

    # ── Step 1: Read + validate image ────────────────────────
    try:
        img_bytes = await image.read()
        validate_image_bytes(img_bytes, image.content_type or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    img_hash = compute_image_hash(img_bytes)
    logger.info(f"[{scan_id}] Image loaded — {len(img_bytes)} bytes hash={img_hash[:8]}")

    # ── Step 2: Extract QR ────────────────────────────────────
    try:
        qr_result = extract_qr(img_bytes)
        logger.info(f"[{scan_id}] QR decoded: {qr_result.raw_content[:80]}")
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NoQRFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"QR extraction failed: {e}")

    # ── Step 3: Normalise URL ─────────────────────────────────
    raw_url = qr_result.raw_content
    if not is_valid_url(raw_url):
        raw_url = normalise_url(raw_url)

    # ── Step 4: Physical tamper analysis ─────────────────────
    try:
        physical_result = analyze_physical(img_bytes)
        logger.info(
            f"[{scan_id}] Physical: tampered={physical_result.tampered} "
            f"confidence={physical_result.confidence}"
        )
    except Exception as e:
        logger.error(f"[{scan_id}] Physical analyzer error: {e}")
        from app.models.response_models import PhysicalLayerResult
        physical_result = PhysicalLayerResult(
            tampered=False, confidence=0,
            evidence=f"Physical analysis unavailable: {e}"
        )

    # ── Step 5: URL intelligence + redirect unrolling ─────────
    try:
        tech_result = analyze_url(raw_url)
        logger.info(
            f"[{scan_id}] URL: hops={tech_result.hop_count} "
            f"final={tech_result.final_url[:60]}"
        )
    except Exception as e:
        logger.error(f"[{scan_id}] URL engine error: {e}")
        from app.models.response_models import (
            TechnicalLayerResult, VirusTotalResult, ReputationClass
        )
        tech_result = TechnicalLayerResult(
            original_url=raw_url, final_url=raw_url,
            redirect_chain=[raw_url], hop_count=0,
            ssl_valid=raw_url.startswith("https://"),
            is_shortener=False, domain_entropy=0.0,
            tld_risk_score=0.0, suspicious_keywords=[],
            domain_age_days=None,
            virustotal=VirusTotalResult(
                reputation_class=ReputationClass.UNKNOWN
            ),
        )

    # ── Step 6: VirusTotal ────────────────────────────────────
    if not skip_virustotal:
        try:
            vt_result = check_virustotal(tech_result.final_url)
            tech_result.virustotal = vt_result
            logger.info(
                f"[{scan_id}] VT: malicious={vt_result.malicious} "
                f"class={vt_result.reputation_class}"
            )
        except Exception as e:
            logger.error(f"[{scan_id}] VirusTotal error: {e}")
            # virustotal already defaults to UNKNOWN — no action needed
    else:
        logger.info(f"[{scan_id}] VirusTotal skipped")

    # ── Step 7: AI context analysis ────────────────────
    if not skip_ai:
        try:
            ai_result = analyze_context(
                img_bytes    = img_bytes,
                final_url    = tech_result.final_url,
                context_hint = context_hint or "",
            )
            logger.info(
                f"[{scan_id}] AI: url_match={ai_result.url_match} "
                f"impersonation={ai_result.impersonation_probability}"
            )
        except Exception as e:
            logger.error(f"[{scan_id}] AI context error: {e}")
            from app.models.response_models import AILayerResult, URLMatch
            ai_result = AILayerResult(
                visual_context="AI analysis unavailable",
                expected_brand="Unknown",
                url_match=URLMatch.UNCERTAIN,
                impersonation_probability=0.041,
                confidence=1.0,
                explanation=f"AI context engine error: {str(e)[:80]}",
            )
    else:
        logger.info(f"[{scan_id}] AI analysis skipped")
        from app.models.response_models import AILayerResult, URLMatch
        ai_result = AILayerResult(
            visual_context="AI analysis skipped",
            expected_brand="Unknown",
            url_match=URLMatch.UNCERTAIN,
            impersonation_probability=0.041,
            confidence=1.0,
            explanation="AI context analysis was skipped by request.",
        )

    # ── Step 8: Risk scoring ──────────────────────────────────
    try:
        risk_result = calculate_risk(physical_result, tech_result, ai_result)
        logger.info(
            f"[{scan_id}] Risk: score={risk_result.score} "
            f"verdict={risk_result.verdict}"
        )
    except Exception as e:
        logger.error(f"[{scan_id}] Risk engine error: {e}")
        from app.models.response_models import RiskResult, RiskBreakdown, Verdict
        risk_result = RiskResult(
            score=50, verdict=Verdict.SUSPICIOUS,
            breakdown=RiskBreakdown(
                physical_score=0, threat_intel_score=0, ai_context_score=0
            )
        )

    # ── Step 9: Threat memory lookup ──────────────────────────
    try:
        from urllib.parse import urlparse
        domain        = urlparse(tech_result.final_url).netloc
        threat_memory = await _check_threat_memory(domain)
        if threat_memory.seen_before:
            logger.info(
                f"[{scan_id}] Threat memory: domain seen "
                f"{threat_memory.previous_scan_count} times before"
            )
    except Exception as e:
        logger.debug(f"[{scan_id}] Threat memory error: {e}")
        threat_memory = ThreatMemory()

    # ── Step 10: Assemble final response ──────────────────────
    response = ScanResponse(
        scan_id         = scan_id,
        timestamp       = timestamp,
        threat_memory   = threat_memory,
        qr              = qr_result,
        physical_layer  = physical_result,
        technical_layer = tech_result,
        ai_layer        = ai_result,
        risk            = risk_result,
    )

    # ── Step 11: Save to DB (non-blocking) ────────────────────
    await _save_scan(scan_id, img_hash, response)

    logger.info(
        f"[{scan_id}] Scan complete — "
        f"verdict={risk_result.verdict} score={risk_result.score}"
    )

    return response