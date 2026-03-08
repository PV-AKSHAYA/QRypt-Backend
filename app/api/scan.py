"""
app/api/scan.py
────────────────
Two endpoints:
  POST /api/v1/scan          — upload image, returns scan_id
  WS   /api/v1/scan/ws/{id} — real-time progress + final result

Flow (WebSocket):
  Stage 1 — QR extraction
  Stage 2 — Physical tamper detection
  Stage 3 — Redirect chain analysis
  Stage 4 — VirusTotal threat intel
  Stage 5 — AI context analysis
  Stage 6 — Risk scoring + DB save + final result
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from app.models.response_models import (
    ScanResponse, ErrorResponse, ThreatMemory,
    QRResult, BoundingBox, Verdict, RiskBreakdown, RiskResult,
    AILayerResult, URLMatch, PhysicalLayerResult,
)
from app.services.qr_extractor      import extract_qr, NoQRFoundError, InvalidImageError
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

# Pending scans — holds image bytes between POST and WebSocket connection
pending_scans: Dict[str, dict] = {}


# ══════════════════════════════════════════════════════════════
#  THREAT MEMORY helpers — use ThreatMemoryEngine, never crash
# ══════════════════════════════════════════════════════════════

async def _check_threat_memory(domain: str) -> ThreatMemory:
    try:
        from app.database.db import ThreatMemoryEngine
        mem = await ThreatMemoryEngine.lookup(domain)
        return ThreatMemory(
            seen_before         = mem["seen_before"],
            previous_scan_count = mem["previous_scan_count"],
            first_seen          = mem.get("first_seen") or "",
            last_verdict        = mem.get("last_verdict") or "",
        )
    except Exception as e:
        logger.debug(f"Threat memory lookup skipped: {e}")
    return ThreatMemory()


async def _save_scan(scan_id: str, img_hash: str, image_size: int, result: ScanResponse) -> None:
    try:
        from app.database.db import ThreatMemoryEngine
        from app.database.models import build_scan_document
        from urllib.parse import urlparse

        final_domain = urlparse(result.technical_layer.final_url).netloc \
                       or result.technical_layer.final_url

        scan_doc = build_scan_document(
            scan_id    = scan_id,
            image_hash = img_hash,
            image_size = image_size,
            qr_result  = result.qr,
            physical   = result.physical_layer,
            technical  = result.technical_layer,
            ai_layer   = result.ai_layer,
            risk       = result.risk,
        )

        await ThreatMemoryEngine.save_scan(scan_doc)
        await ThreatMemoryEngine.record(
            domain       = final_domain,
            verdict      = result.risk.verdict.value,
            vt_malicious = result.technical_layer.virustotal.malicious,
        )
        logger.info(f"Scan saved + threat memory updated: {scan_id}")
    except Exception as e:
        logger.error(f"DB save failed for scan {scan_id}: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════
#  GET /history — fetch last 50 scans
# ══════════════════════════════════════════════════════════════

@router.get(
    "/history",
    summary     = "Fetch recent scan history",
    description = "Returns the last 50 scan records from the database.",
)
async def get_history():
    try:
        from app.database.db import get_db
        db = get_db()
        
        cursor = db.scans.find({}, {
            "scan_id": 1, 
            "timestamp": 1, 
            "final_domain": 1, 
            "verdict": 1, 
            "risk_score": 1,
            "_id": 0
        }).sort("timestamp", -1).limit(50)
        
        history = await cursor.to_list(length=50)
        
        # Rename final_domain to domain for frontend compatibility
        for item in history:
            item["domain"] = item.pop("final_domain", "unknown")
            if isinstance(item.get("timestamp"), datetime):
                item["timestamp"] = item["timestamp"].isoformat()
                
        return history
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return []


# ══════════════════════════════════════════════════════════════
#  POST /scan — upload image, get scan_id back immediately
# ══════════════════════════════════════════════════════════════

@router.post(
    "/scan",
    summary     = "Upload QR image — returns scan_id for WebSocket",
    description = "Upload the QR image. You get a scan_id back instantly. "
                  "Then connect to WS /api/v1/scan/ws/{scan_id} for live progress.",
    responses   = {400: {"model": ErrorResponse}},
)
async def upload_for_scan(
    image:           UploadFile    = File(...),
    context_hint:    Optional[str] = Form(None),
    skip_virustotal: bool          = Form(False),
    skip_ai:         bool          = Form(False),
):
    scan_id = str(uuid.uuid4())

    try:
        img_bytes = await image.read()
        validate_image_bytes(img_bytes, image.content_type or "")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    pending_scans[scan_id] = {
        "bytes":        img_bytes,
        "size":         len(img_bytes),
        "filename":     image.filename or "unknown",
        "context_hint": context_hint,
        "skip_vt":      skip_virustotal,
        "skip_ai":      skip_ai,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

    logger.info(f"[{scan_id}] Image uploaded — file={image.filename} size={len(img_bytes)}B")
    return {"scan_id": scan_id}


# ══════════════════════════════════════════════════════════════
#  WS /scan/ws/{scan_id} — real-time progress + final result
# ══════════════════════════════════════════════════════════════

@router.websocket("/scan/ws/{scan_id}")
async def scan_websocket(websocket: WebSocket, scan_id: str):
    await websocket.accept()

    if scan_id not in pending_scans:
        await websocket.send_json({"type": "error", "message": "Scan ID not found or expired"})
        await websocket.close()
        return

    data      = pending_scans.pop(scan_id)
    img_bytes = data["bytes"]
    img_hash  = compute_image_hash(img_bytes)

    async def progress(stage: int, message: str):
        await websocket.send_json({"type": "progress", "stage": stage, "total": 6, "message": message})

    try:
        # ── Stage 1: QR Extraction ────────────────────────────
        await progress(1, "Decoding QR matrix...")
        try:
            qr_result = extract_qr(img_bytes)
        except (NoQRFoundError, InvalidImageError) as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close()
            return

        raw_url = qr_result.raw_content
        if not is_valid_url(raw_url):
            raw_url = normalise_url(raw_url)

        logger.info(f"[{scan_id}] QR decoded: {raw_url}")

        # ── Stage 2: Physical Tamper Detection ────────────────
        await progress(2, "Scanning physical integrity...")
        try:
            physical_result = analyze_physical(img_bytes)
            logger.info(f"[{scan_id}] Physical: tampered={physical_result.tampered} confidence={physical_result.confidence}")
        except Exception as e:
            logger.error(f"[{scan_id}] Physical analyzer error: {e}")
            physical_result = PhysicalLayerResult(
                tampered=False, confidence=0,
                evidence=f"Physical analysis unavailable: {e}"
            )

        # ── Stage 3: Redirect Chain + URL Intelligence ────────
        await progress(3, "Unrolling redirect chain...")
        try:
            tech_result = analyze_url(raw_url)
            logger.info(f"[{scan_id}] URL: hops={tech_result.hop_count} final={tech_result.final_url}")
        except Exception as e:
            logger.error(f"[{scan_id}] URL analysis error: {e}")
            from app.models.response_models import TechnicalLayerResult, VirusTotalResult, ReputationClass
            tech_result = TechnicalLayerResult(
                original_url=raw_url, final_url=raw_url,
                redirect_chain=[], hop_count=0, ssl_valid=False,
                is_shortener=False, domain_entropy=0.0, tld_risk_score=0.0,
                suspicious_keywords=[],
                virustotal=VirusTotalResult(malicious=0, suspicious=0, harmless=0,
                                            total_engines=0, reputation_class=ReputationClass.UNKNOWN)
            )

        # ── Stage 4: VirusTotal ───────────────────────────────
        await progress(4, "Querying VirusTotal threat intelligence...")
        if not data["skip_vt"]:
            try:
                vt_result = check_virustotal(tech_result.final_url)
                tech_result.virustotal = vt_result
                logger.info(f"[{scan_id}] VT: malicious={vt_result.malicious}")
            except Exception as e:
                logger.error(f"[{scan_id}] VirusTotal error: {e}")
        else:
            logger.info(f"[{scan_id}] VirusTotal skipped")

        # ── Stage 5: AI Context Analysis ─────────────────────
        await progress(5, "Running AI forensic analysis...")
        if not data["skip_ai"]:
            try:
                ai_result = analyze_context(
                    img_bytes    = img_bytes,
                    final_url    = tech_result.final_url,
                    context_hint = data["context_hint"] or "",
                )
                logger.info(f"[{scan_id}] AI: url_match={ai_result.url_match} impersonation={ai_result.impersonation_probability}")
            except Exception as e:
                logger.error(f"[{scan_id}] AI context error: {e}")
                ai_result = AILayerResult(
                    visual_context="AI analysis unavailable",
                    expected_brand="Unknown",
                    url_match=URLMatch.UNCERTAIN,
                    impersonation_probability=0.5,
                    confidence=0.0,
                    explanation=f"AI error: {str(e)[:80]}",
                )
        else:
            logger.info(f"[{scan_id}] AI skipped")
            ai_result = AILayerResult(
                visual_context="AI analysis skipped",
                expected_brand="Unknown",
                url_match=URLMatch.UNCERTAIN,
                impersonation_probability=0.0,
                confidence=0.0,
                explanation="AI context analysis was skipped by request.",
            )

        # ── Stage 6: Risk Score + DB Save ─────────────────────
        await progress(6, "Computing forensic risk score...")
        risk_result = calculate_risk(physical_result, tech_result, ai_result)
        logger.info(f"[{scan_id}] Risk: score={risk_result.score} verdict={risk_result.verdict}")

        from urllib.parse import urlparse
        domain        = urlparse(tech_result.final_url).netloc
        threat_memory = await _check_threat_memory(domain)

        response = ScanResponse(
            scan_id         = scan_id,
            timestamp       = data["timestamp"],
            threat_memory   = threat_memory,
            qr              = qr_result,
            physical_layer  = physical_result,
            technical_layer = tech_result,
            ai_layer        = ai_result,
            risk            = risk_result,
        )

        await _save_scan(scan_id, img_hash, data["size"], response)

        # ── Send final result ─────────────────────────────────
        await websocket.send_json({
            "type":    "complete",
            "stage":   6,
            "total":   6,
            "message": "Scan complete",
            "data":    response.model_dump(mode="json"),
        })

        logger.info(f"[{scan_id}] Scan complete — verdict={risk_result.verdict} score={risk_result.score}")

    except WebSocketDisconnect:
        logger.info(f"[{scan_id}] Client disconnected mid-scan")
    except Exception as e:
        logger.error(f"[{scan_id}] Unexpected scan error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": f"Scan failed: {str(e)[:100]}"})
        except Exception:
            pass
    finally:
        await websocket.close()