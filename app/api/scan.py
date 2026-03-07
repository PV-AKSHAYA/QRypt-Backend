from fastapi import APIRouter, UploadFile, File
from app.services.qr_extractor import QRExtractor
from app.services.url_validator import URLValidator

router = APIRouter()


@router.post("/scan")
async def scan_qr(file: UploadFile = File(...)):

    image_bytes = await file.read()

    # Step 1: Extract QR
    qr_url = QRExtractor.extract_qr_data(image_bytes)

    if not qr_url:
        return {
            "status": "error",
            "message": "No QR code detected"
        }

    # Step 2: Validate URL
    url_analysis = URLValidator.analyze(qr_url)

    if not url_analysis["valid"]:
        return {
            "status": "error",
            "message": url_analysis["reason"]
        }

    return {
        "status": "success",
        "qr_url": qr_url,
        "domain": url_analysis["domain_info"]
    }