# import cv2
# import numpy as np
# from pyzbar.pyzbar import decode
# from PIL import Image


# class QRExtractor:

#     @staticmethod
#     def extract_qr_data(image_bytes: bytes):
#         """
#         Extract QR code data from uploaded image
#         """

#         # Convert bytes → numpy array
#         np_arr = np.frombuffer(image_bytes, np.uint8)

#         # Decode image
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Detect QR codes
#         qr_codes = decode(gray)

#         if not qr_codes:
#             return None

#         # Take first QR
#         qr_data = qr_codes[0].data.decode("utf-8")

#         return qr_data



"""
SafeQR — qr_extractor.py
QR Extraction Engine.

Input : Raw image bytes
Output: QRResult (found, count, raw_content, bounding_box, all_qr_contents, error)

Strategy (3-pass for maximum detection):
  Pass 1 — pyzbar on original image
  Pass 2 — pyzbar on enhanced/preprocessed image (if Pass 1 found nothing)
  Pass 3 — OpenCV QRCodeDetector (if Pass 2 found nothing)

Handles:
  - No QR found
  - Multiple QR codes
  - Invalid / corrupt image
  - Non-URL QR content (still returned, flagged)
"""

import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

from app.utils.image_utils import (
    load_image_from_bytes,
    resize_if_needed,
    enhance_for_qr,
    ImageLoadError,
)
from app.models.response_models import QRResult, BoundingBox

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Internal data class
# ─────────────────────────────────────────────

class _DecodedQR:
    """Intermediate holder before we build the Pydantic model."""
    def __init__(self, data: str, bbox: Optional[BoundingBox]):
        self.data = data
        self.bbox = bbox


# ─────────────────────────────────────────────
# pyzbar helpers
# ─────────────────────────────────────────────

def _pyzbar_decode(img: np.ndarray) -> List[_DecodedQR]:
    """
    Run pyzbar on a BGR or grayscale ndarray.
    Returns list of decoded QR objects.
    """
    results = pyzbar.decode(img, symbols=[ZBarSymbol.QRCODE])
    decoded = []
    for obj in results:
        try:
            data = obj.data.decode("utf-8", errors="replace").strip()
        except Exception:
            data = str(obj.data)

        rect = obj.rect
        bbox = BoundingBox(
            x=rect.left,
            y=rect.top,
            width=rect.width,
            height=rect.height,
        )
        decoded.append(_DecodedQR(data=data, bbox=bbox))
        logger.debug(f"pyzbar found QR: {data[:80]}")

    return decoded


# ─────────────────────────────────────────────
# OpenCV fallback
# ─────────────────────────────────────────────

def _opencv_decode(img: np.ndarray) -> List[_DecodedQR]:
    """
    OpenCV QRCodeDetector fallback.
    Less reliable than pyzbar but catches some edge cases.
    """
    detector = cv2.QRCodeDetector()
    decoded_list = []

    # detectAndDecodeMulti available in OpenCV 4.x+
    try:
        retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(img)
        if retval and decoded_info:
            for i, data in enumerate(decoded_info):
                if not data:
                    continue
                data = data.strip()
                bbox = None
                if points is not None and i < len(points):
                    pts = points[i].astype(int)
                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]
                    x = int(x_coords.min())
                    y = int(y_coords.min())
                    w = int(x_coords.max()) - x
                    h = int(y_coords.max()) - y
                    bbox = BoundingBox(x=x, y=y, width=w, height=h)
                decoded_list.append(_DecodedQR(data=data, bbox=bbox))
                logger.debug(f"OpenCV fallback found QR: {data[:80]}")
    except Exception as e:
        logger.warning(f"OpenCV QR decode error: {e}")

    return decoded_list


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def extract_qr(image_bytes: bytes) -> QRResult:
    """
    Main entry point.
    Accepts raw image bytes, returns fully-typed QRResult.

    Raises nothing — all errors are captured into QRResult.decode_error.
    """
    # ── Load image ────────────────────────────
    try:
        img = load_image_from_bytes(image_bytes)
    except ImageLoadError as e:
        logger.error(f"Image load failed: {e}")
        return QRResult(
            found=False,
            count=0,
            raw_content="",
            bounding_box=None,
            all_qr_contents=[],
            decode_error=str(e),
        )

    img = resize_if_needed(img)

    # ── Pass 1: pyzbar on original ────────────
    found_qrs: List[_DecodedQR] = _pyzbar_decode(img)

    # ── Pass 2: pyzbar on enhanced image ─────
    if not found_qrs:
        logger.debug("Pass 1 found nothing — trying enhanced image")
        enhanced = enhance_for_qr(img)
        found_qrs = _pyzbar_decode(enhanced)

    # ── Pass 3: OpenCV fallback ───────────────
    if not found_qrs:
        logger.debug("Pass 2 found nothing — trying OpenCV detector")
        found_qrs = _opencv_decode(img)

    # ── No QR found ───────────────────────────
    if not found_qrs:
        logger.info("No QR code detected in image after 3 passes")
        return QRResult(
            found=False,
            count=0,
            raw_content="",
            bounding_box=None,
            all_qr_contents=[],
            decode_error="No QR code found in image",
        )

    # ── Deduplicate (same content can appear from multiple passes) ──
    seen = set()
    unique_qrs: List[_DecodedQR] = []
    for qr in found_qrs:
        if qr.data not in seen:
            seen.add(qr.data)
            unique_qrs.append(qr)

    primary = unique_qrs[0]

    logger.info(f"QR extraction success: {len(unique_qrs)} unique code(s) found")

    return QRResult(
        found=True,
        count=len(unique_qrs),
        raw_content=primary.data,
        bounding_box=primary.bbox,
        all_qr_contents=[q.data for q in unique_qrs],
        decode_error=None,
    )


# ─────────────────────────────────────────────
# Quick self-test (run file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python qr_extractor.py <image_path>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        raw = f.read()

    result = extract_qr(raw)
    print(json.dumps(result.model_dump(), indent=2))
