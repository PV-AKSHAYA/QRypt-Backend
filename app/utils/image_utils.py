"""
app/utils/image_utils.py
─────────────────────────
All image loading, preprocessing, and conversion helpers.
Every service that touches an image imports from here.
Never do raw cv2/PIL operations outside this file.
"""

import cv2
import numpy as np
from PIL import Image
import io
import hashlib
import logging

logger = logging.getLogger("safeqr.image_utils")


# ══════════════════════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════════════════════

def bytes_to_cv2(img_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes → OpenCV BGR numpy array.
    Raises ValueError if bytes are not a valid image.
    """
    if not img_bytes:
        raise ValueError("Empty image received")
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file")
    return img


def bytes_to_pil(img_bytes: bytes) -> Image.Image:
    """
    Convert raw image bytes → PIL Image (RGB).
    Used by pyzbar and AI context engine.
    """
    try:
        pil = Image.open(io.BytesIO(img_bytes))
        pil = pil.convert("RGB")   # normalise — handles RGBA, palette, CMYK
        return pil
    except Exception as e:
        raise ValueError(f"PIL could not open image: {e}")


def cv2_to_bytes(img: np.ndarray, ext: str = ".jpg") -> bytes:
    """
    Convert OpenCV array → JPEG/PNG bytes.
    Useful when you need to re-encode after processing.
    """
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"cv2.imencode failed for format {ext}")
    return buffer.tobytes()


def pil_to_bytes(pil: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
#  PREPROCESS — improve QR detection on bad photos
# ══════════════════════════════════════════════════════════════

def preprocess_for_qr(img: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of progressively enhanced versions of the image.
    QR extractor tries each one until it gets a decode.

    Order matters — cheapest transformations first.
    """
    variants = []

    # 1. Original
    variants.append(img)

    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    # 3. Sharpened
    kernel  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp   = cv2.filter2D(gray, -1, kernel)
    variants.append(sharp)

    # 4. Adaptive threshold (handles uneven lighting)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    variants.append(thresh)

    # 5. Upscaled (helps if QR is small in frame)
    h, w = img.shape[:2]
    if max(h, w) < 800:
        upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        variants.append(upscaled)

    # 6. Denoised + contrast boost
    denoised  = cv2.fastNlMeansDenoising(gray, h=10)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast  = clahe.apply(denoised)
    variants.append(contrast)

    return variants


# ══════════════════════════════════════════════════════════════
#  HASHING — for Threat Memory Engine
# ══════════════════════════════════════════════════════════════

def compute_image_hash(img_bytes: bytes) -> str:
    """
    MD5 hash of raw image bytes.
    Used to detect duplicate scans in MongoDB without re-running all services.
    """
    return hashlib.md5(img_bytes).hexdigest()


# ══════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_BYTES    = 10 * 1024 * 1024   # 10 MB

def validate_image_bytes(img_bytes: bytes, content_type: str = "") -> None:
    """
    Raise ValueError if image is invalid, too large, or wrong type.
    Called at the API boundary before any processing starts.
    """
    if not img_bytes:
        raise ValueError("Empty image received")

    if len(img_bytes) > MAX_IMAGE_BYTES:
        mb = len(img_bytes) / (1024 * 1024)
        raise ValueError(f"Image too large: {mb:.1f}MB (max 10MB)")

    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Unsupported image type: {content_type}. Use JPEG, PNG, or WebP.")

    # Attempt decode as final check
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File is not a valid image")


def get_image_dimensions(img_bytes: bytes) -> tuple[int, int]:
    """Return (width, height) of image."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return (0, 0)
    h, w = img.shape[:2]
    return (w, h)