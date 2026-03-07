"""
SafeQR — image_utils.py
Low-level image handling. Load, validate, hash, preprocess.
All functions are pure — no side effects, no state.
"""

import hashlib
import io
import logging
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Max image dimension we'll process (resize above this)
MAX_DIMENSION = 2048
# Min dimension — anything smaller is likely garbage
MIN_DIMENSION = 64
# Allowed MIME types
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}


class ImageLoadError(Exception):
    """Raised when an image cannot be decoded or is invalid."""
    pass


def load_image_from_bytes(raw: bytes) -> np.ndarray:
    """
    Decode raw bytes → OpenCV BGR ndarray.
    Raises ImageLoadError on failure.
    """
    if not raw:
        raise ImageLoadError("Empty image bytes received")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ImageLoadError("cv2.imdecode returned None — not a valid image format")

    h, w = img.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ImageLoadError(f"Image too small: {w}x{h}px (min {MIN_DIMENSION}px)")

    return img


def sha256_of_bytes(raw: bytes) -> str:
    """Return lowercase hex SHA-256 of raw image bytes."""
    return hashlib.sha256(raw).hexdigest()


def resize_if_needed(img: np.ndarray) -> np.ndarray:
    """
    Downscale image if either dimension exceeds MAX_DIMENSION.
    Preserves aspect ratio. Never upscales.
    """
    h, w = img.shape[:2]
    if h <= MAX_DIMENSION and w <= MAX_DIMENSION:
        return img

    scale = MAX_DIMENSION / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug(f"Resized image from {w}x{h} → {new_w}x{new_h}")
    return resized


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """BGR → grayscale."""
    if len(img.shape) == 2:
        return img  # already gray
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def enhance_for_qr(img: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing pipeline to improve QR detection on difficult images:
    1. Convert to grayscale
    2. Adaptive threshold (handles uneven lighting)
    3. Mild sharpening
    Returns grayscale enhanced image.
    """
    gray = to_grayscale(img)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Sharpen via unsharp mask
    blur = cv2.GaussianBlur(thresh, (0, 0), 3)
    sharp = cv2.addWeighted(thresh, 1.5, blur, -0.5, 0)

    return sharp


def get_image_dimensions(img: np.ndarray) -> Tuple[int, int]:
    """Returns (width, height)."""
    h, w = img.shape[:2]
    return w, h


def crop_region(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Safe crop — clamps to image bounds."""
    ih, iw = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)
    return img[y1:y2, x1:x2]


def encode_to_jpeg_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    """Encode ndarray back to JPEG bytes (for passing to AI engine)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageLoadError("Failed to re-encode image to JPEG")
    return buf.tobytes()
