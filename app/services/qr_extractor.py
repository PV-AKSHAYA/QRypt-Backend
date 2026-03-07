import cv2
import numpy as np
from pyzbar.pyzbar import decode
from PIL import Image


class QRExtractor:

    @staticmethod
    def extract_qr_data(image_bytes: bytes):
        """
        Extract QR code data from uploaded image
        """

        # Convert bytes → numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect QR codes
        qr_codes = decode(gray)

        if not qr_codes:
            return None

        # Take first QR
        qr_data = qr_codes[0].data.decode("utf-8")

        return qr_data