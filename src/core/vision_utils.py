import cv2
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders coordinates in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies a perspective transform to obtain a top-down view of the image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def preprocess_image(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Full CV pipeline: Load, find document contour, deskew, and binarize.
    Optimized for modern VLM consumption.
    """
    try:
        # 1. Load image from bytes
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image format.")

        original = image.copy()
        ratio = image.shape[0] / 500.0
        orig_height, orig_width = image.shape[:2]

        # Resize for faster edge detection processing
        image = cv2.resize(image, (int(orig_width / ratio), 500))

        # 2. Grayscale & CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # CLAHE is far superior to simple grayscaling for mitigating shadows from smartphone photos.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Edge detection
        edged = cv2.Canny(gray, 75, 200)

        # 4. Find contours
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        
        screenCnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        # 5. Perspective Transform
        if screenCnt is not None:
            warped = four_point_transform(original, screenCnt.reshape(4, 2) * ratio)
        else:
            logger.warning("Could not find 4-point boundaries. Proceeding with original image.")
            warped = original

        # 6. Adaptive Thresholding / Binarization for VLM clarity
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Adaptive thresholding handles varying lighting conditions across the document better
        # than a global binary threshold.
        binary = cv2.adaptiveThreshold(
            warped_gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 10
        )

        return binary

    except Exception as e:
        logger.error(f"Error in CV pipeline: {e}")
        return None
