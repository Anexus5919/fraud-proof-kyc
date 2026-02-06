import numpy as np
from typing import Optional, Tuple
import cv2
import logging
import hashlib

logger = logging.getLogger(__name__)

# Global instance for singleton pattern
_face_app = None
_use_fallback = False

# OpenCV face detector for fallback
_opencv_face_cascade = None


class NoFaceError(Exception):
    """Raised when no face is detected in the image"""
    pass


class MultipleFacesError(Exception):
    """Raised when multiple faces are detected"""
    pass


class FallbackFaceApp:
    """Fallback face detection using OpenCV when InsightFace is unavailable."""

    def __init__(self):
        # Load OpenCV's pre-trained face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logger.info("Using OpenCV fallback for face detection")

    def get(self, image: np.ndarray):
        """Detect faces and return face objects compatible with InsightFace API."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        result = []
        for (x, y, w, h) in faces:
            face_obj = FallbackFace(image, x, y, x+w, y+h)
            result.append(face_obj)

        return result


class FallbackFace:
    """Face object compatible with InsightFace API, using feature-based embedding."""

    def __init__(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        self.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        self.det_score = 0.99  # OpenCV doesn't provide confidence, use fixed value
        self.landmark_2d_106 = None

        # Extract face region and compute embedding
        face_region = image[y1:y2, x1:x2]
        self.embedding = self._compute_embedding(face_region)

    def _compute_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Compute a 512-dimensional embedding using image features.

        This is a fallback that uses histogram and texture features.
        Not as accurate as deep learning, but works for development.
        """
        # Resize to standard size
        face_resized = cv2.resize(face_img, (112, 112))

        # Convert to grayscale
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        embedding = []

        # 1. Histogram features (256 dims)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-6)
        embedding.extend(hist)

        # 2. Color histogram features (3 channels x 32 bins = 96 dims)
        for i in range(3):
            c_hist = cv2.calcHist([face_resized], [i], None, [32], [0, 256])
            c_hist = c_hist.flatten() / (c_hist.sum() + 1e-6)
            embedding.extend(c_hist)

        # 3. HOG-like gradient features (128 dims)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)

        # Divide into 4x4 grid, 8 orientation bins each
        cell_size = 28
        for cy in range(4):
            for cx in range(4):
                cell_mag = magnitude[cy*cell_size:(cy+1)*cell_size,
                                    cx*cell_size:(cx+1)*cell_size]
                cell_dir = direction[cy*cell_size:(cy+1)*cell_size,
                                    cx*cell_size:(cx+1)*cell_size]
                hist_hog, _ = np.histogram(cell_dir.flatten(), bins=8,
                                          range=(-np.pi, np.pi),
                                          weights=cell_mag.flatten())
                hist_hog = hist_hog / (hist_hog.sum() + 1e-6)
                embedding.extend(hist_hog)

        # 4. DCT features (32 dims) - captures facial structure
        dct = cv2.dct(gray.astype(np.float32))
        dct_features = dct[:8, :4].flatten()  # Low-frequency coefficients
        dct_features = dct_features / (np.abs(dct_features).max() + 1e-6)
        embedding.extend(dct_features)

        # Convert to numpy array and normalize to 512 dimensions
        embedding = np.array(embedding, dtype=np.float32)

        # Pad or truncate to exactly 512 dimensions
        if len(embedding) < 512:
            embedding = np.pad(embedding, (0, 512 - len(embedding)))
        else:
            embedding = embedding[:512]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


def get_face_app():
    """Get or initialize the InsightFace app (singleton)"""
    global _face_app, _use_fallback

    if _face_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis

            logger.info("Initializing InsightFace...")
            _face_app = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            _face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace initialized successfully")
            _use_fallback = False
        except Exception as e:
            logger.warning(f"InsightFace not available: {e}")
            logger.info("Using OpenCV fallback for face detection and embedding")
            _face_app = FallbackFaceApp()
            _use_fallback = True

    return _face_app


def get_face_embedding(image: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Extract face embedding from image.

    Args:
        image: BGR image as numpy array

    Returns:
        Tuple of (512-dim embedding vector, face_info dict)

    Raises:
        NoFaceError: If no face is detected
        MultipleFacesError: If multiple faces are detected
    """
    app = get_face_app()

    # Detect faces
    faces = app.get(image)

    if len(faces) == 0:
        raise NoFaceError("No face detected in the image")

    if len(faces) > 1:
        raise MultipleFacesError(f"Multiple faces detected ({len(faces)})")

    face = faces[0]

    # Extract embedding
    embedding = face.embedding

    # Get face info
    face_info = {
        'bbox': face.bbox.tolist(),
        'det_score': float(face.det_score),
        'landmarks': face.landmark_2d_106.tolist() if face.landmark_2d_106 is not None else None,
    }

    return embedding, face_info


def get_face_bbox(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Get face bounding box from image.

    Returns:
        Tuple of (x1, y1, x2, y2) or None if no face detected
    """
    app = get_face_app()
    faces = app.get(image)

    if len(faces) == 0:
        return None

    # Return first face bbox
    bbox = faces[0].bbox
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode and preprocess image bytes to numpy array.

    Args:
        image_bytes: Raw image bytes

    Returns:
        BGR image as numpy array
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image")

    return image
