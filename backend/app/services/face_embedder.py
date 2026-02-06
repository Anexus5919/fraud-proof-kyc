"""
Face Detection & Embedding using ONNX Runtime directly.

Detection: SCRFD model (det_10g.onnx) — state-of-art face detection with 5-point landmarks
Recognition: ArcFace model (w600k_r50.onnx) — 512-dim face embedding for deduplication

Bypasses the InsightFace Python library entirely — loads ONNX models directly for
maximum compatibility and reliability. No Cython compilation or specific library version needed.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Model paths — buffalo_l pack from InsightFace
# InsightFace's FaceAnalysis downloads to buffalo_l/ subdirectory;
# check that first, then flat path for manual installations
_BASE_MODEL_DIR = Path.home() / '.insightface' / 'models'
_BUFFALO_DIR = _BASE_MODEL_DIR / 'buffalo_l'
MODEL_DIR = _BUFFALO_DIR if (_BUFFALO_DIR / 'det_10g.onnx').exists() else _BASE_MODEL_DIR
DET_MODEL_PATH = MODEL_DIR / 'det_10g.onnx'
REC_MODEL_PATH = MODEL_DIR / 'w600k_r50.onnx'

# ArcFace standard 5-point reference landmarks for 112x112 alignment
ARCFACE_REF_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)

# Global singleton
_face_app = None
_use_fallback = False


class NoFaceError(Exception):
    """Raised when no face is detected in the image"""
    pass


class MultipleFacesError(Exception):
    """Raised when multiple faces are detected"""
    pass


class FaceResult:
    """Face detection result with embedding, compatible with InsightFace API."""

    def __init__(self, bbox, score, landmarks_5, embedding=None):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = float(score)
        self.landmarks_5 = np.array(landmarks_5, dtype=np.float32) if landmarks_5 is not None else None
        self.landmark_2d_106 = None
        self.embedding = embedding


class SCRFDDetector:
    """
    SCRFD face detector using ONNX Runtime.

    Model: det_10g.onnx (SCRFD-10GF, 16.9 MB)
    Architecture: Anchor-free detection at 3 scales (strides 8, 16, 32)
    Outputs per stride: scores (N,1), bboxes (N,4), keypoints (N,10)
    """

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        import onnxruntime as ort

        self.input_size = input_size  # (width, height)
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.score_threshold = 0.5
        self.nms_threshold = 0.4

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

        # Pre-compute anchor centers for each stride
        self._anchor_centers = {}
        for stride in self.strides:
            fw = input_size[0] // stride  # feature map width
            fh = input_size[1] // stride  # feature map height
            grid_y, grid_x = np.mgrid[:fh, :fw]
            centers = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32) * stride
            # Repeat for num_anchors per location
            centers = np.repeat(centers, self.num_anchors, axis=0)
            self._anchor_centers[stride] = centers

        logger.info(f"SCRFD detector loaded: input={input_size}, strides={self.strides}")

    def detect(self, image: np.ndarray) -> List[FaceResult]:
        """
        Detect faces in a BGR image.

        Returns:
            List of FaceResult with bbox, score, and 5-point landmarks.
        """
        h, w = image.shape[:2]
        input_w, input_h = self.input_size

        # Resize preserving aspect ratio
        scale = min(input_w / w, input_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Pad to input size (top-left aligned, zero padding)
        padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Preprocess: BGR→RGB, normalize (subtract 127.5, divide by 128), NCHW
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        # Run inference
        outputs = self.session.run(None, {self.input_name: blob})

        # Parse outputs: 9 tensors = 3 strides x (scores, bboxes, keypoints)
        # Output order: scores[0,1,2], bboxes[3,4,5], keypoints[6,7,8]
        all_bboxes = []
        all_scores = []
        all_kps = []

        for i, stride in enumerate(self.strides):
            scores = outputs[i]           # (N, 1) — raw scores
            bboxes = outputs[i + 3]       # (N, 4) — distance offsets from anchor
            keypoints = outputs[i + 6]    # (N, 10) — 5 landmarks x 2 coords

            anchors = self._anchor_centers[stride]

            # Decode bounding boxes: anchor_center ± offset * stride
            decoded = np.empty_like(bboxes)
            decoded[:, 0] = anchors[:, 0] - bboxes[:, 0] * stride  # x1
            decoded[:, 1] = anchors[:, 1] - bboxes[:, 1] * stride  # y1
            decoded[:, 2] = anchors[:, 0] + bboxes[:, 2] * stride  # x2
            decoded[:, 3] = anchors[:, 1] + bboxes[:, 3] * stride  # y2

            # Decode keypoints: anchor_center + offset * stride
            decoded_kps = np.empty_like(keypoints)
            for k in range(5):
                decoded_kps[:, k * 2] = anchors[:, 0] + keypoints[:, k * 2] * stride
                decoded_kps[:, k * 2 + 1] = anchors[:, 1] + keypoints[:, k * 2 + 1] * stride

            # Filter by confidence threshold
            mask = scores[:, 0] > self.score_threshold
            all_bboxes.append(decoded[mask])
            all_scores.append(scores[mask, 0])
            all_kps.append(decoded_kps[mask])

        # Concatenate all detections across strides
        if not any(len(b) > 0 for b in all_bboxes):
            return []

        bboxes = np.concatenate(all_bboxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        kps = np.concatenate(all_kps, axis=0)

        if len(bboxes) == 0:
            return []

        # Non-maximum suppression
        keep = self._nms(bboxes, scores, self.nms_threshold)

        # Build results, scaling coordinates back to original image space
        results = []
        for idx in keep:
            bbox = bboxes[idx] / scale
            score = scores[idx]
            landmarks = kps[idx].reshape(5, 2) / scale

            # Clamp to image boundaries
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(w, bbox[2])
            bbox[3] = min(h, bbox[3])

            results.append(FaceResult(bbox, score, landmarks))

        # Sort by detection score (highest first)
        results.sort(key=lambda f: f.det_score, reverse=True)
        return results

    @staticmethod
    def _nms(bboxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Standard non-maximum suppression."""
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(int(i))

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

            mask = iou <= threshold
            order = order[1:][mask]

        return keep


class ArcFaceRecognizer:
    """
    ArcFace face recognizer using ONNX Runtime.

    Model: w600k_r50.onnx (ResNet50 trained on WebFace600K, 166 MB)
    Input: 112x112 aligned face (RGB, normalized to [-1, 1])
    Output: 512-dim L2-normalized embedding
    """

    def __init__(self, model_path: str):
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("ArcFace recognizer loaded: input=112x112, output=512-dim embedding")

    def get_embedding(self, image: np.ndarray, landmarks_5: np.ndarray) -> np.ndarray:
        """
        Get 512-dim face embedding using landmark-based alignment.

        Args:
            image: BGR image (full frame)
            landmarks_5: 5-point facial landmarks from SCRFD [(5, 2) array]

        Returns:
            512-dim L2-normalized embedding vector
        """
        # Align face using similarity transform
        aligned = self._align_face(image, landmarks_5)

        # Preprocess: BGR→RGB, normalize to [-1, 1], NCHW
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = (rgb - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 112, 112)

        # Run ArcFace inference
        outputs = self.session.run(None, {self.input_name: blob})
        embedding = outputs[0].flatten()  # (512,)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def _align_face(image: np.ndarray, landmarks_5: np.ndarray) -> np.ndarray:
        """
        Align face to 112x112 using similarity transform from 5-point landmarks
        to ArcFace reference landmarks.
        """
        # Estimate similarity transform (rotation, scale, translation)
        M, _ = cv2.estimateAffinePartial2D(
            landmarks_5.astype(np.float32),
            ARCFACE_REF_LANDMARKS,
            method=cv2.LMEDS
        )

        if M is None:
            # Fallback: use full affine if similarity fails
            M = cv2.getAffineTransform(
                landmarks_5[:3].astype(np.float32),
                ARCFACE_REF_LANDMARKS[:3]
            )

        aligned = cv2.warpAffine(image, M, (112, 112), borderValue=(0, 0, 0))
        return aligned


class OnnxFaceAnalyzer:
    """
    Complete face analysis pipeline using ONNX Runtime directly.
    Combines SCRFD detection with ArcFace embedding.
    """

    def __init__(self):
        self.detector = None
        self.recognizer = None
        self._load()

    def _load(self):
        logger.info(f"[ONNX] Model directory resolved to: {MODEL_DIR}")
        logger.info(f"[ONNX] Detection model: {DET_MODEL_PATH} (exists={DET_MODEL_PATH.exists()})")
        logger.info(f"[ONNX] Recognition model: {REC_MODEL_PATH} (exists={REC_MODEL_PATH.exists()})")

        if not DET_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"SCRFD detection model not found at {DET_MODEL_PATH}. "
                f"Checked: {_BUFFALO_DIR}, {_BASE_MODEL_DIR}"
            )
        if not REC_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ArcFace recognition model not found at {REC_MODEL_PATH}. "
                f"Checked: {_BUFFALO_DIR}, {_BASE_MODEL_DIR}"
            )

        self.detector = SCRFDDetector(str(DET_MODEL_PATH))
        self.recognizer = ArcFaceRecognizer(str(REC_MODEL_PATH))
        logger.info("OnnxFaceAnalyzer ready: SCRFD + ArcFace loaded")

    def get(self, image: np.ndarray) -> List[FaceResult]:
        """
        Detect faces and compute embeddings.

        Args:
            image: BGR image as numpy array

        Returns:
            List of FaceResult with bbox, score, landmarks, and embedding
        """
        faces = self.detector.detect(image)

        for face in faces:
            if face.landmarks_5 is not None:
                face.embedding = self.recognizer.get_embedding(image, face.landmarks_5)

        return faces


# ── OpenCV Fallback (only used if ONNX models are missing) ────────────

class FallbackFaceApp:
    """Fallback face detection using OpenCV when ONNX models are unavailable."""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        logger.warning("Using OpenCV Haar cascade fallback — face embeddings will NOT be accurate!")

    def get(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        results = []
        for (x, y, w, h) in faces:
            # Create face with bbox but NO embedding (force review instead of false match)
            face = FaceResult(
                bbox=[x, y, x + w, y + h],
                score=0.99,
                landmarks_5=None,
                embedding=np.zeros(512, dtype=np.float32)  # Zero embedding = won't match anything
            )
            results.append(face)

        return results


# ── Public API (same interface as before) ─────────────────────────────

def get_face_app():
    """Get or initialize the face analysis app (singleton)."""
    global _face_app, _use_fallback

    if _face_app is None:
        try:
            _face_app = OnnxFaceAnalyzer()
            _use_fallback = False
            logger.info("Face analysis: using ONNX models (SCRFD + ArcFace)")
        except Exception as e:
            logger.warning(f"ONNX face analyzer not available: {e}")
            logger.warning("Falling back to OpenCV — deduplication will NOT work accurately!")
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
    logger.info(f"[EMBED] Extracting embedding from image {image.shape}")
    app = get_face_app()
    faces = app.get(image)
    logger.info(f"[EMBED] Detected {len(faces)} face(s)")

    if len(faces) == 0:
        logger.warning("[EMBED] No face — raising NoFaceError")
        raise NoFaceError("No face detected in the image")

    if len(faces) > 1:
        logger.warning(f"[EMBED] {len(faces)} faces — raising MultipleFacesError")
        raise MultipleFacesError(f"Multiple faces detected ({len(faces)})")

    face = faces[0]
    emb_norm = float(np.linalg.norm(face.embedding)) if face.embedding is not None else 0
    logger.info(f"[EMBED] Face: bbox={face.bbox.tolist()}, det_score={face.det_score:.4f}, "
               f"embedding_norm={emb_norm:.4f}")

    face_info = {
        'bbox': face.bbox.tolist(),
        'det_score': float(face.det_score),
        'landmarks': face.landmarks_5.tolist() if face.landmarks_5 is not None else None,
    }

    return face.embedding, face_info


def get_face_bbox(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Get face bounding box from image.

    Returns:
        Tuple of (x1, y1, x2, y2) or None if no face detected

    Raises:
        MultipleFacesError: If multiple faces are detected
    """
    logger.info(f"[BBOX] Detecting face in image {image.shape}")
    app = get_face_app()
    faces = app.get(image)
    logger.info(f"[BBOX] Found {len(faces)} face(s)")

    if len(faces) == 0:
        logger.warning("[BBOX] No face detected")
        return None

    if len(faces) > 1:
        logger.warning(f"[BBOX] {len(faces)} faces detected")
        raise MultipleFacesError(f"Multiple faces detected ({len(faces)})")

    bbox = faces[0].bbox
    result = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    logger.info(f"[BBOX] Face bbox: {result}, score={faces[0].det_score:.4f}")
    return result


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode and preprocess image bytes to numpy array.

    Args:
        image_bytes: Raw image bytes

    Returns:
        BGR image as numpy array
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image")

    return image
