"""
Layer 3: Deepfake Detection

Uses a pre-trained Vision Transformer (ViT) model for real deepfake detection.
Model: onnx-community/Deep-Fake-Detector-v2-Model-ONNX
- Architecture: ViT-base-patch16-224, fine-tuned on real/fake face dataset
- Accuracy: 92.12% on 56,001 test images (F1: 0.917 Real, 0.925 Deepfake)
- ONNX Runtime inference (CPU) — no PyTorch/TensorFlow needed

Supplementary: Frequency domain analysis (research-backed GAN detection).
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Model configuration
MODEL_REPO = "onnx-community/Deep-Fake-Detector-v2-Model-ONNX"
MODEL_FILE = "onnx/model_quantized.onnx"
MODEL_CACHE_DIR = str(Path(__file__).parent.parent.parent / "ml_models" / "deepfake")

# ViT preprocessing constants (ImageNet normalization)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224

_detector = None


class DeepfakeDetector:
    """ML-based deepfake detection using Vision Transformer (ViT) ONNX model."""

    def __init__(self):
        self._session = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        """Download and load the ONNX deepfake detection model."""
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            logger.info("DeepfakeDetector: Downloading model (first run may take a minute)...")

            model_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                cache_dir=MODEL_CACHE_DIR,
            )

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4

            self._session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._model_loaded = True

            input_info = self._session.get_inputs()[0]
            output_info = self._session.get_outputs()[0]
            logger.info(
                f"DeepfakeDetector: Model loaded — "
                f"input={input_info.name} {input_info.shape}, "
                f"output={output_info.name} {output_info.shape}"
            )

        except ImportError as e:
            logger.warning(f"DeepfakeDetector: Missing dependency ({e}). Heuristic fallback active.")
        except Exception as e:
            logger.warning(f"DeepfakeDetector: Model load failed ({e}). Heuristic fallback active.")

    def _preprocess(self, image: np.ndarray, face_bbox: Optional[Tuple] = None) -> np.ndarray:
        """
        Preprocess image for ViT model inference.

        Steps:
        1. Crop face region (with padding) if bbox provided
        2. Resize to 224x224
        3. Convert BGR -> RGB
        4. Normalize to [0,1] then apply ImageNet stats
        5. Transpose to NCHW format (1, 3, 224, 224)
        """
        h, w = image.shape[:2]

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            face_w = x2 - x1
            face_h = y2 - y1
            pad_x = int(face_w * 0.3)
            pad_y = int(face_h * 0.3)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            cropped = image[y1:y2, x1:x2]
            # Guard against empty crop from malformed bbox
            if cropped.size == 0:
                logger.warning(f"Empty face crop from bbox ({x1},{y1},{x2},{y2}), using full image")
            else:
                image = cropped

        resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)

        return batched

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _ml_inference(self, image: np.ndarray, face_bbox: Optional[Tuple] = None) -> Tuple[Optional[float], Dict]:
        """Run ViT model inference and return (real_probability, details)."""
        if not self._model_loaded or self._session is None:
            return None, {"error": "Model not loaded"}

        try:
            input_tensor = self._preprocess(image, face_bbox)
            input_name = self._session.get_inputs()[0].name

            outputs = self._session.run(None, {input_name: input_tensor})
            logits = outputs[0]  # shape: (1, 2) — [Realism, Deepfake]

            # Guard against NaN/Inf from corrupted model output
            if not np.all(np.isfinite(logits)):
                logger.warning("DeepfakeDetector: Model produced NaN/Inf logits, falling back")
                return None, {"error": "Model produced invalid output"}

            probabilities = self._softmax(logits)[0]
            real_prob = float(probabilities[0])
            deepfake_prob = float(probabilities[1])

            details = {
                "model": "ViT-base-patch16-224 (ONNX quantized)",
                "real_probability": round(real_prob, 4),
                "deepfake_probability": round(deepfake_prob, 4),
                "prediction": "real" if real_prob > deepfake_prob else "deepfake",
                "confidence": round(max(real_prob, deepfake_prob), 4),
            }

            return real_prob, details
        except Exception as e:
            logger.warning(f"DeepfakeDetector ML inference failed: {e}")
            return None, {"error": str(e)}

    def _frequency_analysis(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Supplementary: Frequency domain analysis for GAN artifacts.
        Based on "Unmasking DeepFakes with simple Features" (Durall et al., 2019).
        """
        # Handle both BGR and grayscale images
        if len(image.shape) == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_r = min(h, w) // 2
        radial_profile = np.zeros(max_r)

        y_coords, x_coords = np.mgrid[:h, :w]
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(int)

        for r in range(max_r):
            mask = distances == r
            if mask.any():
                radial_profile[r] = magnitude[mask].mean()

        if radial_profile.max() > 0:
            radial_profile /= radial_profile.max()

        mid_start = max_r // 4
        high_start = 3 * max_r // 4

        mid_energy = float(radial_profile[mid_start:high_start].mean()) if high_start > mid_start else 0
        high_energy = float(radial_profile[high_start:].mean()) if high_start < max_r else 0

        if len(radial_profile) > 2:
            profile_diff = np.diff(radial_profile[mid_start:])
            smoothness = float(np.abs(profile_diff).std())
        else:
            smoothness = 0

        details = {
            "mid_frequency_energy": round(mid_energy, 4),
            "high_frequency_energy": round(high_energy, 4),
            "profile_smoothness": round(smoothness, 4),
        }

        score = 0.7
        if smoothness > 0.08:
            score -= 0.3
        if high_energy > 0.3:
            score -= 0.2
        if mid_energy < 0.1:
            score -= 0.15

        score = max(0.0, min(1.0, score))
        details["score"] = round(score, 4)
        return score, details

    def detect(self, image: np.ndarray, face_bbox: Optional[Tuple] = None) -> Tuple[float, Dict]:
        """
        Analyze image for deepfake indicators.

        Primary: ViT ML model (80% weight)
        Supplementary: Frequency domain analysis (20% weight)
        Fallback: Frequency-only if model unavailable.

        Args:
            image: BGR image as numpy array
            face_bbox: Optional (x1, y1, x2, y2) face bounding box

        Returns:
            Tuple of (authenticity_score, details)
            authenticity_score: 0-1, higher = more likely real
        """
        logger.info(f"[DEEPFAKE] Starting detection — image={image.shape}, bbox={face_bbox}, model_loaded={self._model_loaded}")
        details = {}

        h, w = image.shape[:2]
        if h < 50 or w < 50:
            logger.warning(f"[DEEPFAKE] Image too small ({h}x{w}) — returning 0.5")
            return 0.5, {"error": "Image too small for analysis"}

        # Primary: ML model inference
        ml_score, ml_details = self._ml_inference(image, face_bbox)
        details["ml_model"] = ml_details
        logger.info(f"[DEEPFAKE] ML score: {ml_score if ml_score is not None else 'UNAVAILABLE'}")
        if ml_score is not None:
            logger.info(f"[DEEPFAKE]   real_prob={ml_details.get('real_probability')}, "
                       f"deepfake_prob={ml_details.get('deepfake_probability')}")

        # Supplementary: Frequency analysis
        freq_score, freq_details = self._frequency_analysis(image)
        details["frequency_analysis"] = freq_details
        logger.info(f"[DEEPFAKE] Frequency score: {freq_score:.4f} "
                    f"(mid={freq_details.get('mid_frequency_energy')}, "
                    f"high={freq_details.get('high_frequency_energy')}, "
                    f"smooth={freq_details.get('profile_smoothness')})")

        # Combine scores
        if ml_score is not None:
            final_score = 0.80 * ml_score + 0.20 * freq_score
            details["method"] = "ml_model + frequency_analysis"
            details["weights"] = {"ml_model": 0.80, "frequency_analysis": 0.20}
            logger.info(f"[DEEPFAKE] Combined: 0.80*{ml_score:.4f} + 0.20*{freq_score:.4f} = {final_score:.4f}")
        else:
            final_score = freq_score
            details["method"] = "frequency_analysis_only (model unavailable)"
            details["weights"] = {"frequency_analysis": 1.0}
            logger.warning("[DEEPFAKE] Using frequency-only fallback (model unavailable)")

        # Critical flags
        critical_flag = None
        if ml_score is not None and ml_score < 0.2:
            critical_flag = "high_confidence_deepfake"
            final_score = min(final_score, 0.25)
            logger.warning(f"[DEEPFAKE] CRITICAL: high_confidence_deepfake (ml={ml_score:.4f} < 0.2)")
        elif freq_score < 0.25:
            critical_flag = "gan_frequency_signature"
            if ml_score is None:
                final_score = min(final_score, 0.35)
            logger.warning(f"[DEEPFAKE] CRITICAL: gan_frequency_signature (freq={freq_score:.4f} < 0.25)")

        if critical_flag:
            details["critical_flag"] = critical_flag

        final_score = max(0.0, min(1.0, final_score))
        details["score"] = round(float(final_score), 4)

        logger.info(f"[DEEPFAKE] FINAL SCORE: {final_score:.4f}")
        return float(final_score), details


def get_deepfake_detector() -> DeepfakeDetector:
    """Get or create the global deepfake detector instance."""
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector()
        logger.info("DeepfakeDetector initialized")
    return _detector


def check_deepfake(image: np.ndarray, face_bbox=None) -> Tuple[float, Dict]:
    """
    Check if an image is a deepfake.

    Args:
        image: BGR image as numpy array
        face_bbox: Optional (x1, y1, x2, y2) face bounding box

    Returns:
        Tuple of (authenticity_score, details)
        authenticity_score: 0-1, higher = more likely real
    """
    detector = get_deepfake_detector()
    return detector.detect(image, face_bbox)
