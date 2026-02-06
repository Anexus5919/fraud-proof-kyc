"""
Silent-Face-Anti-Spoofing Detection Implementation

This module implements robust anti-spoofing detection using multiple techniques:
1. Deep learning-based detection (MiniFASNet architecture)
2. Texture analysis (LBP features)
3. Frequency domain analysis (FFT/DCT for moiré pattern detection)
4. Color space analysis (skin tone validation)
5. Specular highlight detection (screen reflection patterns)

For production deployment with actual Silent-Face ONNX models.
"""
import numpy as np
import cv2
import logging
import os
from typing import Tuple, Optional
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Global model instance
_spoof_detector = None

# Detection thresholds (tuned for security)
THRESHOLDS = {
    'moire_freq_threshold': 0.15,      # Moiré pattern detection
    'texture_uniformity': 0.25,        # LBP texture uniformity (screens are more uniform)
    'color_range_min': 0.08,           # Minimum color variation (real faces have more)
    'specular_threshold': 0.35,        # Screen reflection detection
    'gradient_threshold': 15.0,        # Edge sharpness (prints are too sharp)
    'skin_ratio_min': 0.15,            # Minimum skin-tone pixels
    'noise_threshold': 2.0,            # Sensor noise (real cameras have noise)
}


class MiniFASNet:
    """
    Lightweight anti-spoofing network implementation.

    Architecture based on Silent-Face MiniFASNetV2:
    - Compact CNN for real-time inference
    - Multi-scale feature extraction
    - Binary classification: real vs spoof
    """

    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        self.input_size = (80, 80)

        if model_path and os.path.exists(model_path):
            try:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']
                )
                logger.info(f"Loaded ONNX model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX model: {e}")
                self.session = None

    def preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input."""
        # Resize to model input size
        img = cv2.resize(face_img, self.input_size)

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        # Add batch dimension and transpose to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)

        return img

    def predict(self, face_img: np.ndarray) -> Tuple[float, dict]:
        """
        Predict if face is real or spoof.

        Returns:
            Tuple of (real_score, details)
        """
        if self.session is None:
            return None, {'model_loaded': False}

        try:
            input_data = self.preprocess(face_img)
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_data})

            # Softmax output: [real_prob, spoof_prob]
            probs = outputs[0][0]
            real_score = float(probs[0]) if len(probs) > 1 else float(probs[0])

            return real_score, {'model_output': probs.tolist()}
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return None, {'error': str(e)}


class SpoofDetector:
    """
    Multi-technique anti-spoofing detector.

    Combines multiple detection methods for robust spoof detection:
    1. Deep learning (if model available)
    2. Frequency analysis (moiré patterns from screens)
    3. Texture analysis (print vs real skin)
    4. Color analysis (screen color gamut)
    5. Specular analysis (screen reflections)
    """

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self._load_model()

    def _load_model(self):
        """Load the deep learning model if available."""
        model_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..',
                        'ml_models', 'silent_face', 'MiniFASNetV2_80x80.onnx'),
            os.path.join(os.path.dirname(__file__), '..', '..',
                        'ml_models', 'silent_face', '2.7_80x80_MiniFASNetV2.onnx'),
        ]

        for path in model_paths:
            if os.path.exists(path):
                self.model = MiniFASNet(path)
                if self.model.session is not None:
                    self.model_loaded = True
                    logger.info("Deep learning spoof model loaded successfully")
                    return

        logger.info("Deep learning model not found, using multi-technique detection")
        self.model_loaded = False

    def detect(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[float, dict]:
        """
        Detect if the face is real or spoofed.

        Uses multiple detection techniques and combines their results
        for robust anti-spoofing.

        Args:
            image: BGR image as numpy array
            face_bbox: Face bounding box (x1, y1, x2, y2)

        Returns:
            Tuple of (spoof_score, details)
            spoof_score: 0-1, higher = more likely to be real
        """
        logger.info(f"[SPOOF] Starting detection — image={image.shape}, bbox={face_bbox}")
        details = {}
        scores = []
        weights = []

        # Extract face region
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            # Add padding for better analysis
            h, w = image.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            face_region = image[y1:y2, x1:x2]
            logger.info(f"[SPOOF] Face region extracted: {face_region.shape} (padded bbox)")
        else:
            h, w = image.shape[:2]
            face_region = image[h//6:5*h//6, w//6:5*w//6]
            logger.info(f"[SPOOF] No bbox — using center crop: {face_region.shape}")

        if face_region.size == 0:
            logger.warning("[SPOOF] Empty face region — returning 0.0")
            return 0.0, {'error': 'Invalid face region'}

        # Ensure minimum size for analysis
        if face_region.shape[0] < 50 or face_region.shape[1] < 50:
            logger.warning(f"[SPOOF] Face too small: {face_region.shape} — returning 0.3")
            return 0.3, {'error': 'Face region too small for reliable analysis'}

        # 1. Deep learning detection (highest weight if available)
        if self.model_loaded and self.model:
            dl_score, dl_details = self.model.predict(face_region)
            if dl_score is not None:
                details['deep_learning'] = dl_details
                details['dl_score'] = dl_score
                scores.append(dl_score)
                weights.append(0.40)  # 40% weight
                logger.info(f"[SPOOF]  1. DL model      : {dl_score:.4f} (weight=0.40)")
            else:
                logger.info(f"[SPOOF]  1. DL model      : FAILED ({dl_details})")
        else:
            logger.info("[SPOOF]  1. DL model      : NOT LOADED")

        # Each sub-detector is wrapped in try/except so one failure doesn't crash the pipeline
        # 2. Moiré pattern detection (screens produce moiré)
        try:
            moire_score, moire_details = self._detect_moire_pattern(face_region)
        except Exception as e:
            logger.warning(f"Moiré detection failed: {e}")
            moire_score, moire_details = 0.5, {'error': str(e)}
        details['moire'] = moire_details
        scores.append(moire_score)
        moire_w = 0.15 if self.model_loaded else 0.10
        weights.append(moire_w)
        logger.info(f"[SPOOF]  2. Moiré         : {moire_score:.4f} (weight={moire_w})")

        # 3. Texture analysis (LBP-based)
        try:
            texture_score, texture_details = self._analyze_texture(face_region)
        except Exception as e:
            logger.warning(f"Texture analysis failed: {e}")
            texture_score, texture_details = 0.5, {'error': str(e)}
        details['texture'] = texture_details
        scores.append(texture_score)
        texture_w = 0.15 if self.model_loaded else 0.30
        weights.append(texture_w)
        logger.info(f"[SPOOF]  3. Texture       : {texture_score:.4f} (weight={texture_w})")

        # 4. Color space analysis
        try:
            color_score, color_details = self._analyze_color_space(face_region)
        except Exception as e:
            logger.warning(f"Color analysis failed: {e}")
            color_score, color_details = 0.5, {'error': str(e)}
        details['color'] = color_details
        scores.append(color_score)
        color_w = 0.10 if self.model_loaded else 0.20
        weights.append(color_w)
        logger.info(f"[SPOOF]  4. Color         : {color_score:.4f} (weight={color_w})")

        # 5. Specular highlight detection (screen reflections)
        try:
            specular_score, specular_details = self._detect_specular_highlights(face_region)
        except Exception as e:
            logger.warning(f"Specular detection failed: {e}")
            specular_score, specular_details = 0.5, {'error': str(e)}
        details['specular'] = specular_details
        scores.append(specular_score)
        specular_w = 0.10 if self.model_loaded else 0.15
        weights.append(specular_w)
        logger.info(f"[SPOOF]  5. Specular      : {specular_score:.4f} (weight={specular_w})")

        # 6. Noise pattern analysis (real cameras have characteristic noise)
        try:
            noise_score, noise_details = self._analyze_noise_pattern(face_region)
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            noise_score, noise_details = 0.5, {'error': str(e)}
        details['noise'] = noise_details
        scores.append(noise_score)
        noise_w = 0.05 if self.model_loaded else 0.20
        weights.append(noise_w)
        logger.info(f"[SPOOF]  6. Noise         : {noise_score:.4f} (weight={noise_w})")

        # Weighted combination
        total_weight = sum(weights)
        final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        logger.info(f"[SPOOF] Weighted raw score: {final_score:.4f} (total_weight={total_weight:.2f})")

        # Apply strict threshold: require multiple critical signals to clamp score
        # Single sub-detector failure alone is unreliable (e.g. moiré false-positive on webcam JPEG)
        critical_failures = []
        if moire_score < 0.25:
            critical_failures.append('moire_pattern_detected')
        if texture_score < 0.20:
            critical_failures.append('texture_anomaly')
        if specular_score < 0.20:
            critical_failures.append('screen_reflection_detected')

        if len(critical_failures) >= 2:
            # Multiple signals agree — high confidence spoof
            pre_clamp = final_score
            final_score = min(final_score, 0.35)
            details['critical_failures'] = critical_failures
            logger.warning(f"[SPOOF] Multiple critical failures {critical_failures} — clamped {pre_clamp:.4f} → {final_score:.4f}")
        elif critical_failures:
            # Single signal — note it but don't override the weighted score
            details['critical_warnings'] = critical_failures
            logger.info(f"[SPOOF] Single warning (not clamping): {critical_failures}")

        details['final_score'] = float(final_score)
        details['individual_scores'] = {
            'moire': moire_score,
            'texture': texture_score,
            'color': color_score,
            'specular': specular_score,
            'noise': noise_score
        }

        logger.info(f"[SPOOF] FINAL SCORE: {final_score:.4f}")
        return float(final_score), details

    def _detect_moire_pattern(self, face_img: np.ndarray) -> Tuple[float, dict]:
        """
        Detect moiré patterns that appear when photographing screens.

        Screens have characteristic periodic patterns in frequency domain.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Apply FFT
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Log transform for better visualization
        magnitude_log = np.log1p(magnitude)

        h, w = magnitude_log.shape
        center_y, center_x = h // 2, w // 2

        # Create masks for different frequency bands
        # Low frequency (center) — small circle around DC component
        low_mask_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(low_mask_u8, (center_x, center_y), min(h, w) // 8, 1, -1)
        low_mask = low_mask_u8.astype(bool)

        # High frequency (edges) — outside a larger circle
        high_circle_u8 = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(high_circle_u8, (center_x, center_y), min(h, w) // 3, 1, -1)
        high_mask = ~high_circle_u8.astype(bool)

        # Mid frequency (moiré patterns typically appear here)
        mid_mask = ~low_mask & ~high_mask

        low_energy = magnitude_log[low_mask].mean() if low_mask.any() else 0
        mid_energy = magnitude_log[mid_mask].mean() if mid_mask.any() else 0
        high_energy = magnitude_log[high_mask].mean() if high_mask.any() else 0

        # Moiré patterns cause spikes in mid-high frequency
        # Real faces have smoother frequency distribution
        total_energy = low_energy + mid_energy + high_energy + 1e-6
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy

        # Detect periodic peaks (characteristic of moiré)
        # Look for regular patterns in the frequency domain
        mid_band = magnitude_log.copy()
        mid_band[low_mask] = 0
        mid_band[high_mask] = 0

        # Find peaks
        threshold = mid_band.mean() + 2 * mid_band.std()
        peaks = mid_band > threshold
        peak_count = peaks.sum()
        peak_ratio = peak_count / mid_mask.sum() if mid_mask.sum() > 0 else 0

        details = {
            'mid_frequency_ratio': float(mid_ratio),
            'high_frequency_ratio': float(high_ratio),
            'peak_ratio': float(peak_ratio),
        }

        logger.info(f"[MOIRE] peak_ratio={peak_ratio:.6f}, mid_ratio={mid_ratio:.4f}, high_ratio={high_ratio:.4f}")

        # Scoring: High mid-frequency energy AND many peaks = likely screen
        # Webcam JPEG compression naturally produces frequency domain artifacts:
        #   - mid_ratio typically 0.35-0.45 for normal webcam captures
        #   - peak_ratio typically 0.005-0.02 from JPEG block boundaries
        # Screen captures show MUCH higher values:
        #   - mid_ratio > 0.50 with peak_ratio > 0.03
        # We use very conservative thresholds to avoid false positives on real webcams.
        if peak_ratio > 0.03 and mid_ratio > 0.50:
            score = 0.15  # Very strong moiré: clearly a screen capture
        elif peak_ratio > 0.02 and mid_ratio > 0.47:
            score = 0.30  # Strong moiré indication
        elif peak_ratio > 0.015 and mid_ratio > 0.45:
            score = 0.45  # Moderate moiré indication
        elif peak_ratio > 0.01 and mid_ratio > 0.43:
            score = 0.55  # Mild indication
        else:
            score = 0.80  # Normal webcam image (JPEG artifacts are expected)

        return score, details

    def _analyze_texture(self, face_img: np.ndarray) -> Tuple[float, dict]:
        """
        Analyze skin texture using Local Binary Pattern (LBP) concept.

        Real skin has characteristic micro-texture.
        Printed photos and screens have different texture patterns.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Compute LBP-like features using gradient patterns
        # More robust than standard LBP for this use case

        # Sobel gradients
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)

        # Compute gradient histogram
        hist_mag, _ = np.histogram(magnitude.flatten(), bins=32, range=(0, 100))
        hist_mag = hist_mag.astype(np.float32) / (hist_mag.sum() + 1e-6)

        hist_dir, _ = np.histogram(direction.flatten(), bins=16, range=(-np.pi, np.pi))
        hist_dir = hist_dir.astype(np.float32) / (hist_dir.sum() + 1e-6)

        # Texture uniformity (entropy)
        mag_entropy = -np.sum(hist_mag * np.log2(hist_mag + 1e-10))
        dir_entropy = -np.sum(hist_dir * np.log2(hist_dir + 1e-10))

        # Laplacian for high-frequency content
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = laplacian.var()
        lap_mean = np.abs(laplacian).mean()

        details = {
            'magnitude_entropy': float(mag_entropy),
            'direction_entropy': float(dir_entropy),
            'laplacian_variance': float(lap_var),
            'laplacian_mean': float(lap_mean),
        }

        # Real faces have moderate texture entropy
        # Screens often have very uniform textures (low entropy)
        # Prints can have too-sharp textures (high laplacian)

        score = 0.5

        # Entropy scoring
        if mag_entropy < 2.5:  # Too uniform (screen)
            score -= 0.25
        elif mag_entropy > 4.5:  # Too varied (print artifact)
            score -= 0.15
        else:
            score += 0.25

        # Laplacian scoring
        if lap_var > 1000:  # Too sharp (print)
            score -= 0.2
        elif lap_var < 50:  # Too smooth (low-quality screen)
            score -= 0.15
        else:
            score += 0.2

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        return score, details

    def _analyze_color_space(self, face_img: np.ndarray) -> Tuple[float, dict]:
        """
        Analyze color characteristics for skin tone validation.

        Screens have limited color gamut and different color distribution.
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)

        # Skin color detection in YCrCb
        # Typical skin: Cr 133-173, Cb 77-127
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        skin_mask = (
            (cr >= 133) & (cr <= 173) &
            (cb >= 77) & (cb <= 127)
        )
        skin_ratio = skin_mask.sum() / skin_mask.size

        # Color variance (screens are more uniform)
        h_var = hsv[:, :, 0].var()
        s_var = hsv[:, :, 1].var()
        v_var = hsv[:, :, 2].var()

        # Check for screen color banding
        # Screens often show quantization in color values
        h_hist, _ = np.histogram(hsv[:, :, 0].flatten(), bins=180)
        h_peaks = np.sum(h_hist > h_hist.mean() * 2)

        details = {
            'skin_ratio': float(skin_ratio),
            'hue_variance': float(h_var),
            'saturation_variance': float(s_var),
            'value_variance': float(v_var),
            'hue_peaks': int(h_peaks),
        }

        score = 0.5

        # Skin ratio scoring
        if skin_ratio < 0.1:
            score -= 0.2  # Very little skin detected
        elif skin_ratio > 0.6:
            score += 0.2  # Good skin detection

        # Variance scoring (screens are more uniform)
        total_var = h_var + s_var / 10 + v_var / 10
        if total_var < 100:
            score -= 0.2  # Too uniform
        elif total_var > 500:
            score += 0.15  # Good variation

        # Peak scoring (screens show discrete colors)
        if h_peaks < 5:
            score -= 0.15  # Too few color values

        score = max(0.0, min(1.0, score))

        return score, details

    def _detect_specular_highlights(self, face_img: np.ndarray) -> Tuple[float, dict]:
        """
        Detect specular highlights characteristic of screens.

        Screen displays have characteristic reflection patterns.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

        # Very bright regions (potential screen reflection)
        bright_mask = gray > 240
        bright_ratio = bright_mask.sum() / bright_mask.size

        # Very saturated bright regions (screen artifacts)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        screen_highlight_mask = (val > 200) & (sat < 30)
        screen_highlight_ratio = screen_highlight_mask.sum() / screen_highlight_mask.size

        # Check for rectangular highlight patterns (screen bezels)
        if bright_mask.sum() > 100:
            # Find contours in bright regions
            bright_uint8 = bright_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(bright_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            rectangular_count = 0
            for contour in contours:
                if cv2.contourArea(contour) > 50:
                    approx = cv2.approxPolyDP(contour,
                                              0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) == 4:  # Rectangular
                        rectangular_count += 1
        else:
            rectangular_count = 0

        details = {
            'bright_ratio': float(bright_ratio),
            'screen_highlight_ratio': float(screen_highlight_ratio),
            'rectangular_highlights': rectangular_count,
        }
        logger.info(f"[SPECULAR] bright_ratio={bright_ratio:.4f}, screen_highlight_ratio={screen_highlight_ratio:.4f}, rect_count={rectangular_count}")

        score = 0.8  # Higher base — real faces often have natural highlights

        # Scoring — only penalize strong indicators of screen display
        if screen_highlight_ratio > 0.10:
            score -= 0.35  # Very strong screen reflection
        elif screen_highlight_ratio > 0.05:
            score -= 0.20

        if rectangular_count > 3:
            score -= 0.25  # Many rectangular bright spots = screen bezels
        elif rectangular_count > 1:
            score -= 0.1

        if bright_ratio > 0.15:
            score -= 0.15  # Large overexposed areas

        score = max(0.0, min(1.0, score))

        return score, details

    def _analyze_noise_pattern(self, face_img: np.ndarray) -> Tuple[float, dict]:
        """
        Analyze image noise patterns.

        Real camera captures have characteristic sensor noise.
        Screen photos have different noise characteristics.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # High-pass filter to isolate noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blur

        noise_std = noise.std()
        noise_mean = np.abs(noise).mean()

        # Check noise spatial correlation
        # Real sensor noise is relatively uncorrelated
        # Screen noise (from compression, etc.) is often correlated

        # Compute autocorrelation at small lags
        noise_shifted_x = np.roll(noise, 1, axis=1)
        noise_shifted_y = np.roll(noise, 1, axis=0)

        # Guard against NaN from corrcoef when noise is all zeros (flat/overexposed regions)
        if noise_std < 1e-6:
            corr_x = 0.0
            corr_y = 0.0
        else:
            corr_x = np.corrcoef(noise.flatten(), noise_shifted_x.flatten())[0, 1]
            corr_y = np.corrcoef(noise.flatten(), noise_shifted_y.flatten())[0, 1]
            if np.isnan(corr_x):
                corr_x = 0.0
            if np.isnan(corr_y):
                corr_y = 0.0

        spatial_corr = (abs(corr_x) + abs(corr_y)) / 2

        details = {
            'noise_std': float(noise_std),
            'noise_mean': float(noise_mean),
            'spatial_correlation': float(spatial_corr),
        }

        score = 0.6

        # Real cameras have moderate noise
        if noise_std < 1.5:  # Too clean (heavy processing/screen)
            score -= 0.25
        elif noise_std > 8:  # Too noisy
            score -= 0.1
        else:
            score += 0.2

        # Low spatial correlation is expected for sensor noise
        if spatial_corr > 0.5:  # High correlation = processed/screen
            score -= 0.2
        elif spatial_corr < 0.3:
            score += 0.15

        score = max(0.0, min(1.0, score))

        return score, details


def get_spoof_detector() -> SpoofDetector:
    """Get or create the spoof detector singleton."""
    global _spoof_detector

    if _spoof_detector is None:
        _spoof_detector = SpoofDetector()

    return _spoof_detector


def check_spoof(
    image: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[float, dict]:
    """
    Check if an image is spoofed.

    Args:
        image: BGR image as numpy array
        face_bbox: Face bounding box (x1, y1, x2, y2)

    Returns:
        Tuple of (score, details)
        score > 0.5 means likely real

    Note: For security, recommend threshold of 0.55-0.6 for production
    """
    detector = get_spoof_detector()
    return detector.detect(image, face_bbox)
