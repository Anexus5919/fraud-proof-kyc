"""
Layer 3: Deepfake Detection

Detects AI-generated/manipulated face images using multiple techniques:
1. Frequency Domain Analysis - GAN artifacts in frequency spectrum
2. Face Consistency Check - Symmetry and proportion anomalies
3. GAN Artifact Detection - Checkerboard patterns from upsampling
4. Blending Boundary Detection - Face swap boundary artifacts
5. Eye and Teeth Analysis - Common deepfake failure points

All techniques use OpenCV (CPU-only, no external models needed).
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

_detector = None


class DeepfakeDetector:
    """Multi-technique deepfake detection engine."""

    def detect(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze image for deepfake indicators.

        Args:
            image: BGR image as numpy array

        Returns:
            Tuple of (authenticity_score, details)
            authenticity_score: 0-1, higher = more likely real
        """
        details = {}
        scores = []
        weights = []

        h, w = image.shape[:2]
        if h < 50 or w < 50:
            return 0.5, {'error': 'Image too small for analysis'}

        # 1. Frequency Domain Analysis (25% weight)
        freq_score, freq_details = self._frequency_analysis(image)
        details['frequency_analysis'] = freq_details
        scores.append(freq_score)
        weights.append(0.25)

        # 2. Face Consistency Check (20% weight)
        consistency_score, consistency_details = self._face_consistency(image)
        details['face_consistency'] = consistency_details
        scores.append(consistency_score)
        weights.append(0.20)

        # 3. GAN Artifact Detection (20% weight)
        gan_score, gan_details = self._gan_artifact_detection(image)
        details['gan_artifacts'] = gan_details
        scores.append(gan_score)
        weights.append(0.20)

        # 4. Blending Boundary Detection (15% weight)
        blend_score, blend_details = self._blending_analysis(image)
        details['blending_analysis'] = blend_details
        scores.append(blend_score)
        weights.append(0.15)

        # 5. Eye and Teeth Analysis (20% weight)
        eye_score, eye_details = self._eye_teeth_analysis(image)
        details['eye_teeth_analysis'] = eye_details
        scores.append(eye_score)
        weights.append(0.20)

        # Weighted combination
        total_weight = sum(weights)
        final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Check for critical flags
        critical_flag = None
        if freq_score < 0.25:
            critical_flag = 'gan_frequency_signature'
        elif gan_score < 0.25:
            critical_flag = 'gan_upsampling_artifacts'
        elif blend_score < 0.25:
            critical_flag = 'face_swap_boundary'

        if critical_flag:
            final_score = min(final_score, 0.35)
            details['critical_flag'] = critical_flag

        final_score = max(0.0, min(1.0, final_score))
        details['score'] = float(final_score)

        return float(final_score), details

    def _frequency_analysis(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Detect GAN artifacts in frequency domain.

        GANs produce characteristic artifacts in the high-frequency
        spectrum due to upsampling operations.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply FFT
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Analyze radial energy distribution
        # GANs often show unusual energy in specific frequency bands
        max_r = min(h, w) // 2
        radial_profile = np.zeros(max_r)
        counts = np.zeros(max_r)

        y_coords, x_coords = np.mgrid[:h, :w]
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(int)

        for r in range(max_r):
            mask = distances == r
            if mask.any():
                radial_profile[r] = magnitude[mask].mean()
                counts[r] = mask.sum()

        # Normalize
        if radial_profile.max() > 0:
            radial_profile /= radial_profile.max()

        # Check for abnormal peaks in high frequencies
        # Real images have smooth decay; GANs often have bumps
        mid_start = max_r // 4
        mid_end = 3 * max_r // 4
        high_start = 3 * max_r // 4

        mid_energy = radial_profile[mid_start:mid_end].mean() if mid_end > mid_start else 0
        high_energy = radial_profile[high_start:].mean() if high_start < max_r else 0

        # Compute smoothness of radial profile
        if len(radial_profile) > 2:
            profile_diff = np.diff(radial_profile[mid_start:])
            smoothness = np.abs(profile_diff).std()
        else:
            smoothness = 0

        details = {
            'mid_frequency_energy': float(mid_energy),
            'high_frequency_energy': float(high_energy),
            'profile_smoothness': float(smoothness),
        }

        # Score: smooth decay = real, spiky profile = GAN
        score = 0.7
        if smoothness > 0.08:
            score -= 0.3  # Spiky = potential GAN
        if high_energy > 0.3:
            score -= 0.2  # Too much high-freq energy
        if mid_energy < 0.1:
            score -= 0.15  # Unusual frequency gap

        score = max(0.0, min(1.0, score))
        details['score'] = float(score)
        return score, details

    def _face_consistency(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Check facial consistency and symmetry.

        Deepfakes often have subtle asymmetries and inconsistencies
        in lighting, color, and geometry across the face.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Split face into left and right halves
        mid = w // 2
        left_half = gray[:, :mid]
        right_half = cv2.flip(gray[:, mid:], 1)

        # Ensure same size
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_w]
        right_half = right_half[:, :min_w]

        # Compute symmetry score
        if left_half.size > 0 and right_half.size > 0:
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            symmetry_score = 1 - (diff.mean() / 255.0)
        else:
            symmetry_score = 0.5

        # Check color consistency across regions
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        top_half = hsv[:h // 2, :]
        bottom_half = hsv[h // 2:, :]

        top_hue_mean = top_half[:, :, 0].mean()
        bottom_hue_mean = bottom_half[:, :, 0].mean()
        hue_diff = abs(top_hue_mean - bottom_hue_mean)

        top_sat_std = top_half[:, :, 1].std()
        bottom_sat_std = bottom_half[:, :, 1].std()
        sat_diff = abs(top_sat_std - bottom_sat_std)

        details = {
            'symmetry_score': float(symmetry_score),
            'hue_consistency': float(hue_diff),
            'saturation_consistency': float(sat_diff),
        }

        # Real faces have moderate symmetry (not perfect)
        # Deepfakes are sometimes too symmetric or have color discontinuities
        score = 0.7

        if symmetry_score > 0.98:
            score -= 0.15  # Suspiciously symmetric
        elif symmetry_score < 0.7:
            score -= 0.2  # Very asymmetric (bad blend)

        if hue_diff > 15:
            score -= 0.2  # Color inconsistency across face
        if sat_diff > 30:
            score -= 0.15  # Saturation discontinuity

        score = max(0.0, min(1.0, score))
        details['score'] = float(score)
        return score, details

    def _gan_artifact_detection(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Detect checkerboard and grid artifacts from GAN upsampling.

        Transpose convolution in GANs produces characteristic
        checkerboard patterns visible in gradient domain.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute second-order derivatives to detect checkerboard
        kernel_x = np.array([[1, -2, 1]], dtype=np.float32)
        kernel_y = np.array([[1], [-2], [1]], dtype=np.float32)

        d2x = cv2.filter2D(gray, -1, kernel_x)
        d2y = cv2.filter2D(gray, -1, kernel_y)

        # Checkerboard pattern shows up as alternating sign pattern
        # in second derivatives
        d2x_sign = np.sign(d2x)
        d2y_sign = np.sign(d2y)

        # Count sign alternations
        x_alternations = np.abs(np.diff(d2x_sign, axis=1)).mean()
        y_alternations = np.abs(np.diff(d2y_sign, axis=0)).mean()

        # High alternation rate = potential checkerboard
        alternation_score = (x_alternations + y_alternations) / 2

        # Also check for periodic patterns in gradient
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Autocorrelation of gradients at stride-2 (GAN artifact period)
        if gx.shape[1] > 4:
            gx_corr = np.corrcoef(
                gx[:, :-2].flatten(),
                gx[:, 2:].flatten()
            )[0, 1]
        else:
            gx_corr = 0

        if gy.shape[0] > 4:
            gy_corr = np.corrcoef(
                gy[:-2, :].flatten(),
                gy[2:, :].flatten()
            )[0, 1]
        else:
            gy_corr = 0

        periodic_corr = (abs(gx_corr) + abs(gy_corr)) / 2

        details = {
            'alternation_score': float(alternation_score),
            'periodic_correlation': float(periodic_corr),
        }

        # Score
        score = 0.75
        if alternation_score > 1.2:
            score -= 0.3  # Strong checkerboard
        elif alternation_score > 0.9:
            score -= 0.15

        if periodic_corr > 0.6:
            score -= 0.25  # Strong periodic pattern
        elif periodic_corr > 0.4:
            score -= 0.1

        score = max(0.0, min(1.0, score))
        details['score'] = float(score)
        return score, details

    def _blending_analysis(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Detect face-swap blending boundaries.

        Face swaps often leave visible boundaries where the pasted
        face meets the original image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Look for sharp intensity transitions in the face boundary area
        # Face swaps typically have a boundary at ~70-90% radius from center
        cy, cx = h // 2, w // 2
        r_inner = int(min(h, w) * 0.3)
        r_outer = int(min(h, w) * 0.45)

        # Create ring mask for boundary region
        y_coords, x_coords = np.mgrid[:h, :w]
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        ring_mask = (distances >= r_inner) & (distances <= r_outer)

        # Compute gradient magnitude in boundary region
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        boundary_grad = grad_mag[ring_mask].mean() if ring_mask.any() else 0
        interior_grad = grad_mag[distances < r_inner].mean() if (distances < r_inner).any() else 0

        # Ratio of boundary gradient to interior gradient
        grad_ratio = boundary_grad / (interior_grad + 1e-6)

        # Check color discontinuity at boundary
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        boundary_hue_std = hsv[:, :, 0][ring_mask].std() if ring_mask.any() else 0
        interior_hue_std = hsv[:, :, 0][distances < r_inner].std() if (distances < r_inner).any() else 0
        hue_ratio = boundary_hue_std / (interior_hue_std + 1e-6)

        details = {
            'boundary_gradient': float(boundary_grad),
            'interior_gradient': float(interior_grad),
            'gradient_ratio': float(grad_ratio),
            'hue_discontinuity_ratio': float(hue_ratio),
        }

        score = 0.75

        # High gradient ratio at boundary = potential face swap
        if grad_ratio > 2.5:
            score -= 0.35
        elif grad_ratio > 1.8:
            score -= 0.2
        elif grad_ratio > 1.3:
            score -= 0.1

        # Color discontinuity at boundary
        if hue_ratio > 2.0:
            score -= 0.2
        elif hue_ratio > 1.5:
            score -= 0.1

        score = max(0.0, min(1.0, score))
        details['score'] = float(score)
        return score, details

    def _eye_teeth_analysis(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze eye and teeth regions for deepfake artifacts.

        Deepfakes often fail at rendering:
        - Consistent eye reflections (catch-lights)
        - Natural teeth boundaries
        - Symmetric iris detail
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Approximate eye regions (upper 30-50% of face, left and right quarters)
        eye_y_start = int(h * 0.25)
        eye_y_end = int(h * 0.45)
        left_eye = gray[eye_y_start:eye_y_end, :w // 2]
        right_eye = gray[eye_y_start:eye_y_end, w // 2:]

        # Compare eye region textures
        if left_eye.size > 100 and right_eye.size > 100:
            # Resize to same dimensions
            min_h = min(left_eye.shape[0], right_eye.shape[0])
            min_w = min(left_eye.shape[1], right_eye.shape[1])
            left_eye = left_eye[:min_h, :min_w]
            right_eye = cv2.flip(right_eye[:min_h, :min_w], 1)

            # Compare Laplacian (texture detail)
            left_lap = cv2.Laplacian(left_eye, cv2.CV_64F)
            right_lap = cv2.Laplacian(right_eye, cv2.CV_64F)

            eye_texture_diff = np.abs(left_lap.var() - right_lap.var())

            # Check for specular highlights (catch-lights)
            left_bright = (left_eye > 220).sum()
            right_bright = (right_eye > 220).sum()
            catchlight_diff = abs(left_bright - right_bright) / max(left_bright + right_bright, 1)
        else:
            eye_texture_diff = 0
            catchlight_diff = 0

        # Mouth/teeth region (lower 60-80% of face)
        mouth_region = gray[int(h * 0.6):int(h * 0.85), int(w * 0.25):int(w * 0.75)]
        if mouth_region.size > 100:
            mouth_lap = cv2.Laplacian(mouth_region, cv2.CV_64F)
            mouth_detail = mouth_lap.var()
        else:
            mouth_detail = 0

        details = {
            'eye_texture_difference': float(eye_texture_diff),
            'catchlight_asymmetry': float(catchlight_diff),
            'mouth_detail_level': float(mouth_detail),
        }

        score = 0.7

        # Large texture difference between eyes = potential deepfake
        if eye_texture_diff > 500:
            score -= 0.25
        elif eye_texture_diff > 200:
            score -= 0.1

        # Catch-light asymmetry
        if catchlight_diff > 0.7:
            score -= 0.2
        elif catchlight_diff > 0.4:
            score -= 0.1

        # Very low mouth detail = blurry teeth (common deepfake artifact)
        if mouth_detail < 10:
            score -= 0.15

        score = max(0.0, min(1.0, score))
        details['score'] = float(score)
        return score, details


def get_deepfake_detector() -> DeepfakeDetector:
    """Get or create the global deepfake detector instance."""
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector()
        logger.info("DeepfakeDetector initialized")
    return _detector


def check_deepfake(image: np.ndarray) -> Tuple[float, Dict]:
    """
    Check if an image is a deepfake.

    Args:
        image: BGR image as numpy array

    Returns:
        Tuple of (authenticity_score, details)
        authenticity_score: 0-1, higher = more likely real
    """
    detector = get_deepfake_detector()
    return detector.detect(image)
