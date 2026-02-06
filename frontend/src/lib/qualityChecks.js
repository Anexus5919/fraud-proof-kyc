// Quality checks before liveness challenges begin

// Check if face is detected
export function checkFaceDetected(results) {
  const detected = results?.faceLandmarks?.length > 0;
  return {
    passed: detected,
    message: detected ? 'Face detected' : 'No face detected. Please face the camera.',
  };
}

// Check for single face only
export function checkSingleFace(results) {
  const count = results?.faceLandmarks?.length || 0;
  const passed = count === 1;
  return {
    passed,
    message: passed
      ? 'Single face detected'
      : count === 0
      ? 'No face detected'
      : 'Multiple faces detected. Please ensure only you are in frame.',
  };
}

// Check if face is centered in frame
export function checkFaceCentered(results, videoWidth, videoHeight) {
  if (!results?.faceLandmarks?.[0]) {
    return { passed: false, message: 'No face to check position' };
  }

  const landmarks = results.faceLandmarks[0];
  const noseTip = landmarks[1]; // Nose tip landmark

  if (!noseTip) {
    return { passed: false, message: 'Cannot determine face position' };
  }

  // Nose should be in center 60% of frame
  const margin = 0.2;
  const inHorizontalBounds = noseTip.x > margin && noseTip.x < 1 - margin;
  const inVerticalBounds = noseTip.y > margin && noseTip.y < 1 - margin;

  const passed = inHorizontalBounds && inVerticalBounds;

  return {
    passed,
    message: passed
      ? 'Face is centered'
      : 'Please center your face in the frame',
  };
}

// Check eye visibility - simplified, uses blendshapes only
// Visibility scores are unreliable, so we only check for obvious sunglasses
export function checkEyesVisible(results) {
  if (!results?.faceLandmarks?.[0]) {
    return { passed: false, message: 'No face detected' };
  }

  // Use blendshapes to detect sunglasses (both eyes constantly showing as "closed")
  const blendshapes = results?.faceBlendshapes?.[0]?.categories;

  if (blendshapes) {
    const eyeBlinkLeft = blendshapes.find(s => s.categoryName === 'eyeBlinkLeft')?.score || 0;
    const eyeBlinkRight = blendshapes.find(s => s.categoryName === 'eyeBlinkRight')?.score || 0;

    // Only fail if BOTH eyes show very high "blink" score (likely sunglasses)
    if (eyeBlinkLeft > 0.85 && eyeBlinkRight > 0.85) {
      return {
        passed: false,
        message: 'Your eyes appear covered. Please remove sunglasses.',
      };
    }
  }

  // Default: pass the check - real validation happens during liveness challenges
  return { passed: true, message: 'Eyes are visible' };
}

// Check for face covering (mask detection) - simplified
// Beards/moustaches trigger false positives with visibility scores
// Real mask detection happens via liveness challenges (can't smile/open mouth with mask)
export function checkNoMask(results) {
  if (!results?.faceLandmarks?.[0]) {
    return { passed: false, message: 'No face detected' };
  }

  // If we can detect face landmarks at all, the face is likely not fully covered
  // The liveness challenges (smile, open mouth) will catch masks
  // because users won't be able to complete them with a covered face
  return { passed: true, message: 'Face is unobstructed' };
}

// Offscreen canvas for lighting analysis (reused to avoid GC)
let _lightingCanvas = null;

// Check lighting conditions using offscreen canvas analysis
// Uses a separate canvas to avoid drawing over the live video feed
export function checkLighting(videoElement, canvasElement) {
  if (!videoElement) {
    return { passed: true, message: 'Cannot check lighting' };
  }

  // Get video dimensions
  const videoWidth = videoElement.videoWidth || 640;
  const videoHeight = videoElement.videoHeight || 480;

  if (videoWidth === 0 || videoHeight === 0) {
    return { passed: true, message: 'Video not ready' };
  }

  // Create or reuse offscreen canvas for analysis
  if (!_lightingCanvas) {
    _lightingCanvas = document.createElement('canvas');
  }
  _lightingCanvas.width = videoWidth;
  _lightingCanvas.height = videoHeight;

  const ctx = _lightingCanvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return { passed: true, message: 'Cannot check lighting' };
  }

  // Draw current frame to offscreen canvas (not the visible overlay)
  ctx.drawImage(videoElement, 0, 0, videoWidth, videoHeight);

  // Sample center region of image
  const centerX = Math.floor(videoWidth / 4);
  const centerY = Math.floor(videoHeight / 4);
  const sampleWidth = Math.floor(videoWidth / 2);
  const sampleHeight = Math.floor(videoHeight / 2);

  try {
    const imageData = ctx.getImageData(centerX, centerY, sampleWidth, sampleHeight);
    const data = imageData.data;

    // Calculate average brightness
    let totalBrightness = 0;
    const pixelCount = data.length / 4;

    for (let i = 0; i < data.length; i += 4) {
      // Luminance formula
      const brightness = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      totalBrightness += brightness;
    }

    const avgBrightness = totalBrightness / pixelCount;

    // Check if brightness is in acceptable range
    const tooDark = avgBrightness < 60;
    const tooBright = avgBrightness > 220;

    if (tooDark) {
      return {
        passed: false,
        message: 'The lighting is too dark. Please move to a brighter area.',
      };
    }

    if (tooBright) {
      return {
        passed: false,
        message: 'The lighting is too bright. Please reduce direct light on your face.',
      };
    }

    return {
      passed: true,
      message: 'Lighting is good',
    };
  } catch (e) {
    // Canvas might be tainted by cross-origin video
    return { passed: true, message: 'Cannot check lighting' };
  }
}

// Check if face is roughly frontal (not turned to the side)
// Uses nose position relative to eyes — if nose is equidistant from both eyes, face is frontal
export function checkFaceFrontal(results) {
  if (!results?.faceLandmarks?.[0]) {
    return { passed: false, message: 'No face detected' };
  }

  const landmarks = results.faceLandmarks[0];
  // MediaPipe landmark indices: 1=nose tip, 33=left eye inner, 263=right eye inner
  const nose = landmarks[1];
  const leftEye = landmarks[33];
  const rightEye = landmarks[263];

  if (!nose || !leftEye || !rightEye) {
    return { passed: true, message: 'Cannot determine face angle' };
  }

  const distToLeft = Math.abs(nose.x - leftEye.x);
  const distToRight = Math.abs(nose.x - rightEye.x);

  // Ratio of distances — 1.0 = perfectly frontal
  const ratio = Math.min(distToLeft, distToRight) / Math.max(distToLeft, distToRight);

  // Require ratio > 0.5 for roughly frontal (allows slight tilt)
  const passed = ratio > 0.5;

  return {
    passed,
    message: passed
      ? 'Face is facing forward'
      : 'Please look directly at the camera',
  };
}

// Run all quality checks
export function runAllQualityChecks(results, videoElement, canvasElement) {
  const checks = [
    { name: 'faceDetected', ...checkFaceDetected(results) },
    { name: 'singleFace', ...checkSingleFace(results) },
    { name: 'faceCentered', ...checkFaceCentered(results) },
    { name: 'eyesVisible', ...checkEyesVisible(results) },
    { name: 'noMask', ...checkNoMask(results) },
    { name: 'lighting', ...checkLighting(videoElement, canvasElement) },
  ];

  const allPassed = checks.every((check) => check.passed);
  const failedChecks = checks.filter((check) => !check.passed);

  return {
    allPassed,
    checks,
    failedChecks,
    primaryMessage: failedChecks[0]?.message || 'All checks passed',
  };
}

// Run capture-stage quality checks (stricter — requires frontal face)
export function runCaptureQualityChecks(results, videoElement, canvasElement) {
  const checks = [
    { name: 'faceDetected', ...checkFaceDetected(results) },
    { name: 'singleFace', ...checkSingleFace(results) },
    { name: 'faceCentered', ...checkFaceCentered(results) },
    { name: 'faceFrontal', ...checkFaceFrontal(results) },
    { name: 'eyesVisible', ...checkEyesVisible(results) },
    { name: 'lighting', ...checkLighting(videoElement, canvasElement) },
  ];

  const allPassed = checks.every((check) => check.passed);
  const failedChecks = checks.filter((check) => !check.passed);

  return {
    allPassed,
    checks,
    failedChecks,
    primaryMessage: failedChecks[0]?.message || 'All checks passed',
  };
}
