// Blendshape detection functions for liveness challenges

// Helper to get a specific blendshape value from the results
export function getBlendshapeValue(blendshapes, name) {
  if (!blendshapes || !blendshapes[0] || !blendshapes[0].categories) {
    return 0;
  }

  const shape = blendshapes[0].categories.find(
    (s) => s.categoryName === name || s.displayName === name
  );

  return shape ? shape.score : 0;
}

// Detect blink (both eyes closed)
export function detectBlink(blendshapes) {
  const leftBlink = getBlendshapeValue(blendshapes, 'eyeBlinkLeft');
  const rightBlink = getBlendshapeValue(blendshapes, 'eyeBlinkRight');

  // Strict threshold: both eyes must be significantly closed
  return leftBlink > 0.6 && rightBlink > 0.6;
}

// Detect smile
export function detectSmile(blendshapes) {
  const leftSmile = getBlendshapeValue(blendshapes, 'mouthSmileLeft');
  const rightSmile = getBlendshapeValue(blendshapes, 'mouthSmileRight');

  // Both sides must show smile
  return leftSmile > 0.5 && rightSmile > 0.5;
}

// Detect open mouth
export function detectOpenMouth(blendshapes) {
  const jawOpen = getBlendshapeValue(blendshapes, 'jawOpen');

  // Mouth must be significantly open
  return jawOpen > 0.6;
}

// Detect raised eyebrows
export function detectRaiseEyebrows(blendshapes) {
  const browInnerUp = getBlendshapeValue(blendshapes, 'browInnerUp');
  const browOuterUpLeft = getBlendshapeValue(blendshapes, 'browOuterUpLeft');
  const browOuterUpRight = getBlendshapeValue(blendshapes, 'browOuterUpRight');

  // Either inner brow up, or both outer brows up
  return browInnerUp > 0.5 || (browOuterUpLeft > 0.4 && browOuterUpRight > 0.4);
}

// Calculate head yaw angle from landmarks
export function calculateHeadYaw(landmarks) {
  if (!landmarks || landmarks.length < 468) {
    return 0;
  }

  // Key landmarks for head pose
  const noseTip = landmarks[1];
  const leftCheek = landmarks[234];
  const rightCheek = landmarks[454];

  if (!noseTip || !leftCheek || !rightCheek) {
    return 0;
  }

  // Calculate face width and nose offset
  const faceWidth = Math.abs(rightCheek.x - leftCheek.x);
  const faceCenterX = (leftCheek.x + rightCheek.x) / 2;
  const noseOffset = noseTip.x - faceCenterX;

  // Convert to approximate angle (simplified)
  // When face turns left, nose moves right relative to cheeks (positive offset)
  // When face turns right, nose moves left relative to cheeks (negative offset)
  const yawAngle = (noseOffset / faceWidth) * -90;

  return yawAngle;
}

// Detect head turn left
export function detectTurnLeft(blendshapes, landmarks) {
  const yaw = calculateHeadYaw(landmarks);
  // Negative yaw = turned left
  return yaw < -15;
}

// Detect head turn right
export function detectTurnRight(blendshapes, landmarks) {
  const yaw = calculateHeadYaw(landmarks);
  // Positive yaw = turned right
  return yaw > 15;
}

// Map of detection functions
export const detectionFunctions = {
  detectBlink,
  detectSmile,
  detectOpenMouth,
  detectRaiseEyebrows,
  detectTurnLeft,
  detectTurnRight,
};

// Get detection function by name
export function getDetectionFunction(name) {
  return detectionFunctions[name] || (() => false);
}
