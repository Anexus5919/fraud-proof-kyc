// API calls to backend for verification

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Convert canvas to base64 image
export function canvasToBase64(canvas) {
  return canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
}

// Submit face for verification
export async function verifyFace({ image, sessionId, challengesCompleted, motionAnalysis }) {
  const body = {
    image,
    session_id: sessionId,
    challenges_completed: challengesCompleted,
  };
  if (motionAnalysis) {
    body.motionAnalysis = motionAnalysis;
  }

  const response = await fetch(`${API_URL}/api/verify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Network error' }));
    throw new Error(error.message || 'Verification failed');
  }

  return response.json();
}

// Submit dispute when user claims they are real
export async function submitDispute({ sessionId, image, reason }) {
  const response = await fetch(`${API_URL}/api/verify/dispute`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      session_id: sessionId,
      image,
      reason,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Network error' }));
    throw new Error(error.message || 'Dispute submission failed');
  }

  return response.json();
}

// Generate unique session ID
export function generateSessionId() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}
