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

  console.log('%c[KYC API] POST /api/verify', 'color: #2563eb; font-weight: bold');
  console.log('[KYC API] Request:', {
    sessionId,
    challengesCompleted,
    imageSize: `${image.length} chars (${Math.round(image.length * 0.75 / 1024)} KB)`,
    motionAnalysis: motionAnalysis || 'none',
  });

  const startTime = performance.now();

  const response = await fetch(`${API_URL}/api/verify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Network error' }));
    console.error(`[KYC API] Response ERROR (${elapsed}s):`, response.status, error);
    throw new Error(error.message || 'Verification failed');
  }

  const result = await response.json();

  // Detailed response logging
  console.log(`%c[KYC API] Response (${elapsed}s): ${result.status}`,
    `color: ${result.status === 'success' ? '#16a34a' : result.status === 'pending_review' ? '#d97706' : '#dc2626'}; font-weight: bold`);
  console.log('[KYC API] Full response:', result);

  if (result.layer_results) {
    console.group('[KYC] Verification Layer Results');
    const layers = result.layer_results;
    if (layers.layer1_face_detection) {
      const l1 = layers.layer1_face_detection;
      console.log(`  L1 Face Detection : ${l1.status} (confidence: ${l1.confidence || 'N/A'})`);
    }
    if (layers.layer2_liveness) {
      const l2 = layers.layer2_liveness;
      console.log(`  L2 Liveness       : %c${l2.status}%c (score: ${l2.score})`,
        l2.status === 'passed' ? 'color: green' : 'color: red', '');
    }
    if (layers.layer3_deepfake) {
      const l3 = layers.layer3_deepfake;
      console.log(`  L3 Deepfake       : %c${l3.status}%c (score: ${l3.score})`,
        l3.status === 'passed' ? 'color: green' : 'color: red', '');
    }
    if (layers.layer4_duplicate) {
      const l4 = layers.layer4_duplicate;
      console.log(`  L4 Duplicate      : ${l4.status} (matches: ${l4.matches_found})`);
    }
    if (layers.layer5_risk_score) {
      const l5 = layers.layer5_risk_score;
      console.log(`  L5 Risk Score     : ${l5.score}/100 (${l5.level}) → ${l5.decision}`);
      if (l5.flags?.length) console.warn('  Risk flags:', l5.flags);
    }
    console.groupEnd();
  }

  if (result.risk_score !== undefined) {
    console.log(`[KYC] Risk: ${result.risk_score}/100 (${result.risk_level})`);
  }

  return result;
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
