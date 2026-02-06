# Frontend Architecture

## Overview

The frontend is a React application that handles:
1. Camera access via WebRTC
2. Real-time face detection via MediaPipe
3. Quality checks (lighting, obstructions, face position)
4. Liveness challenges (blink, smile, etc.)
5. Frame capture and submission to backend

## Technology

- **React 18** - UI framework
- **Vite** - Build tool (fast, modern)
- **Tailwind CSS** - Styling (utility-first)
- **MediaPipe Face Landmarker** - Face detection + blendshapes

## Component Structure

```
src/
├── main.jsx                    # Entry point
├── App.jsx                     # Main app with routing
├── index.css                   # Tailwind imports + custom styles
│
├── components/
│   ├── LivenessCheck.jsx       # Main orchestrator component
│   ├── CameraFeed.jsx          # WebRTC camera component
│   ├── FaceOverlay.jsx         # Canvas overlay for landmarks
│   ├── QualityGate.jsx         # Pre-challenge quality checks
│   ├── Challenge.jsx           # Individual challenge display
│   ├── ChallengeProgress.jsx   # Progress indicator (1/3, 2/3, 3/3)
│   ├── ResultScreen.jsx        # Success/failure display
│   └── DisputeModal.jsx        # "I'm really here" dispute flow
│
├── hooks/
│   ├── useFaceLandmarker.js    # MediaPipe initialization
│   ├── useLivenessState.js     # State machine logic
│   └── useCamera.js            # Camera stream management
│
├── lib/
│   ├── challenges.js           # Challenge definitions
│   ├── qualityChecks.js        # Quality check functions
│   ├── blendshapeDetector.js   # Detect blink, smile, etc.
│   └── frameCapture.js         # Capture best frame from video
│
└── api/
    └── verify.js               # Backend API calls
```

## State Machine

```javascript
const STATES = {
  IDLE: 'idle',
  INITIALIZING: 'initializing',
  QUALITY_CHECK: 'quality_check',
  CHALLENGE_1: 'challenge_1',
  CHALLENGE_2: 'challenge_2',
  CHALLENGE_3: 'challenge_3',
  CAPTURING: 'capturing',
  PROCESSING: 'processing',
  SUCCESS: 'success',
  FAILURE: 'failure',
  FLAGGED: 'flagged',
  DISPUTE: 'dispute'
};
```

## MediaPipe Setup

```javascript
// Load from CDN
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const faceLandmarker = await FaceLandmarker.createFromOptions(
  await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  ),
  {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU'
    },
    outputFaceBlendshapes: true,
    runningMode: 'VIDEO',
    numFaces: 1
  }
);
```

## Quality Checks

Each check returns `{ passed: boolean, message: string }`:

| Check | Logic |
|-------|-------|
| Face detected | `landmarks.length > 0` |
| Single face | `landmarks.length === 1` |
| Face centered | Nose landmark within center 60% of frame |
| Good lighting | Average pixel brightness between 80-200 |
| Eyes visible | Eye landmarks visibility > 0.8 |
| No mask | Nose + mouth landmarks visibility > 0.8 |

## Challenge Detection

### Blink Detection
```javascript
function detectBlink(blendshapes) {
  const leftBlink = getBlendshape(blendshapes, 'eyeBlinkLeft');
  const rightBlink = getBlendshape(blendshapes, 'eyeBlinkRight');
  return leftBlink > 0.6 && rightBlink > 0.6;
}
```

### Smile Detection
```javascript
function detectSmile(blendshapes) {
  const leftSmile = getBlendshape(blendshapes, 'mouthSmileLeft');
  const rightSmile = getBlendshape(blendshapes, 'mouthSmileRight');
  return leftSmile > 0.5 && rightSmile > 0.5;
}
```

### Head Turn Detection
```javascript
function detectHeadTurn(landmarks, direction) {
  // Calculate head rotation from nose and ear positions
  const nose = landmarks[1];
  const leftEar = landmarks[234];
  const rightEar = landmarks[454];

  const rotation = calculateYawAngle(nose, leftEar, rightEar);

  if (direction === 'left') return rotation < -15;
  if (direction === 'right') return rotation > 15;
}
```

## Frame Capture Strategy

1. During challenges, continuously store recent frames
2. On challenge success, pick frame with:
   - Highest face detection confidence
   - Best lighting
   - Face most centered
3. Convert to base64 for upload

## Design Principles (from DESIGN_GUIDE.md)

- **No decorative animations** - Only functional feedback
- **Clear hierarchy** - One primary action per screen
- **Calm colors** - Desaturated, professional palette
- **Clear messaging** - Tell user exactly what to do
- **Progress indication** - Always show where user is in flow

## Color Palette

```css
:root {
  --color-primary: #1a1a2e;      /* Deep navy - primary actions */
  --color-surface: #f8f9fa;      /* Light gray - backgrounds */
  --color-accent: #4361ee;       /* Blue - highlights */
  --color-success: #2d6a4f;      /* Green - success states */
  --color-warning: #bc6c25;      /* Amber - warnings */
  --color-error: #9b2226;        /* Red - errors */
  --color-text: #212529;         /* Dark - primary text */
  --color-text-muted: #6c757d;   /* Gray - secondary text */
}
```

## Responsive Breakpoints

- Mobile: < 640px (single column, full-width camera)
- Tablet: 640px - 1024px (centered camera with padding)
- Desktop: > 1024px (centered camera, max-width container)
