# Liveness Challenge Specifications

## Overview

The liveness check uses **active challenges** - user must perform specific actions that are difficult to replicate with photos or videos.

Each session requires completing **3 random challenges** from a pool of 6.

## Challenge Pool

### 1. Blink

**Instruction:** "Blink your eyes"

**Detection:**
```javascript
const eyeBlinkLeft = getBlendshape('eyeBlinkLeft');
const eyeBlinkRight = getBlendshape('eyeBlinkRight');
const detected = eyeBlinkLeft > 0.6 && eyeBlinkRight > 0.6;
```

**Requirements:**
- Both eyes must close simultaneously
- Must hold for 3+ consecutive frames
- Threshold: 0.6 (strict)

**Why it works:** Photos have static open eyes. Videos would need exact timing.

---

### 2. Smile

**Instruction:** "Give us a smile"

**Detection:**
```javascript
const mouthSmileLeft = getBlendshape('mouthSmileLeft');
const mouthSmileRight = getBlendshape('mouthSmileRight');
const detected = mouthSmileLeft > 0.5 && mouthSmileRight > 0.5;
```

**Requirements:**
- Both sides of mouth must curve up
- Must hold for 5+ consecutive frames (longer to ensure genuine)
- Threshold: 0.5

**Why it works:** Hard to fake with static images. Requires actual muscle movement.

---

### 3. Open Mouth

**Instruction:** "Open your mouth wide"

**Detection:**
```javascript
const jawOpen = getBlendshape('jawOpen');
const detected = jawOpen > 0.6;
```

**Requirements:**
- Jaw must open significantly
- Must hold for 3+ consecutive frames
- Threshold: 0.6

**Why it works:** Requires 3D movement that's hard to replicate with flat media.

---

### 4. Raise Eyebrows

**Instruction:** "Raise your eyebrows"

**Detection:**
```javascript
const browInnerUp = getBlendshape('browInnerUp');
const browOuterUpLeft = getBlendshape('browOuterUpLeft');
const browOuterUpRight = getBlendshape('browOuterUpRight');
const detected = browInnerUp > 0.5 || (browOuterUpLeft > 0.4 && browOuterUpRight > 0.4);
```

**Requirements:**
- Eyebrows must visibly raise
- Must hold for 3+ consecutive frames
- Threshold: 0.5 inner, 0.4 outer

**Why it works:** Subtle movement that's difficult to predict and replay.

---

### 5. Turn Head Left

**Instruction:** "Slowly turn your head to the left"

**Detection:**
```javascript
// Calculate yaw angle from landmark positions
const noseTip = landmarks[1];
const leftCheek = landmarks[234];
const rightCheek = landmarks[454];

const faceWidth = distance(leftCheek, rightCheek);
const noseOffset = noseTip.x - ((leftCheek.x + rightCheek.x) / 2);
const yawAngle = (noseOffset / faceWidth) * 90;

const detected = yawAngle < -15; // Negative = turned left
```

**Requirements:**
- Head must turn at least 15 degrees left
- Must hold for 3+ consecutive frames
- Face must remain in frame

**Why it works:** Reveals 3D structure of face. Photos are flat and don't respond.

---

### 6. Turn Head Right

**Instruction:** "Slowly turn your head to the right"

**Detection:**
```javascript
// Same calculation as left, opposite direction
const detected = yawAngle > 15; // Positive = turned right
```

**Requirements:**
- Head must turn at least 15 degrees right
- Must hold for 3+ consecutive frames
- Face must remain in frame

---

## Randomization

```javascript
const CHALLENGE_POOL = [
  { id: 'blink', name: 'Blink', instruction: 'Blink your eyes' },
  { id: 'smile', name: 'Smile', instruction: 'Give us a smile' },
  { id: 'open_mouth', name: 'Open Mouth', instruction: 'Open your mouth wide' },
  { id: 'raise_eyebrows', name: 'Raise Eyebrows', instruction: 'Raise your eyebrows' },
  { id: 'turn_left', name: 'Turn Left', instruction: 'Slowly turn your head to the left' },
  { id: 'turn_right', name: 'Turn Right', instruction: 'Slowly turn your head to the right' },
];

function selectRandomChallenges(count = 3) {
  const shuffled = [...CHALLENGE_POOL].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}
```

## Timing

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Time per challenge | 8 seconds | Enough time for user to understand and perform |
| Hold frames required | 3-5 | Prevents accidental triggers |
| Transition delay | 1 second | Let user see success before next challenge |
| Total max time | ~30 seconds | 3 challenges + transitions |

## Feedback States

### During Challenge

| State | UI Feedback |
|-------|-------------|
| Waiting | Instruction text + countdown timer |
| Detecting | Subtle indicator that we're processing |
| Success | Green checkmark + brief success message |
| Timeout | Retry prompt (not failure) |

### Challenge Success Flow

```
Challenge instruction shown
    ↓
Timer starts (8 seconds)
    ↓
User performs action
    ↓
Detection hits threshold
    ↓
Hold for required frames (debounce)
    ↓
Success animation (200ms)
    ↓
Transition delay (1s)
    ↓
Next challenge OR capture phase
```

## Edge Cases

### User performs action before instruction shown
- Ignore. Only count after instruction is displayed.

### User performs wrong action
- No penalty. Just wait for correct action.

### User performs action then stops before hold requirement
- Reset hold counter. User must sustain the action.

### Timer runs out
- Show "Let's try again" message
- Restart same challenge (not a new random one)
- Allow 3 retries per challenge before moving on

### Face leaves frame during challenge
- Pause timer
- Show "We lost you - please face the camera"
- Resume when face returns

## Anti-Gaming Measures

1. **Random order** - Attackers can't prepare a video

2. **Hold requirement** - Single frame spikes don't count

3. **Strict thresholds** - Borderline detections are rejected

4. **Server-side verification** - Frontend results are not trusted; final check happens on backend with spoof detection

5. **Session binding** - Challenges are generated per session, can't be reused

## Blendshape Reference

Key blendshapes used (from MediaPipe's 52 blendshapes):

| Blendshape | Index | Description |
|------------|-------|-------------|
| eyeBlinkLeft | 9 | Left eye closed |
| eyeBlinkRight | 10 | Right eye closed |
| jawOpen | 25 | Mouth open |
| mouthSmileLeft | 44 | Left smile |
| mouthSmileRight | 45 | Right smile |
| browInnerUp | 1 | Inner eyebrows raised |
| browOuterUpLeft | 3 | Left outer eyebrow raised |
| browOuterUpRight | 4 | Right outer eyebrow raised |

Full list: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
