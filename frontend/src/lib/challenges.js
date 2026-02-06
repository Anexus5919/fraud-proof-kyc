// Challenge definitions for liveness detection

export const CHALLENGES = [
  {
    id: 'blink',
    name: 'Blink',
    instruction: 'Blink your eyes',
    detectFn: 'detectBlink',
    holdFrames: 3,
    timeLimit: 8000,
  },
  {
    id: 'smile',
    name: 'Smile',
    instruction: 'Give us a smile',
    detectFn: 'detectSmile',
    holdFrames: 5,
    timeLimit: 8000,
  },
  {
    id: 'open_mouth',
    name: 'Open Mouth',
    instruction: 'Open your mouth wide',
    detectFn: 'detectOpenMouth',
    holdFrames: 3,
    timeLimit: 8000,
  },
  {
    id: 'raise_eyebrows',
    name: 'Raise Eyebrows',
    instruction: 'Raise your eyebrows',
    detectFn: 'detectRaiseEyebrows',
    holdFrames: 3,
    timeLimit: 8000,
  },
  {
    id: 'turn_left',
    name: 'Turn Left',
    instruction: 'Slowly turn your head to the left',
    detectFn: 'detectTurnLeft',
    holdFrames: 3,
    timeLimit: 8000,
  },
  {
    id: 'turn_right',
    name: 'Turn Right',
    instruction: 'Slowly turn your head to the right',
    detectFn: 'detectTurnRight',
    holdFrames: 3,
    timeLimit: 8000,
  },
];

// Shuffle array using Fisher-Yates algorithm
function shuffleArray(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// Select random challenges for a session
export function selectRandomChallenges(count = 3) {
  const shuffled = shuffleArray(CHALLENGES);
  return shuffled.slice(0, count);
}

// Get challenge by ID
export function getChallengeById(id) {
  return CHALLENGES.find(c => c.id === id);
}
