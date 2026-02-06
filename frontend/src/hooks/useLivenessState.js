import { useReducer, useCallback, useRef } from 'react';
import { selectRandomChallenges } from '../lib/challenges';
import { getDetectionFunction } from '../lib/blendshapeDetector';
import { generateSessionId } from '../api/verify';

// State machine states
export const STATES = {
  IDLE: 'idle',
  INITIALIZING: 'initializing',
  QUALITY_CHECK: 'quality_check',
  CHALLENGE_1: 'challenge_1',
  CHALLENGE_2: 'challenge_2',
  CHALLENGE_3: 'challenge_3',
  READY_TO_CAPTURE: 'ready_to_capture',
  CAPTURING: 'capturing',
  PROCESSING: 'processing',
  SUCCESS: 'success',
  FAILURE: 'failure',
  FLAGGED: 'flagged',
  DISPUTE: 'dispute',
};

// Action types
const ACTIONS = {
  START: 'START',
  INITIALIZED: 'INITIALIZED',
  QUALITY_PASSED: 'QUALITY_PASSED',
  CHALLENGE_PASSED: 'CHALLENGE_PASSED',
  CAPTURE_CONFIRMED: 'CAPTURE_CONFIRMED',
  CAPTURE_COMPLETE: 'CAPTURE_COMPLETE',
  VERIFY_SUCCESS: 'VERIFY_SUCCESS',
  VERIFY_FAILURE: 'VERIFY_FAILURE',
  VERIFY_FLAGGED: 'VERIFY_FLAGGED',
  START_DISPUTE: 'START_DISPUTE',
  DISPUTE_SUBMITTED: 'DISPUTE_SUBMITTED',
  RESET: 'RESET',
  SET_ERROR: 'SET_ERROR',
  UPDATE_HOLD_COUNT: 'UPDATE_HOLD_COUNT',
};

// Initial state
const initialState = {
  currentState: STATES.IDLE,
  sessionId: null,
  challenges: [],
  currentChallengeIndex: 0,
  completedChallenges: [],
  holdFrameCount: 0,
  capturedImage: null,
  verificationResult: null,
  error: null,
};

// Reducer
function livenessReducer(state, action) {
  switch (action.type) {
    case ACTIONS.START:
      return {
        ...initialState,
        currentState: STATES.INITIALIZING,
        sessionId: generateSessionId(),
        challenges: selectRandomChallenges(3),
      };

    case ACTIONS.INITIALIZED:
      return {
        ...state,
        currentState: STATES.QUALITY_CHECK,
      };

    case ACTIONS.QUALITY_PASSED:
      return {
        ...state,
        currentState: STATES.CHALLENGE_1,
        holdFrameCount: 0,
      };

    case ACTIONS.UPDATE_HOLD_COUNT:
      return {
        ...state,
        holdFrameCount: action.payload,
      };

    case ACTIONS.CHALLENGE_PASSED: {
      const newCompletedChallenges = [
        ...state.completedChallenges,
        state.challenges[state.currentChallengeIndex].id,
      ];

      const nextIndex = state.currentChallengeIndex + 1;

      if (nextIndex >= state.challenges.length) {
        // All challenges completed → go to capture stage (user must confirm)
        return {
          ...state,
          currentState: STATES.READY_TO_CAPTURE,
          completedChallenges: newCompletedChallenges,
          holdFrameCount: 0,
        };
      }

      // Move to next challenge
      const nextState =
        nextIndex === 1
          ? STATES.CHALLENGE_2
          : STATES.CHALLENGE_3;

      return {
        ...state,
        currentState: nextState,
        currentChallengeIndex: nextIndex,
        completedChallenges: newCompletedChallenges,
        holdFrameCount: 0,
      };
    }

    case ACTIONS.CAPTURE_CONFIRMED:
      // Security: only allow capture if all challenges were completed
      if (state.completedChallenges.length < state.challenges.length) {
        return state;
      }
      return {
        ...state,
        currentState: STATES.CAPTURING,
      };

    case ACTIONS.CAPTURE_COMPLETE:
      return {
        ...state,
        currentState: STATES.PROCESSING,
        capturedImage: action.payload,
      };

    case ACTIONS.VERIFY_SUCCESS:
      return {
        ...state,
        currentState: STATES.SUCCESS,
        verificationResult: action.payload,
      };

    case ACTIONS.VERIFY_FAILURE:
      return {
        ...state,
        currentState: STATES.FAILURE,
        verificationResult: action.payload,
        error: action.payload?.message,
      };

    case ACTIONS.VERIFY_FLAGGED:
      return {
        ...state,
        currentState: STATES.FLAGGED,
        verificationResult: action.payload,
      };

    case ACTIONS.START_DISPUTE:
      return {
        ...state,
        currentState: STATES.DISPUTE,
      };

    case ACTIONS.DISPUTE_SUBMITTED:
      return {
        ...state,
        currentState: STATES.FLAGGED,
        verificationResult: {
          ...state.verificationResult,
          disputeSubmitted: true,
        },
      };

    case ACTIONS.RESET:
      return initialState;

    case ACTIONS.SET_ERROR:
      return {
        ...state,
        error: action.payload,
      };

    default:
      return state;
  }
}

export function useLivenessState() {
  const [state, dispatch] = useReducer(livenessReducer, initialState);
  const holdCountRef = useRef(0);

  // Start the liveness check flow
  const start = useCallback(() => {
    holdCountRef.current = 0;
    dispatch({ type: ACTIONS.START });
  }, []);

  // Called when MediaPipe is ready
  const onInitialized = useCallback(() => {
    dispatch({ type: ACTIONS.INITIALIZED });
  }, []);

  // Called when quality checks pass
  const onQualityPassed = useCallback(() => {
    holdCountRef.current = 0;
    dispatch({ type: ACTIONS.QUALITY_PASSED });
  }, []);

  // Check if current challenge is detected
  const checkChallenge = useCallback(
    (blendshapes, landmarks) => {
      const currentChallenge = state.challenges[state.currentChallengeIndex];
      if (!currentChallenge) return false;

      const detectFn = getDetectionFunction(currentChallenge.detectFn);
      const detected = detectFn(blendshapes, landmarks);

      if (detected) {
        holdCountRef.current += 1;
        dispatch({ type: ACTIONS.UPDATE_HOLD_COUNT, payload: holdCountRef.current });

        // Check if held long enough
        if (holdCountRef.current >= currentChallenge.holdFrames) {
          return true;
        }
      } else {
        // Reset hold count if detection lost
        holdCountRef.current = 0;
        dispatch({ type: ACTIONS.UPDATE_HOLD_COUNT, payload: 0 });
      }

      return false;
    },
    [state.challenges, state.currentChallengeIndex]
  );

  // Called when a challenge is completed
  const onChallengePassed = useCallback(() => {
    holdCountRef.current = 0;
    dispatch({ type: ACTIONS.CHALLENGE_PASSED });
  }, []);

  // Called when user confirms capture (spacebar press with quality checks passing)
  const onCaptureConfirmed = useCallback(() => {
    dispatch({ type: ACTIONS.CAPTURE_CONFIRMED });
  }, []);

  // Called when frame is captured
  const onCaptureComplete = useCallback((imageData) => {
    dispatch({ type: ACTIONS.CAPTURE_COMPLETE, payload: imageData });
  }, []);

  // Called when verification succeeds
  const onVerifySuccess = useCallback((result) => {
    dispatch({ type: ACTIONS.VERIFY_SUCCESS, payload: result });
  }, []);

  // Called when verification fails
  const onVerifyFailure = useCallback((result) => {
    dispatch({ type: ACTIONS.VERIFY_FAILURE, payload: result });
  }, []);

  // Called when verification is flagged for review
  const onVerifyFlagged = useCallback((result) => {
    dispatch({ type: ACTIONS.VERIFY_FLAGGED, payload: result });
  }, []);

  // Start dispute process
  const startDispute = useCallback(() => {
    dispatch({ type: ACTIONS.START_DISPUTE });
  }, []);

  // Called when dispute is submitted
  const onDisputeSubmitted = useCallback(() => {
    dispatch({ type: ACTIONS.DISPUTE_SUBMITTED });
  }, []);

  // Reset to initial state
  const reset = useCallback(() => {
    holdCountRef.current = 0;
    dispatch({ type: ACTIONS.RESET });
  }, []);

  // Set error
  const setError = useCallback((error) => {
    dispatch({ type: ACTIONS.SET_ERROR, payload: error });
  }, []);

  // Get current challenge
  const getCurrentChallenge = useCallback(() => {
    return state.challenges[state.currentChallengeIndex] || null;
  }, [state.challenges, state.currentChallengeIndex]);

  // Check if in a challenge state
  const isInChallengeState = useCallback(() => {
    return [STATES.CHALLENGE_1, STATES.CHALLENGE_2, STATES.CHALLENGE_3].includes(
      state.currentState
    );
  }, [state.currentState]);

  return {
    state,
    start,
    onInitialized,
    onQualityPassed,
    checkChallenge,
    onChallengePassed,
    onCaptureConfirmed,
    onCaptureComplete,
    onVerifySuccess,
    onVerifyFailure,
    onVerifyFlagged,
    startDispute,
    onDisputeSubmitted,
    reset,
    setError,
    getCurrentChallenge,
    isInChallengeState,
  };
}
