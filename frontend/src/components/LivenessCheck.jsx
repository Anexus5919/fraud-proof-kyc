import { useEffect, useRef, useCallback, useState } from 'react';
import { useCamera } from '../hooks/useCamera';
import { useFaceLandmarker } from '../hooks/useFaceLandmarker';
import { useLivenessState, STATES } from '../hooks/useLivenessState';
import { runAllQualityChecks } from '../lib/qualityChecks';
import { getMotionAnalyzer, resetMotionAnalyzer } from '../lib/motionAnalyzer';
import { verifyFace, submitDispute, canvasToBase64 } from '../api/verify';
import CameraFeed from './CameraFeed';
import QualityGate from './QualityGate';
import Challenge from './Challenge';
import ResultScreen from './ResultScreen';

function LivenessCheck() {
  const {
    videoRef,
    isStreaming,
    hasPermission,
    error: cameraError,
    startCamera,
    stopCamera,
    captureFrame,
  } = useCamera();

  const {
    isLoading: isLoadingMediaPipe,
    isReady: isMediaPipeReady,
    error: mediaPipeError,
    detectForVideo,
  } = useFaceLandmarker();

  const {
    state,
    start,
    onInitialized,
    onQualityPassed,
    checkChallenge,
    onChallengePassed,
    onCaptureComplete,
    onVerifySuccess,
    onVerifyFailure,
    onVerifyFlagged,
    startDispute,
    onDisputeSubmitted,
    reset,
    getCurrentChallenge,
    isInChallengeState,
  } = useLivenessState();

  const canvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const isProcessingRef = useRef(false); // Synchronous guard — prevents duplicate API calls across animation frames

  const [qualityResults, setQualityResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false); // Async state for UI rendering only
  const [qualityPassed, setQualityPassed] = useState(false); // Tracks if user can start challenges

  // Set canvas ref callback
  const setCanvasRef = useCallback((node) => {
    canvasRef.current = node;
  }, []);

  // Main detection loop
  const runDetectionLoop = useCallback(() => {
    if (!videoRef.current || !isMediaPipeReady) {
      animationFrameRef.current = requestAnimationFrame(runDetectionLoop);
      return;
    }

    const results = detectForVideo(videoRef.current, performance.now());

    if (results) {
      // Quality check phase - continuous checking, no auto-transition
      if (state.currentState === STATES.QUALITY_CHECK) {
        const quality = runAllQualityChecks(
          results,
          videoRef.current,
          canvasRef.current
        );
        setQualityResults(quality);
        if (quality.allPassed && !qualityPassed) {
          console.log('%c[KYC] All quality checks PASSED', 'color: #16a34a; font-weight: bold');
        }
        setQualityPassed(quality.allPassed);
      }

      // Challenge phase - also track motion for anti-spoofing
      if (isInChallengeState()) {
        const blendshapes = results.faceBlendshapes;
        const landmarks = results.faceLandmarks?.[0];

        // Track landmarks for motion analysis (anti-spoofing)
        if (landmarks) {
          const motionAnalyzer = getMotionAnalyzer();
          motionAnalyzer.addFrame(landmarks);
        }

        const challengePassed = checkChallenge(blendshapes, landmarks);

        if (challengePassed) {
          const ch = getCurrentChallenge();
          console.log(`%c[KYC] Challenge PASSED: ${ch?.name || 'unknown'}`, 'color: #16a34a; font-weight: bold');
          onChallengePassed();
        }
      }

      // Capturing phase — use ref for synchronous check to prevent duplicate calls
      if (state.currentState === STATES.CAPTURING && !isProcessingRef.current) {
        handleCapture();
      }
    }

    animationFrameRef.current = requestAnimationFrame(runDetectionLoop);
  }, [
    videoRef,
    isMediaPipeReady,
    detectForVideo,
    state.currentState,
    qualityPassed,
    isInChallengeState,
    checkChallenge,
    getCurrentChallenge,
    onQualityPassed,
    onChallengePassed,
  ]);

  // Handle frame capture and verification
  const handleCapture = useCallback(async () => {
    // Synchronous ref check — blocks next animation frame immediately
    if (isProcessingRef.current) return;
    isProcessingRef.current = true;
    setIsProcessing(true); // Update UI state (async, but ref already blocks duplicates)

    console.log('%c[KYC] Starting capture & verification...', 'color: #2563eb; font-weight: bold');
    console.log('[KYC] Session:', state.sessionId);
    console.log('[KYC] Challenges completed:', state.completedChallenges);

    try {
      // Analyze motion data for anti-spoofing
      const motionAnalyzer = getMotionAnalyzer();
      const motionAnalysis = motionAnalyzer.analyze();
      const motionSummary = motionAnalyzer.getSummary();

      console.log('[KYC] Motion analysis:', {
        ready: motionAnalysis.ready,
        isLikelyReal: motionAnalysis.isLikelyReal,
        livenessScore: motionAnalysis.livenessScore,
        flags: motionAnalysis.flags,
        summary: motionSummary,
      });

      // Client-side motion check - reject obvious spoofs early
      if (motionAnalysis.ready && !motionAnalysis.isLikelyReal) {
        console.warn('[KYC] Motion analysis detected potential spoof:', motionAnalysis.flags);

        // If motion analysis strongly suggests a spoof, warn user
        if (motionAnalysis.livenessScore < 0.3) {
          onVerifyFailure({
            status: 'spoof_detected',
            message: 'We detected unnatural motion patterns. Please use a live camera, not a photo or video.',
            canDispute: true,
          });
          isProcessingRef.current = false;
          setIsProcessing(false);
          return;
        }
      }

      // Create capture canvas if not exists
      if (!captureCanvasRef.current) {
        captureCanvasRef.current = document.createElement('canvas');
      }

      const imageData = captureFrame(captureCanvasRef.current);

      if (!imageData) {
        throw new Error('Failed to capture frame');
      }

      onCaptureComplete(imageData);

      // Send to backend for verification with motion analysis data
      const result = await verifyFace({
        image: imageData.split(',')[1], // Remove data URL prefix
        sessionId: state.sessionId,
        challengesCompleted: state.completedChallenges,
        motionAnalysis: motionSummary, // Include motion data for server-side validation
      });

      if (result.status === 'success') {
        console.log('%c[KYC] RESULT: SUCCESS', 'color: #16a34a; font-weight: bold; font-size: 14px');
        onVerifySuccess(result);
      } else if (result.status === 'duplicate_found' || result.status === 'flagged' || result.status === 'pending_review') {
        console.log('%c[KYC] RESULT: PENDING REVIEW', 'color: #d97706; font-weight: bold; font-size: 14px');
        onVerifyFlagged(result);
      } else {
        // Handle spoof_detected, deepfake_detected, rejected, error
        console.log(`%c[KYC] RESULT: FAILED (${result.status})`, 'color: #dc2626; font-weight: bold; font-size: 14px');
        console.log('[KYC] Failure details:', result);
        onVerifyFailure(result);
      }
    } catch (error) {
      console.error('[KYC] Verification exception:', error);
      onVerifyFailure({
        status: 'error',
        message: error.message || 'Verification failed. Please try again.',
        canDispute: true,
      });
    } finally {
      isProcessingRef.current = false;
      setIsProcessing(false);
    }
  }, [
    captureFrame,
    onCaptureComplete,
    state.sessionId,
    state.completedChallenges,
    onVerifySuccess,
    onVerifyFlagged,
    onVerifyFailure,
  ]);

  // Handle dispute submission
  const handleDispute = useCallback(async () => {
    try {
      setIsProcessing(true);
      startDispute();

      await submitDispute({
        sessionId: state.sessionId,
        image: state.capturedImage?.split(',')[1],
        reason: state.verificationResult?.status || 'verification_failed',
      });

      onDisputeSubmitted();
    } catch (error) {
      console.error('Dispute error:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [
    state.sessionId,
    state.capturedImage,
    state.verificationResult,
    startDispute,
    onDisputeSubmitted,
  ]);

  // Handle retry
  const handleRetry = useCallback(() => {
    console.log('%c[KYC] === RETRY — resetting everything ===', 'color: #d97706; font-weight: bold');
    isProcessingRef.current = false;
    setIsProcessing(false);
    setQualityPassed(false);
    setQualityResults(null);
    resetMotionAnalyzer(); // Reset motion tracking
    reset();
    start();
  }, [reset, start]);

  // Handle manual start of challenges
  const handleStartChallenges = useCallback(() => {
    if (qualityPassed) {
      console.log('%c[KYC] Starting challenges...', 'color: #2563eb; font-weight: bold');
      onQualityPassed();
    }
  }, [qualityPassed, onQualityPassed]);

  // Start camera and detection when user clicks begin
  const handleBegin = useCallback(async () => {
    console.log('%c[KYC] === VERIFICATION SESSION STARTED ===', 'color: #2563eb; font-weight: bold; font-size: 14px');
    resetMotionAnalyzer(); // Reset motion tracking for new session
    start();
    await startCamera();
    console.log('[KYC] Camera started, MediaPipe ready:', isMediaPipeReady);
  }, [start, startCamera, isMediaPipeReady]);

  // Initialize MediaPipe when ready and camera is streaming
  useEffect(() => {
    if (isMediaPipeReady && isStreaming && state.currentState === STATES.INITIALIZING) {
      onInitialized();
    }
  }, [isMediaPipeReady, isStreaming, state.currentState, onInitialized]);

  // Start/stop detection loop
  useEffect(() => {
    if (
      isStreaming &&
      isMediaPipeReady &&
      state.currentState !== STATES.IDLE &&
      state.currentState !== STATES.SUCCESS &&
      state.currentState !== STATES.FAILURE &&
      state.currentState !== STATES.FLAGGED &&
      state.currentState !== STATES.DISPUTE
    ) {
      runDetectionLoop();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isStreaming, isMediaPipeReady, state.currentState, runDetectionLoop]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [stopCamera]);

  // Get current challenge
  const currentChallenge = getCurrentChallenge();

  // Render error state
  if (cameraError || mediaPipeError) {
    return (
      <div className="max-w-md mx-auto p-6 text-center">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-red-100 flex items-center justify-center">
          <svg
            className="w-8 h-8 text-red-600"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-900 mb-2">
          Something went wrong
        </h2>
        <p className="text-gray-600 mb-6">{cameraError || mediaPipeError}</p>
        <button
          onClick={() => window.location.reload()}
          className="py-2 px-6 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800"
        >
          Refresh Page
        </button>
      </div>
    );
  }

  // Render idle state (start screen)
  if (state.currentState === STATES.IDLE) {
    return (
      <div className="max-w-md mx-auto p-6 text-center">
        <div className="mb-8">
          <h1 className="text-2xl font-semibold text-gray-900 mb-2">
            Identity Verification
          </h1>
          <p className="text-gray-600">
            We need to verify that you're a real person. This will take about 30 seconds.
          </p>
        </div>

        <div className="mb-8 p-4 bg-gray-50 rounded-lg text-left">
          <h3 className="font-medium text-gray-900 mb-3">Before you begin:</h3>
          <ul className="space-y-2 text-sm text-gray-600">
            <li className="flex items-start gap-2">
              <svg className="w-5 h-5 text-green-600 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Find a well-lit area
            </li>
            <li className="flex items-start gap-2">
              <svg className="w-5 h-5 text-green-600 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Remove glasses and hats
            </li>
            <li className="flex items-start gap-2">
              <svg className="w-5 h-5 text-green-600 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Face the camera directly
            </li>
          </ul>
        </div>

        <button
          onClick={handleBegin}
          disabled={isLoadingMediaPipe}
          className="w-full py-3 px-6 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
        >
          {isLoadingMediaPipe ? 'Loading...' : 'Begin Verification'}
        </button>
      </div>
    );
  }

  // Render result states
  if (
    state.currentState === STATES.SUCCESS ||
    state.currentState === STATES.FAILURE ||
    state.currentState === STATES.FLAGGED
  ) {
    return (
      <div className="max-w-md mx-auto p-6">
        <ResultScreen
          status={state.verificationResult?.status || state.currentState}
          message={state.verificationResult?.message}
          onRetry={handleRetry}
          onDispute={handleDispute}
          canDispute={state.verificationResult?.canDispute}
          disputeSubmitted={state.verificationResult?.disputeSubmitted}
          riskScore={state.verificationResult?.risk_score}
          riskLevel={state.verificationResult?.risk_level}
          layerResults={state.verificationResult?.layer_results}
        />
      </div>
    );
  }

  // Render main verification flow
  return (
    <div className="max-w-md mx-auto p-4">
      {/* Camera feed */}
      <CameraFeed
        ref={videoRef}
        isStreaming={isStreaming}
        onCanvasRef={setCanvasRef}
      />

      {/* Status area below camera */}
      <div className="mt-6">
        {/* Initializing */}
        {state.currentState === STATES.INITIALIZING && (
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-300 border-t-blue-600 mx-auto mb-3" />
            <p className="text-gray-600">Setting up...</p>
          </div>
        )}

        {/* Quality check with manual start button */}
        {state.currentState === STATES.QUALITY_CHECK && (
          <>
            <QualityGate qualityResults={qualityResults} isChecking={true} />
            <button
              onClick={handleStartChallenges}
              disabled={!qualityPassed}
              className={`w-full mt-4 py-3 px-6 rounded-lg font-medium transition-colors ${
                qualityPassed
                  ? 'bg-green-600 text-white hover:bg-green-700'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              {qualityPassed ? 'Start Challenges' : 'Waiting for all checks to pass...'}
            </button>
          </>
        )}

        {/* Challenges */}
        {isInChallengeState() && currentChallenge && (
          <Challenge
            challenge={currentChallenge}
            challengeNumber={state.currentChallengeIndex + 1}
            totalChallenges={state.challenges.length}
            holdProgress={state.holdFrameCount}
            onTimeout={() => {
              // Timeout handled by challenge component
            }}
          />
        )}

        {/* Capturing */}
        {state.currentState === STATES.CAPTURING && (
          <div className="text-center">
            <div className="animate-pulse">
              <div className="w-4 h-4 bg-red-500 rounded-full mx-auto mb-3" />
              <p className="text-gray-600">Capturing...</p>
            </div>
          </div>
        )}

        {/* Processing */}
        {state.currentState === STATES.PROCESSING && (
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-300 border-t-blue-600 mx-auto mb-3" />
            <p className="text-gray-600">Verifying your identity...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default LivenessCheck;
