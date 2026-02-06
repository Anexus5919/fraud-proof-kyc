import { useEffect, useRef, useCallback, useState } from 'react';
import { useCamera } from '../hooks/useCamera';
import { useFaceLandmarker } from '../hooks/useFaceLandmarker';
import { useLivenessState, STATES } from '../hooks/useLivenessState';
import { runAllQualityChecks, runCaptureQualityChecks } from '../lib/qualityChecks';
import { getMotionAnalyzer, resetMotionAnalyzer } from '../lib/motionAnalyzer';
import { verifyFace, submitDispute, canvasToBase64 } from '../api/verify';
import CameraFeed from './CameraFeed';
import QualityGate from './QualityGate';
import Challenge from './Challenge';
import ResultScreen from './ResultScreen';

// Fixed paragraph for the user to read aloud during quality check (~15 seconds).
// Reading forces natural mouth movement — printed photos can't do this.
const READING_PARAGRAPH =
  'I hereby confirm that I am completing this identity verification in person, using my own face, ' +
  'and not on behalf of any other individual. I understand that this process is designed to protect ' +
  'my account from unauthorized access and fraud. I acknowledge that submitting false identity ' +
  'information or using photographs, videos, masks, or any other deceptive means is strictly ' +
  'prohibited and may be considered a punishable offence under applicable laws.';

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
    onCaptureConfirmed,
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
  const [captureQualityResults, setCaptureQualityResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false); // Async state for UI rendering only
  const [qualityPassed, setQualityPassed] = useState(false); // Tracks if user can start challenges
  const [captureReady, setCaptureReady] = useState(false); // Tracks if capture quality checks pass
  const [evaluationProgress, setEvaluationProgress] = useState(0); // 0-100% over 10 seconds
  const evaluationStartRef = useRef(null); // When quality check evaluation began
  const blinkCountRef = useRef(0); // Cumulative blink count during quality check
  const wasBlinkingRef = useRef(false); // Debounce: was the previous frame a blink?

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
      // Quality check phase - continuous checking with 10-second evaluation timer
      // Timer RESETS if any check fails — must pass continuously for 10 seconds
      if (state.currentState === STATES.QUALITY_CHECK) {
        // Feed landmarks to motion analyzer for anti-spoofing temporal analysis
        const landmarks = results.faceLandmarks?.[0];
        if (landmarks) {
          const motionAnalyzer = getMotionAnalyzer();
          motionAnalyzer.addFrame(landmarks);
        }

        // Track blinks via blendshapes (printed photos can't blink)
        const blendshapes = results.faceBlendshapes?.[0]?.categories;
        if (blendshapes) {
          const eyeBlinkL = blendshapes.find(s => s.categoryName === 'eyeBlinkLeft')?.score || 0;
          const eyeBlinkR = blendshapes.find(s => s.categoryName === 'eyeBlinkRight')?.score || 0;
          const isBlinking = eyeBlinkL > 0.5 && eyeBlinkR > 0.5;
          // Count rising edge (open → closed transition) as one blink
          if (isBlinking && !wasBlinkingRef.current) {
            blinkCountRef.current++;
          }
          wasBlinkingRef.current = isBlinking;
        }

        const quality = runAllQualityChecks(
          results,
          videoRef.current,
          canvasRef.current,
          getMotionAnalyzer(),
          blinkCountRef.current
        );
        setQualityResults(quality);

        // Split checks: visible checks drive timer, hidden blink accumulates during timer
        // This avoids circular dependency: blink needs time to reach 2, can't trigger its own reset
        const visibleFailing = quality.checks.some(c => !c.passed && !c.hidden);

        if (visibleFailing) {
          // A visible check failed — reset timer AND blink count
          // This catches blink-then-switch: switch to paper → motion fails → everything resets
          evaluationStartRef.current = null;
          setEvaluationProgress(0);
          setQualityPassed(false);
          blinkCountRef.current = 0;
          wasBlinkingRef.current = false;
        } else {
          // All visible checks passing — run the 10-second timer
          // Blink count accumulates naturally during this window
          if (!evaluationStartRef.current) {
            evaluationStartRef.current = Date.now();
          }
          const elapsed = (Date.now() - evaluationStartRef.current) / 1000;
          const progress = Math.min((elapsed / 10) * 100, 100);
          setEvaluationProgress(progress);

          // Final gate: timer complete AND all checks pass (including hidden blink)
          const nowPassed = progress >= 100 && quality.allPassed;
          if (nowPassed && !qualityPassed) {
            console.log('%c[KYC] All quality checks PASSED for 10s — evaluation complete', 'color: #16a34a; font-weight: bold');
          }
          setQualityPassed(nowPassed);
        }
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

      // Ready-to-capture phase — run stricter quality checks (frontal face required)
      if (state.currentState === STATES.READY_TO_CAPTURE) {
        const captureQuality = runCaptureQualityChecks(
          results,
          videoRef.current,
          canvasRef.current
        );
        setCaptureQualityResults(captureQuality);
        setCaptureReady(captureQuality.allPassed);
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

    // Security: verify challenges were actually completed
    if (state.completedChallenges.length < 3) {
      console.error('[KYC] SECURITY: Capture attempted without completing all challenges!');
      onVerifyFailure({
        status: 'error',
        message: 'Verification integrity error. Please restart.',
        canDispute: false,
      });
      isProcessingRef.current = false;
      setIsProcessing(false);
      return;
    }

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

  // Handle spacebar press for capture confirmation
  const handleCaptureConfirm = useCallback(() => {
    if (captureReady && state.currentState === STATES.READY_TO_CAPTURE) {
      console.log('%c[KYC] Capture confirmed by user (spacebar)', 'color: #2563eb; font-weight: bold');
      onCaptureConfirmed();
    }
  }, [captureReady, state.currentState, onCaptureConfirmed]);

  // Listen for spacebar during READY_TO_CAPTURE state
  useEffect(() => {
    if (state.currentState !== STATES.READY_TO_CAPTURE) return;

    const handleKeyDown = (e) => {
      if (e.code === 'Space' || e.key === ' ') {
        e.preventDefault();
        handleCaptureConfirm();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state.currentState, handleCaptureConfirm]);

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
    setCaptureQualityResults(null);
    setCaptureReady(false);
    setEvaluationProgress(0);
    evaluationStartRef.current = null;
    blinkCountRef.current = 0;
    wasBlinkingRef.current = false;
    resetMotionAnalyzer(); // Reset motion tracking
    reset();
    start();
  }, [reset, start]);

  // Handle manual start of challenges (requires evaluation complete + all checks passing)
  const handleStartChallenges = useCallback(() => {
    if (qualityPassed && evaluationProgress >= 100) {
      console.log('%c[KYC] Starting challenges...', 'color: #2563eb; font-weight: bold');
      onQualityPassed();
    }
  }, [qualityPassed, evaluationProgress, onQualityPassed]);

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

  // Render main verification flow — horizontal layout (camera left, info right)
  const evalSeconds = Math.min(Math.floor((evaluationProgress / 100) * 10), 10);

  // Spoof suspected: timer complete, all visible checks pass, but hidden blink check fails
  const visibleAllPass = qualityResults?.checks?.filter(c => !c.hidden).every(c => c.passed) ?? false;
  const spoofSuspected = evaluationProgress >= 100 && visibleAllPass && !qualityPassed;

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Left column: Camera feed + evaluation progress bar */}
        <div className="w-full lg:w-[55%]">
          <CameraFeed
            ref={videoRef}
            isStreaming={isStreaming}
            onCanvasRef={setCanvasRef}
          />

          {/* Evaluation progress bar — only during quality check */}
          {state.currentState === STATES.QUALITY_CHECK && (
            <div className="mt-4">
              <div className="flex items-center justify-between text-xs text-gray-500 mb-1.5">
                <span className="font-medium">
                  {evaluationProgress >= 100
                    ? qualityPassed ? 'Evaluation complete' : 'Evaluation done — checks failing'
                    : 'Evaluating face liveness'}
                </span>
                <span>{evalSeconds}s / 10s</span>
              </div>
              <div className="h-2.5 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ease-linear ${
                    evaluationProgress >= 100
                      ? qualityPassed ? 'bg-green-500' : 'bg-amber-500'
                      : 'bg-blue-500'
                  }`}
                  style={{ width: `${evaluationProgress}%` }}
                />
              </div>
              <p className="text-xs text-gray-400 mt-1.5">
                {evaluationProgress < 100
                  ? 'Hold still and face the camera — analyzing motion patterns...'
                  : qualityPassed
                  ? 'All checks passed — you may proceed'
                  : 'Please fix the failing checks above'}
              </p>
            </div>
          )}
        </div>

        {/* Right column: Status, quality checks, challenges, controls */}
        <div className="w-full lg:w-[45%]">
          {/* Initializing */}
          {state.currentState === STATES.INITIALIZING && (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-300 border-t-blue-600 mx-auto mb-3" />
              <p className="text-gray-600">Setting up...</p>
            </div>
          )}

          {/* Quality check with evaluation-gated start button */}
          {state.currentState === STATES.QUALITY_CHECK && (
            <>
              <QualityGate qualityResults={qualityResults} isChecking={true} />

              {/* Spoof alert — shown when hidden blink check fails after timer completes */}
              {spoofSuspected && (
                <div className="mt-3 flex items-center gap-2.5 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <svg className="w-5 h-5 text-red-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <span className="text-sm font-medium text-red-800 flex-1">Spoof attempt detected</span>
                  <div className="relative group">
                    <svg className="w-4.5 h-4.5 text-red-400 cursor-help" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div className="absolute bottom-full right-0 mb-2 px-3 py-1.5 bg-gray-900 text-white text-xs rounded-lg whitespace-nowrap opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 pointer-events-none">
                      Remove spectacles and try again
                      <div className="absolute top-full right-2 w-2 h-2 bg-gray-900 rotate-45 -translate-y-1" />
                    </div>
                  </div>
                </div>
              )}

              <button
                onClick={handleStartChallenges}
                disabled={!qualityPassed || evaluationProgress < 100}
                className={`w-full mt-4 py-3 px-6 rounded-lg font-medium transition-colors ${
                  qualityPassed && evaluationProgress >= 100
                    ? 'bg-green-600 text-white hover:bg-green-700'
                    : spoofSuspected
                    ? 'bg-red-200 text-red-500 cursor-not-allowed'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                {evaluationProgress < 100
                  ? `Evaluating... ${evalSeconds}s / 10s`
                  : qualityPassed
                  ? 'Start Challenges'
                  : spoofSuspected
                  ? 'Verification blocked'
                  : 'Waiting for all checks to pass...'}
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
              onTimeout={() => {}}
            />
          )}

          {/* Ready to capture — user must look at camera and press spacebar */}
          {state.currentState === STATES.READY_TO_CAPTURE && (
            <div className="text-center">
              <div className="mb-4">
                <div className={`w-16 h-16 mx-auto mb-3 rounded-full flex items-center justify-center ${
                  captureReady ? 'bg-green-100' : 'bg-amber-100'
                }`}>
                  <svg className={`w-8 h-8 ${captureReady ? 'text-green-600' : 'text-amber-600'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-1">
                  Challenges Complete
                </h3>
                <p className="text-sm text-gray-600 mb-4">
                  Now look directly at the camera for your verification photo.
                </p>
              </div>

              {/* Capture quality checks */}
              {captureQualityResults && (
                <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                  <div className="space-y-1.5">
                    {captureQualityResults.checks.map((check) => (
                      <div key={check.name} className="flex items-center gap-2 text-xs">
                        {check.passed ? (
                          <svg className="w-4 h-4 text-green-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                        ) : (
                          <svg className="w-4 h-4 text-amber-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01" />
                          </svg>
                        )}
                        <span className={check.passed ? 'text-gray-600' : 'text-amber-700'}>
                          {check.message}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Capture button + spacebar hint */}
              <button
                onClick={handleCaptureConfirm}
                disabled={!captureReady}
                className={`w-full py-3 px-6 rounded-lg font-medium transition-colors ${
                  captureReady
                    ? 'bg-gray-900 text-white hover:bg-gray-800'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                {captureReady ? 'Capture Photo' : 'Please look directly at the camera'}
              </button>
              {captureReady && (
                <p className="text-xs text-gray-400 mt-2">
                  or press <kbd className="px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded text-gray-600 font-mono">Space</kbd> to capture
                </p>
              )}
            </div>
          )}

          {/* Capturing */}
          {state.currentState === STATES.CAPTURING && (
            <div className="text-center py-12">
              <div className="animate-pulse">
                <div className="w-4 h-4 bg-red-500 rounded-full mx-auto mb-3" />
                <p className="text-gray-600">Capturing...</p>
              </div>
            </div>
          )}

          {/* Processing */}
          {state.currentState === STATES.PROCESSING && (
            <div className="text-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-300 border-t-blue-600 mx-auto mb-3" />
              <p className="text-gray-600">Verifying your identity...</p>
            </div>
          )}
        </div>
      </div>

      {/* Reading prompt — full width below columns, only during quality check */}
      {state.currentState === STATES.QUALITY_CHECK && (
        <div className="mt-5 p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <div>
              <p className="text-xs font-medium text-gray-500 mb-1">Please read the following aloud</p>
              <p className="text-sm text-gray-800 leading-relaxed italic">"{READING_PARAGRAPH}"</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default LivenessCheck;
