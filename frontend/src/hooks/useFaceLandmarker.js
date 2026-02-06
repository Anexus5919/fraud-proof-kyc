import { useState, useEffect, useRef, useCallback } from 'react';

const MEDIAPIPE_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export function useFaceLandmarker() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isReady, setIsReady] = useState(false);

  const faceLandmarkerRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  // Initialize MediaPipe
  useEffect(() => {
    let isMounted = true;

    async function initMediaPipe() {
      try {
        setIsLoading(true);
        setError(null);

        // Dynamically import MediaPipe
        const vision = await import(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest'
        );

        const { FaceLandmarker, FilesetResolver } = vision;

        // Load WASM files
        const filesetResolver = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);

        // Create face landmarker
        const faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: 'GPU',
          },
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: false,
          runningMode: 'VIDEO',
          numFaces: 5, // Detect up to 5 faces to catch multiple people in frame
        });

        if (isMounted) {
          faceLandmarkerRef.current = faceLandmarker;
          setIsReady(true);
          setIsLoading(false);
        }
      } catch (err) {
        console.error('Failed to initialize MediaPipe:', err);
        if (isMounted) {
          setError(err.message || 'Failed to load face detection model');
          setIsLoading(false);
        }
      }
    }

    initMediaPipe();

    return () => {
      isMounted = false;
      if (faceLandmarkerRef.current) {
        faceLandmarkerRef.current.close();
      }
    };
  }, []);

  // Detect faces in video frame
  const detectForVideo = useCallback((videoElement, timestamp) => {
    if (!faceLandmarkerRef.current || !videoElement) {
      return null;
    }

    // Only process if video time has changed
    if (videoElement.currentTime === lastVideoTimeRef.current) {
      return null;
    }

    lastVideoTimeRef.current = videoElement.currentTime;

    try {
      const results = faceLandmarkerRef.current.detectForVideo(
        videoElement,
        timestamp || performance.now()
      );
      return results;
    } catch (err) {
      console.error('Detection error:', err);
      return null;
    }
  }, []);

  return {
    isLoading,
    isReady,
    error,
    detectForVideo,
  };
}
