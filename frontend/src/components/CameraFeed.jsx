import { forwardRef } from 'react';

const CameraFeed = forwardRef(function CameraFeed({ isStreaming, onCanvasRef }, ref) {
  return (
    <div className="relative w-full max-w-md mx-auto aspect-[4/3] bg-gray-900 rounded-lg overflow-hidden">
      {/* Video element */}
      <video
        ref={ref}
        className="absolute inset-0 w-full h-full object-cover"
        style={{ transform: 'scaleX(-1)' }} // Mirror for selfie view
        autoPlay
        playsInline
        muted
      />

      {/* Canvas overlay for drawing landmarks */}
      <canvas
        ref={onCanvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ transform: 'scaleX(-1)' }}
      />

      {/* Face guide overlay */}
      <div className="absolute inset-0 pointer-events-none">
        <svg
          className="w-full h-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          {/* Oval face guide */}
          <ellipse
            cx="50"
            cy="45"
            rx="25"
            ry="32"
            fill="none"
            stroke="rgba(255,255,255,0.3)"
            strokeWidth="0.5"
            strokeDasharray="2,2"
          />
        </svg>
      </div>

      {/* Loading state when not streaming */}
      {!isStreaming && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <div className="text-gray-400 text-center">
            <svg
              className="w-12 h-12 mx-auto mb-2 animate-pulse"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
              />
            </svg>
            <p className="text-sm">Initializing camera...</p>
          </div>
        </div>
      )}
    </div>
  );
});

export default CameraFeed;
