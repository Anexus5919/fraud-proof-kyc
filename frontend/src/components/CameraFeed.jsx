import { forwardRef } from 'react';

const CameraFeed = forwardRef(function CameraFeed({ isStreaming, onCanvasRef }, ref) {
  return (
    <div className="relative w-full max-w-md mx-auto">
      {/* Outer frame with subtle shadow */}
      <div className="relative aspect-[4/3] bg-slate-900 rounded-2xl overflow-hidden shadow-lg ring-1 ring-slate-900/5">
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
            viewBox="0 0 400 300"
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Oval face guide — soft, barely-there */}
            <ellipse
              cx="200"
              cy="135"
              rx="85"
              ry="110"
              fill="none"
              stroke="rgba(255,255,255,0.22)"
              strokeWidth="1.2"
              strokeDasharray="6,4"
            />
          </svg>
        </div>

        {/* Subtle vignette effect */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.12) 100%)'
          }}
        />

        {/* Loading state when not streaming */}
        {!isStreaming && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
            <div className="text-slate-400 text-center">
              <div className="w-14 h-14 mx-auto mb-3 rounded-full border-2 border-slate-700 flex items-center justify-center">
                <svg
                  className="w-7 h-7 animate-pulse"
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
              </div>
              <p className="text-sm font-medium text-slate-500">Initializing camera...</p>
            </div>
          </div>
        )}

        {/* Live indicator when streaming */}
        {isStreaming && (
          <div className="absolute top-3 left-3 flex items-center gap-1.5 px-2.5 py-1 bg-black/40 backdrop-blur-sm rounded-full">
            <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-[10px] font-medium text-white/80 uppercase tracking-wider">Live</span>
          </div>
        )}
      </div>
    </div>
  );
});

export default CameraFeed;
