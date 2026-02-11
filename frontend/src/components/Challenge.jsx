import { useState, useEffect } from 'react';

function Challenge({
  challenge,
  challengeNumber,
  totalChallenges,
  holdProgress,
  onTimeout,
}) {
  const [timeLeft, setTimeLeft] = useState(challenge.timeLimit / 1000);

  // Countdown timer
  useEffect(() => {
    if (timeLeft <= 0) {
      onTimeout?.();
      return;
    }

    const timer = setInterval(() => {
      setTimeLeft((prev) => Math.max(0, prev - 1));
    }, 1000);

    return () => clearInterval(timer);
  }, [timeLeft, onTimeout]);

  // Reset timer when challenge changes
  useEffect(() => {
    setTimeLeft(challenge.timeLimit / 1000);
  }, [challenge.id, challenge.timeLimit]);

  // Calculate hold progress percentage
  const holdPercentage = Math.min(100, (holdProgress / challenge.holdFrames) * 100);

  return (
    <div className="space-y-6">
      {/* Step indicators */}
      <div className="flex items-center justify-center gap-2">
        {Array.from({ length: totalChallenges }).map((_, idx) => (
          <div key={idx} className="flex items-center">
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold transition-all duration-300 ${
                idx < challengeNumber - 1
                  ? 'bg-emerald-500 text-white shadow-sm'
                  : idx === challengeNumber - 1
                  ? 'bg-slate-900 text-white shadow-sm'
                  : 'bg-slate-200 text-slate-400'
              }`}
            >
              {idx < challengeNumber - 1 ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                idx + 1
              )}
            </div>
            {idx < totalChallenges - 1 && (
              <div className={`w-6 h-0.5 mx-1 transition-colors duration-300 ${
                idx < challengeNumber - 1 ? 'bg-emerald-400' : 'bg-slate-200'
              }`} />
            )}
          </div>
        ))}
      </div>

      {/* Challenge info */}
      <div className="text-center">
        <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">
          Challenge {challengeNumber} of {totalChallenges}
        </p>
        <h2 className="text-2xl font-semibold text-slate-900">
          {challenge.instruction}
        </h2>
      </div>

      {/* Timer */}
      <div className="flex justify-center">
        <div
          className={`inline-flex items-center justify-center w-16 h-16 rounded-2xl font-mono text-2xl font-bold transition-colors ${
            timeLeft <= 3
              ? 'bg-red-50 text-red-600 ring-1 ring-red-200/60'
              : 'bg-slate-100 text-slate-700 ring-1 ring-slate-200/60'
          }`}
        >
          {timeLeft}
        </div>
      </div>

      {/* Hold progress */}
      <div className="max-w-xs mx-auto">
        <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 rounded-full transition-all duration-100"
            style={{ width: `${holdPercentage}%` }}
          />
        </div>
        {holdPercentage > 0 && (
          <p className="text-center text-sm font-medium text-emerald-600 mt-2">
            Hold it...
          </p>
        )}
      </div>

      {/* Hint */}
      <p className="text-center text-sm text-slate-400">
        Perform the action and hold for a moment
      </p>
    </div>
  );
}

export default Challenge;
