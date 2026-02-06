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
      {/* Progress indicator */}
      <div className="flex justify-center gap-2">
        {Array.from({ length: totalChallenges }).map((_, idx) => (
          <div
            key={idx}
            className={`w-3 h-3 rounded-full transition-colors duration-300 ${
              idx < challengeNumber - 1
                ? 'bg-green-500'
                : idx === challengeNumber - 1
                ? 'bg-blue-500'
                : 'bg-gray-300'
            }`}
          />
        ))}
      </div>

      {/* Challenge info */}
      <div className="text-center">
        <p className="text-sm text-gray-500 mb-1">
          Challenge {challengeNumber} of {totalChallenges}
        </p>
        <h2 className="text-2xl font-semibold text-gray-900">
          {challenge.instruction}
        </h2>
      </div>

      {/* Timer */}
      <div className="flex justify-center">
        <div
          className={`text-4xl font-mono ${
            timeLeft <= 3 ? 'text-red-600' : 'text-gray-700'
          }`}
        >
          {timeLeft}s
        </div>
      </div>

      {/* Hold progress */}
      <div className="max-w-xs mx-auto">
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-green-500 transition-all duration-100"
            style={{ width: `${holdPercentage}%` }}
          />
        </div>
        {holdPercentage > 0 && (
          <p className="text-center text-sm text-green-600 mt-2">
            Hold it...
          </p>
        )}
      </div>

      {/* Hint */}
      <p className="text-center text-sm text-gray-500">
        Perform the action and hold for a moment
      </p>
    </div>
  );
}

export default Challenge;
