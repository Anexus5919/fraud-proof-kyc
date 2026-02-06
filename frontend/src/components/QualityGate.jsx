function QualityGate({ qualityResults, isChecking }) {
  if (!qualityResults) {
    return (
      <div className="text-center py-4">
        <p className="text-gray-600">Checking image quality...</p>
      </div>
    );
  }

  const { allPassed, checks, primaryMessage } = qualityResults;

  return (
    <div className="space-y-4">
      {/* Main message */}
      <div
        className={`text-center p-4 rounded-lg ${
          allPassed
            ? 'bg-green-50 text-green-800'
            : 'bg-amber-50 text-amber-800'
        }`}
      >
        <p className="font-medium">{primaryMessage}</p>
      </div>

      {/* Check list */}
      <div className="space-y-2">
        {checks.map((check) => (
          <div
            key={check.name}
            className={`flex items-center gap-3 p-2 rounded ${
              check.passed ? 'text-green-700' : 'text-amber-700'
            }`}
          >
            {/* Status icon */}
            {check.passed ? (
              <svg
                className="w-5 h-5 text-green-600 shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            ) : (
              <svg
                className="w-5 h-5 text-amber-600 shrink-0"
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
            )}

            {/* Check message */}
            <span className="text-sm">{check.message}</span>
          </div>
        ))}
      </div>

      {/* Continue hint */}
      {allPassed && (
        <p className="text-center text-sm text-gray-500">
          Quality checks passed. Starting challenges...
        </p>
      )}

      {/* Loading indicator */}
      {isChecking && (
        <div className="flex justify-center">
          <div className="animate-spin rounded-full h-5 w-5 border-2 border-gray-300 border-t-blue-600" />
        </div>
      )}
    </div>
  );
}

export default QualityGate;
