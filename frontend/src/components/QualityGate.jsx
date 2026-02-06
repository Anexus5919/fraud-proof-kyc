function QualityGate({ qualityResults, isChecking }) {
  if (!qualityResults) {
    return (
      <div className="text-center py-4">
        <p className="text-gray-600">Checking image quality...</p>
      </div>
    );
  }

  const { allPassed, checks, primaryMessage } = qualityResults;
  const visibleChecks = checks.filter((c) => !c.hidden);
  const visibleAllPassed = visibleChecks.every((c) => c.passed);

  return (
    <div className="space-y-3">
      {/* Main message — only show visible check status */}
      <div
        className={`text-center p-3 rounded-lg ${
          visibleAllPassed
            ? 'bg-green-50 text-green-800'
            : 'bg-amber-50 text-amber-800'
        }`}
      >
        <p className="font-medium text-sm">
          {visibleAllPassed ? 'All checks passed' : primaryMessage}
        </p>
      </div>

      {/* Check list — compact for side panel, hidden checks excluded */}
      <div className="space-y-1.5">
        {visibleChecks.map((check) => (
          <div
            key={check.name}
            className={`flex items-center gap-2 px-2 py-1 rounded ${
              check.passed ? 'text-green-700' : 'text-amber-700'
            }`}
          >
            {check.passed ? (
              <svg className="w-4 h-4 text-green-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <svg className="w-4 h-4 text-amber-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            )}
            <span className="text-xs">{check.message}</span>
          </div>
        ))}
      </div>

      {/* Loading indicator */}
      {isChecking && !allPassed && (
        <div className="flex justify-center">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-gray-300 border-t-blue-600" />
        </div>
      )}
    </div>
  );
}

export default QualityGate;
