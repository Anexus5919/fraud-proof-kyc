function QualityGate({ qualityResults, isChecking }) {
  if (!qualityResults) {
    return (
      <div className="text-center py-4">
        <p className="text-slate-500 text-sm">Checking image quality...</p>
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
        className={`text-center px-4 py-2.5 rounded-xl ${
          visibleAllPassed
            ? 'bg-emerald-50 text-emerald-800 ring-1 ring-emerald-200/60'
            : 'bg-amber-50 text-amber-800 ring-1 ring-amber-200/60'
        }`}
      >
        <p className="font-medium text-sm">
          {visibleAllPassed ? 'All checks passed' : primaryMessage}
        </p>
      </div>

      {/* Check list — compact for side panel, hidden checks excluded */}
      <div className="space-y-1">
        {visibleChecks.map((check) => (
          <div
            key={check.name}
            className={`flex items-center gap-2.5 px-3 py-1.5 rounded-lg transition-colors duration-150 ${
              check.passed ? 'text-emerald-700' : 'text-amber-700'
            }`}
          >
            {check.passed ? (
              <div className="w-5 h-5 rounded-full bg-emerald-100 flex items-center justify-center shrink-0">
                <svg className="w-3 h-3 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
              </div>
            ) : (
              <div className="w-5 h-5 rounded-full bg-amber-100 flex items-center justify-center shrink-0">
                <svg className="w-3 h-3 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 9v2m0 4h.01" />
                </svg>
              </div>
            )}
            <span className="text-[13px]">{check.message}</span>
          </div>
        ))}
      </div>

      {/* Loading indicator */}
      {isChecking && !allPassed && (
        <div className="flex justify-center pt-1">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-slate-200 border-t-blue-500" />
        </div>
      )}
    </div>
  );
}

export default QualityGate;
