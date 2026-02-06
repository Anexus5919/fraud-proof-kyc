function ResultScreen({
  status,
  message,
  onRetry,
  onDispute,
  canDispute,
  disputeSubmitted,
}) {
  const isSuccess = status === 'success';
  const isFlagged = status === 'flagged';
  const isFailure = status === 'failure' || status === 'spoof_detected';

  return (
    <div className="text-center space-y-6 py-8">
      {/* Status icon */}
      <div className="flex justify-center">
        {isSuccess && (
          <div className="w-20 h-20 rounded-full bg-green-100 flex items-center justify-center">
            <svg
              className="w-10 h-10 text-green-600"
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
          </div>
        )}

        {isFlagged && (
          <div className="w-20 h-20 rounded-full bg-amber-100 flex items-center justify-center">
            <svg
              className="w-10 h-10 text-amber-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
        )}

        {isFailure && (
          <div className="w-20 h-20 rounded-full bg-red-100 flex items-center justify-center">
            <svg
              className="w-10 h-10 text-red-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </div>
        )}
      </div>

      {/* Status title */}
      <div>
        <h2
          className={`text-2xl font-semibold ${
            isSuccess
              ? 'text-green-800'
              : isFlagged
              ? 'text-amber-800'
              : 'text-red-800'
          }`}
        >
          {isSuccess && 'Verification Successful'}
          {isFlagged && 'Under Review'}
          {isFailure && 'Verification Failed'}
        </h2>
      </div>

      {/* Message */}
      <p className="text-gray-600 max-w-sm mx-auto">
        {message ||
          (isSuccess
            ? 'Your identity has been verified successfully.'
            : isFlagged
            ? 'Your registration is being reviewed. We will contact you shortly.'
            : 'We could not verify your identity. Please try again.')}
      </p>

      {/* Dispute submitted notice */}
      {disputeSubmitted && (
        <div className="bg-blue-50 text-blue-800 p-4 rounded-lg max-w-sm mx-auto">
          <p className="text-sm">
            Your dispute has been submitted. Our team will review it and contact you.
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex flex-col gap-3 max-w-xs mx-auto">
        {/* Retry button */}
        {(isFailure || isFlagged) && !disputeSubmitted && (
          <button
            onClick={onRetry}
            className="w-full py-3 px-6 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition-colors"
          >
            Try Again
          </button>
        )}

        {/* Dispute button */}
        {isFailure && canDispute && !disputeSubmitted && (
          <button
            onClick={onDispute}
            className="w-full py-3 px-6 bg-white text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition-colors"
          >
            I'm really here
          </button>
        )}

        {/* Success done button */}
        {isSuccess && (
          <button
            onClick={() => window.location.reload()}
            className="w-full py-3 px-6 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors"
          >
            Done
          </button>
        )}
      </div>
    </div>
  );
}

export default ResultScreen;
