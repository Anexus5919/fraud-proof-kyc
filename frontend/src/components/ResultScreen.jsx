function ResultScreen({
  status,
  message,
  onRetry,
  onDispute,
  canDispute,
  disputeSubmitted,
  riskScore,
  riskLevel,
  layerResults,
}) {
  const isSuccess = status === 'success';
  const isPending = status === 'pending_review' || status === 'flagged' || status === 'duplicate_found';
  const isFailure = status === 'failure' || status === 'spoof_detected' || status === 'deepfake_detected' || status === 'rejected' || status === 'error';

  const getLayerIcon = (layer) => {
    if (!layer) return null;
    if (layer.status === 'passed') {
      return (
        <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      );
    } else if (layer.status === 'flagged') {
      return (
        <svg className="w-5 h-5 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      );
    } else {
      return (
        <svg className="w-5 h-5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      );
    }
  };

  return (
    <div className="text-center space-y-6 py-8">
      {/* Status icon */}
      <div className="flex justify-center">
        {isSuccess && (
          <div className="w-20 h-20 rounded-full bg-green-100 flex items-center justify-center">
            <svg className="w-10 h-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
        )}

        {isPending && (
          <div className="w-20 h-20 rounded-full bg-amber-100 flex items-center justify-center">
            <svg className="w-10 h-10 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
        )}

        {isFailure && (
          <div className="w-20 h-20 rounded-full bg-red-100 flex items-center justify-center">
            <svg className="w-10 h-10 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
        )}
      </div>

      {/* Status title */}
      <div>
        <h2
          className={`text-2xl font-semibold ${
            isSuccess ? 'text-green-800' : isPending ? 'text-amber-800' : 'text-red-800'
          }`}
        >
          {isSuccess && 'Verification Successful'}
          {isPending && 'Under Review'}
          {isFailure && 'Verification Failed'}
        </h2>
      </div>

      {/* Risk Score Display */}
      {riskScore !== undefined && riskScore !== null && (
        <div className="max-w-sm mx-auto">
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600">Risk Score</span>
              <span className={`text-2xl font-bold ${
                riskScore <= 30 ? 'text-green-600' :
                riskScore <= 60 ? 'text-amber-600' : 'text-red-600'
              }`}>
                {riskScore}/100
              </span>
            </div>

            {/* Risk bar */}
            <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  riskScore <= 30 ? 'bg-green-500' :
                  riskScore <= 60 ? 'bg-amber-500' : 'bg-red-500'
                }`}
                style={{ width: `${riskScore}%` }}
              />
            </div>

            <div className="flex justify-between mt-2 text-xs text-gray-500">
              <span>Low (0-30)</span>
              <span>Medium (31-60)</span>
              <span>High (61-100)</span>
            </div>

            <div className={`mt-3 text-sm font-medium ${
              riskScore <= 30 ? 'text-green-700' :
              riskScore <= 60 ? 'text-amber-700' : 'text-red-700'
            }`}>
              {riskScore <= 30 && 'Auto Approved'}
              {riskScore > 30 && riskScore <= 60 && 'Manual Review Required'}
              {riskScore > 60 && 'Auto Rejected'}
            </div>
          </div>
        </div>
      )}

      {/* 5-Layer Results */}
      {layerResults && Object.keys(layerResults).length > 0 && (
        <div className="max-w-sm mx-auto">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 text-left">Verification Layers</h3>
          <div className="bg-gray-50 rounded-lg divide-y divide-gray-200">
            {layerResults.layer1_face_detection && (
              <div className="flex items-center justify-between p-3">
                <div className="flex items-center gap-2">
                  {getLayerIcon(layerResults.layer1_face_detection)}
                  <span className="text-sm text-gray-700">Face Detection</span>
                </div>
                <span className="text-xs text-gray-500">
                  {layerResults.layer1_face_detection.confidence
                    ? `${(layerResults.layer1_face_detection.confidence * 100).toFixed(0)}%`
                    : layerResults.layer1_face_detection.status}
                </span>
              </div>
            )}

            {layerResults.layer2_liveness && (
              <div className="flex items-center justify-between p-3">
                <div className="flex items-center gap-2">
                  {getLayerIcon(layerResults.layer2_liveness)}
                  <span className="text-sm text-gray-700">Liveness Check</span>
                </div>
                <span className="text-xs text-gray-500">
                  {layerResults.layer2_liveness.score
                    ? `${(layerResults.layer2_liveness.score * 100).toFixed(0)}%`
                    : layerResults.layer2_liveness.status}
                </span>
              </div>
            )}

            {layerResults.layer3_deepfake && (
              <div className="flex items-center justify-between p-3">
                <div className="flex items-center gap-2">
                  {getLayerIcon(layerResults.layer3_deepfake)}
                  <span className="text-sm text-gray-700">Deepfake Check</span>
                </div>
                <span className="text-xs text-gray-500">
                  {layerResults.layer3_deepfake.score
                    ? `${(layerResults.layer3_deepfake.score * 100).toFixed(0)}%`
                    : layerResults.layer3_deepfake.status}
                </span>
              </div>
            )}

            {layerResults.layer4_duplicate && (
              <div className="flex items-center justify-between p-3">
                <div className="flex items-center gap-2">
                  {getLayerIcon(layerResults.layer4_duplicate)}
                  <span className="text-sm text-gray-700">Duplicate Check</span>
                </div>
                <span className="text-xs text-gray-500">
                  {layerResults.layer4_duplicate.matches_found > 0
                    ? `${layerResults.layer4_duplicate.matches_found} found`
                    : 'Unique'}
                </span>
              </div>
            )}

            {layerResults.layer5_risk_score && (
              <div className="flex items-center justify-between p-3">
                <div className="flex items-center gap-2">
                  <svg className={`w-5 h-5 ${
                    layerResults.layer5_risk_score.score <= 30 ? 'text-green-500' :
                    layerResults.layer5_risk_score.score <= 60 ? 'text-amber-500' : 'text-red-500'
                  }`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  <span className="text-sm text-gray-700">Risk Assessment</span>
                </div>
                <span className={`text-xs font-medium ${
                  layerResults.layer5_risk_score.level === 'low' ? 'text-green-600' :
                  layerResults.layer5_risk_score.level === 'medium' ? 'text-amber-600' : 'text-red-600'
                }`}>
                  {layerResults.layer5_risk_score.level?.toUpperCase()}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Message */}
      <p className="text-gray-600 max-w-sm mx-auto">
        {message ||
          (isSuccess
            ? 'Your identity has been verified successfully.'
            : isPending
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
        {(isFailure || isPending) && !disputeSubmitted && (
          <button
            onClick={onRetry}
            className="w-full py-3 px-6 bg-gray-900 text-white rounded-lg font-medium hover:bg-gray-800 transition-colors"
          >
            Try Again
          </button>
        )}

        {isFailure && canDispute && !disputeSubmitted && (
          <button
            onClick={onDispute}
            className="w-full py-3 px-6 bg-white text-gray-700 border border-gray-300 rounded-lg font-medium hover:bg-gray-50 transition-colors"
          >
            I'm really here
          </button>
        )}

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
