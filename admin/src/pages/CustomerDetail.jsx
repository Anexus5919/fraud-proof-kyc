import { useState, useEffect } from 'react';
import { getCustomer } from '../api/admin';

function CustomerDetail({ customerId, onBack }) {
  const [customer, setCustomer] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchCustomer() {
      if (!customerId) return;
      try {
        setIsLoading(true);
        const data = await getCustomer(customerId);
        setCustomer(data);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    }
    fetchCustomer();
  }, [customerId]);

  const getRiskColor = (score) => {
    if (score == null) return 'gray';
    if (score <= 30) return 'green';
    if (score <= 60) return 'amber';
    return 'red';
  };

  const getResultBadge = (result) => {
    const styles = {
      success: 'bg-green-100 text-green-700',
      pending_review: 'bg-amber-100 text-amber-700',
      rejected: 'bg-red-100 text-red-700',
      spoof_detected: 'bg-red-100 text-red-700',
      deepfake_detected: 'bg-red-100 text-red-700',
      duplicate_found: 'bg-amber-100 text-amber-700',
    };
    return styles[result] || 'bg-gray-100 text-gray-600';
  };

  const formatValue = (value) => {
    if (value === null || value === undefined) return '-';
    if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(4);
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    if (Array.isArray(value)) return value.length ? value.join(', ') : '-';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-300 border-t-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="text-red-600 mb-4">Error: {error}</div>
        <button onClick={onBack} className="text-blue-600 hover:underline">Back to Customers</button>
      </div>
    );
  }

  if (!customer) return null;

  const riskColor = getRiskColor(customer.risk_score);

  // Extract detailed telemetry from the first audit entry (if available)
  const auditDetails = customer.audit_trail?.[0]?.details || {};

  return (
    <div className="p-6 max-w-6xl mx-auto">
      {/* Back button */}
      <button
        onClick={onBack}
        className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-6"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to Customers
      </button>

      {/* Customer header */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <div className="flex gap-6">
          <div className="flex-shrink-0">
            {customer.face_image ? (
              <img
                src={customer.face_image}
                alt="Face"
                className="w-40 h-40 object-cover rounded-lg bg-gray-100"
              />
            ) : (
              <div className="w-40 h-40 bg-gray-100 rounded-lg flex items-center justify-center">
                <svg className="w-16 h-16 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
            )}
          </div>

          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-xl font-semibold text-gray-900">
                {customer.customer_name || 'Customer'}
              </h1>
              <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                customer.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
              }`}>
                {customer.status}
              </span>
            </div>
            <p className="text-sm text-gray-500 font-mono mb-2">{customer.customer_id}</p>
            <p className="text-sm text-gray-500">
              Registered: {customer.created_at ? new Date(customer.created_at).toLocaleString() : 'N/A'}
            </p>
            {customer.challenges && (
              <p className="text-sm text-gray-500 mt-1">
                Challenges: {customer.challenges.join(', ')}
              </p>
            )}
            {customer.session_id && (
              <p className="text-xs text-gray-400 font-mono mt-1">
                Session: {customer.session_id}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Score cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Risk Score</p>
          <p className={`text-2xl font-bold ${
            riskColor === 'green' ? 'text-green-600' :
            riskColor === 'amber' ? 'text-amber-600' :
            riskColor === 'red' ? 'text-red-600' : 'text-gray-400'
          }`}>
            {customer.risk_score != null ? `${customer.risk_score}/100` : 'N/A'}
          </p>
          <p className="text-xs text-gray-400">
            {customer.risk_score != null ? (customer.risk_score <= 30 ? 'Low risk' : customer.risk_score <= 60 ? 'Medium risk' : 'High risk') : ''}
          </p>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Spoof Score</p>
          <p className="text-2xl font-bold text-gray-900">
            {customer.spoof_score != null ? customer.spoof_score.toFixed(2) : 'N/A'}
          </p>
          <p className="text-xs text-gray-400">
            {customer.spoof_score != null ? (customer.spoof_score >= 0.55 ? 'Passed' : 'Failed') : ''}
          </p>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Deepfake Score</p>
          <p className="text-2xl font-bold text-gray-900">
            {customer.deepfake_score != null ? customer.deepfake_score.toFixed(2) : 'N/A'}
          </p>
          <p className="text-xs text-gray-400">
            {customer.deepfake_score != null ? (customer.deepfake_score >= 0.20 ? 'Passed' : 'Failed') : ''}
          </p>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Pipeline Time</p>
          <p className="text-2xl font-bold text-gray-900">
            {auditDetails.pipeline_time_s != null ? `${auditDetails.pipeline_time_s}s` : 'N/A'}
          </p>
        </div>
      </div>

      {/* Detailed telemetry from audit log */}
      {Object.keys(auditDetails).length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Spoof detection details */}
          {auditDetails.spoof_details && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Spoof Detection Details</h3>
              <div className="space-y-1.5">
                {Object.entries(auditDetails.spoof_details).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-gray-500">{key}</span>
                    <span className="text-gray-900 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
                {auditDetails.motion_penalty != null && (
                  <div className="flex justify-between text-xs pt-1 border-t border-gray-100">
                    <span className="text-gray-500">motion_penalty</span>
                    <span className="text-gray-900 font-mono">{auditDetails.motion_penalty}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Deepfake detection details */}
          {(auditDetails.deepfake_ml || auditDetails.deepfake_frequency) && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Deepfake Detection Details</h3>
              <div className="space-y-1.5">
                {auditDetails.deepfake_method && (
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">method</span>
                    <span className="text-gray-900 font-mono text-right">{auditDetails.deepfake_method}</span>
                  </div>
                )}
                {auditDetails.deepfake_ml && Object.entries(auditDetails.deepfake_ml).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-gray-500">ml.{key}</span>
                    <span className="text-gray-900 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
                {auditDetails.deepfake_frequency && Object.entries(auditDetails.deepfake_frequency).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-gray-500">freq.{key}</span>
                    <span className="text-gray-900 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
                {auditDetails.deepfake_critical_flag && (
                  <div className="flex justify-between text-xs pt-1 border-t border-gray-100">
                    <span className="text-gray-500">critical_flag</span>
                    <span className="text-red-600 font-mono">{auditDetails.deepfake_critical_flag}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Motion analysis */}
          {auditDetails.motion_analysis && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Motion Analysis</h3>
              <div className="space-y-1.5">
                {Object.entries(auditDetails.motion_analysis).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-gray-500">{key}</span>
                    <span className="text-gray-900 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Risk breakdown */}
          {auditDetails.risk_breakdown && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Risk Score Breakdown</h3>
              <div className="space-y-1.5">
                {auditDetails.risk_factors && Object.entries(auditDetails.risk_factors).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-gray-500">{key}</span>
                    <span className="text-gray-900 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
                <div className="pt-1 border-t border-gray-100">
                  {Object.entries(auditDetails.risk_breakdown).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-gray-500">{key}</span>
                      <span className="text-gray-900 font-mono text-right max-w-xs truncate">{formatValue(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Flags */}
          {auditDetails.flags && auditDetails.flags.length > 0 && (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Flags</h3>
              <div className="flex flex-wrap gap-2">
                {auditDetails.flags.map((flag, i) => (
                  <span key={i} className="px-2 py-1 rounded bg-red-50 text-red-600 text-xs">{flag}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Audit trail */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Audit Trail</h2>
        </div>
        {customer.audit_trail && customer.audit_trail.length > 0 ? (
          <div className="divide-y divide-gray-100">
            {customer.audit_trail.map((entry) => (
              <div key={entry.id} className="px-6 py-4">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${getResultBadge(entry.result)}`}>
                      {entry.result}
                    </span>
                    <span className="text-sm text-gray-600">{entry.action}</span>
                  </div>
                  <span className="text-xs text-gray-400">
                    {entry.created_at ? new Date(entry.created_at).toLocaleString() : ''}
                  </span>
                </div>
                {entry.details && (
                  <details className="mt-2">
                    <summary className="text-xs text-blue-600 cursor-pointer hover:underline">
                      View full details ({Object.keys(entry.details).length} fields)
                    </summary>
                    <div className="mt-2 text-xs text-gray-500 font-mono bg-gray-50 rounded p-3 overflow-x-auto max-h-96 overflow-y-auto">
                      {Object.entries(entry.details).map(([key, value]) => (
                        <div key={key} className="py-0.5">
                          <span className="text-gray-400">{key}:</span>{' '}
                          <span className="text-gray-700">
                            {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
                {entry.ip_address && (
                  <p className="text-xs text-gray-400 mt-1">IP: {entry.ip_address}</p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="px-6 py-8 text-center text-gray-400">
            No audit entries found
          </div>
        )}
      </div>
    </div>
  );
}

export default CustomerDetail;
