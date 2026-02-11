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
      success: 'bg-emerald-50 text-emerald-700',
      pending_review: 'bg-amber-50 text-amber-700',
      rejected: 'bg-red-50 text-red-700',
      spoof_detected: 'bg-red-50 text-red-700',
      deepfake_detected: 'bg-red-50 text-red-700',
      duplicate_found: 'bg-amber-50 text-amber-700',
    };
    return styles[result] || 'bg-slate-100 text-slate-600';
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
      <div className="flex justify-center py-16">
        <div className="animate-spin rounded-full h-7 w-7 border-2 border-slate-200 border-t-slate-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="text-red-600 mb-4">Error: {error}</div>
        <button onClick={onBack} className="text-sm text-slate-500 hover:text-slate-700">Back to Customers</button>
      </div>
    );
  }

  if (!customer) return null;

  const riskColor = getRiskColor(customer.risk_score);

  // Extract detailed telemetry from the first audit entry (if available)
  const auditDetails = customer.audit_trail?.[0]?.details || {};

  return (
    <div className="p-6 lg:p-8 max-w-6xl mx-auto">
      {/* Back button */}
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-[13px] text-slate-400 hover:text-slate-600 mb-6 transition-colors"
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back to Customers
      </button>

      {/* Customer header */}
      <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm overflow-hidden mb-6">
        <div className="p-6 flex gap-6">
          <div className="flex-shrink-0">
            {customer.face_image ? (
              <img
                src={customer.face_image}
                alt="Face"
                className="w-36 h-36 object-cover rounded-lg bg-slate-100"
              />
            ) : (
              <div className="w-36 h-36 bg-slate-50 rounded-lg flex items-center justify-center">
                <svg className="w-14 h-14 text-slate-200" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2.5 mb-1.5">
              <h1 className="text-lg font-semibold text-slate-900">
                {customer.customer_name || 'Customer'}
              </h1>
              <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                customer.status === 'active' ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-100 text-slate-500'
              }`}>
                {customer.status}
              </span>
            </div>
            <p className="text-[13px] text-slate-400 font-mono mb-2">{customer.customer_id}</p>
            <p className="text-[13px] text-slate-500">
              Registered: {customer.created_at ? new Date(customer.created_at).toLocaleString() : 'N/A'}
            </p>
            {customer.challenges && (
              <p className="text-[13px] text-slate-500 mt-0.5">
                Challenges: {customer.challenges.join(', ')}
              </p>
            )}
            {customer.session_id && (
              <p className="text-[11px] text-slate-400 font-mono mt-1.5">
                Session: {customer.session_id}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Score cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4 relative overflow-hidden">
          <div className={`absolute top-0 left-0 w-full h-0.5 ${
            riskColor === 'green' ? 'bg-emerald-500' :
            riskColor === 'amber' ? 'bg-amber-500' :
            riskColor === 'red' ? 'bg-red-500' : 'bg-slate-300'
          }`} />
          <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">Risk Score</p>
          <p className={`text-2xl font-semibold tabular-nums ${
            riskColor === 'green' ? 'text-emerald-600' :
            riskColor === 'amber' ? 'text-amber-600' :
            riskColor === 'red' ? 'text-red-600' : 'text-slate-400'
          }`}>
            {customer.risk_score != null ? `${customer.risk_score}/100` : 'N/A'}
          </p>
          <p className="text-[11px] text-slate-400">
            {customer.risk_score != null ? (customer.risk_score <= 30 ? 'Low risk' : customer.risk_score <= 60 ? 'Medium risk' : 'High risk') : ''}
          </p>
        </div>

        <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4">
          <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">Spoof Score</p>
          <p className="text-2xl font-semibold text-slate-900 tabular-nums">
            {customer.spoof_score != null ? customer.spoof_score.toFixed(2) : 'N/A'}
          </p>
          <p className="text-[11px] text-slate-400">
            {customer.spoof_score != null ? (customer.spoof_score >= 0.55 ? 'Passed' : 'Failed') : ''}
          </p>
        </div>

        <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4">
          <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">Deepfake Score</p>
          <p className="text-2xl font-semibold text-slate-900 tabular-nums">
            {customer.deepfake_score != null ? customer.deepfake_score.toFixed(2) : 'N/A'}
          </p>
          <p className="text-[11px] text-slate-400">
            {customer.deepfake_score != null ? (customer.deepfake_score >= 0.20 ? 'Passed' : 'Failed') : ''}
          </p>
        </div>

        <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4">
          <p className="text-[11px] text-slate-400 uppercase tracking-wider mb-1">Pipeline Time</p>
          <p className="text-2xl font-semibold text-slate-900 tabular-nums">
            {auditDetails.pipeline_time_s != null ? `${auditDetails.pipeline_time_s}s` : 'N/A'}
          </p>
        </div>
      </div>

      {/* Detailed telemetry from audit log */}
      {Object.keys(auditDetails).length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Spoof detection details */}
          {auditDetails.spoof_details && (
            <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4">
              <h3 className="text-[13px] font-semibold text-slate-900 mb-3">Spoof Detection Details</h3>
              <div className="space-y-1.5">
                {Object.entries(auditDetails.spoof_details).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-slate-400">{key}</span>
                    <span className="text-slate-700 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
                {auditDetails.motion_penalty != null && (
                  <div className="flex justify-between text-xs pt-1.5 border-t border-slate-50">
                    <span className="text-slate-400">motion_penalty</span>
                    <span className="text-slate-700 font-mono">{auditDetails.motion_penalty}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Deepfake detection details */}
          {(auditDetails.deepfake_ml || auditDetails.deepfake_frequency) && (
            <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4">
              <h3 className="text-[13px] font-semibold text-slate-900 mb-3">Deepfake Detection Details</h3>
              <div className="space-y-1.5">
                {auditDetails.deepfake_method && (
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">method</span>
                    <span className="text-slate-700 font-mono text-right">{auditDetails.deepfake_method}</span>
                  </div>
                )}
                {auditDetails.deepfake_ml && Object.entries(auditDetails.deepfake_ml).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-slate-400">ml.{key}</span>
                    <span className="text-slate-700 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
                {auditDetails.deepfake_frequency && Object.entries(auditDetails.deepfake_frequency).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-slate-400">freq.{key}</span>
                    <span className="text-slate-700 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
                {auditDetails.deepfake_critical_flag && (
                  <div className="flex justify-between text-xs pt-1.5 border-t border-slate-50">
                    <span className="text-slate-400">critical_flag</span>
                    <span className="text-red-600 font-mono">{auditDetails.deepfake_critical_flag}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Motion analysis */}
          {auditDetails.motion_analysis && (
            <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4">
              <h3 className="text-[13px] font-semibold text-slate-900 mb-3">Motion Analysis</h3>
              <div className="space-y-1.5">
                {Object.entries(auditDetails.motion_analysis).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-slate-400">{key}</span>
                    <span className="text-slate-700 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Risk breakdown */}
          {auditDetails.risk_breakdown && (
            <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4">
              <h3 className="text-[13px] font-semibold text-slate-900 mb-3">Risk Score Breakdown</h3>
              <div className="space-y-1.5">
                {auditDetails.risk_factors && Object.entries(auditDetails.risk_factors).map(([key, value]) => (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-slate-400">{key}</span>
                    <span className="text-slate-700 font-mono">{formatValue(value)}</span>
                  </div>
                ))}
                <div className="pt-1.5 border-t border-slate-50">
                  {Object.entries(auditDetails.risk_breakdown).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-slate-400">{key}</span>
                      <span className="text-slate-700 font-mono text-right max-w-xs truncate">{formatValue(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Flags */}
          {auditDetails.flags && auditDetails.flags.length > 0 && (
            <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-4">
              <h3 className="text-[13px] font-semibold text-slate-900 mb-3">Flags</h3>
              <div className="flex flex-wrap gap-1.5">
                {auditDetails.flags.map((flag, i) => (
                  <span key={i} className="px-2 py-0.5 rounded bg-red-50 text-red-600 text-xs">{flag}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Audit trail */}
      <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-100">
          <h2 className="text-[15px] font-semibold text-slate-900">Audit Trail</h2>
        </div>
        {customer.audit_trail && customer.audit_trail.length > 0 ? (
          <div className="divide-y divide-slate-50">
            {customer.audit_trail.map((entry) => (
              <div key={entry.id} className="px-5 py-3.5">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${getResultBadge(entry.result)}`}>
                      {entry.result}
                    </span>
                    <span className="text-[13px] text-slate-600">{entry.action}</span>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    {entry.created_at ? new Date(entry.created_at).toLocaleString() : ''}
                  </span>
                </div>
                {entry.details && (
                  <details className="mt-2">
                    <summary className="text-[11px] text-blue-600 cursor-pointer hover:underline">
                      View full details ({Object.keys(entry.details).length} fields)
                    </summary>
                    <div className="mt-2 text-xs text-slate-500 font-mono bg-slate-50 rounded-lg p-3 overflow-x-auto max-h-96 overflow-y-auto">
                      {Object.entries(entry.details).map(([key, value]) => (
                        <div key={key} className="py-0.5">
                          <span className="text-slate-400">{key}:</span>{' '}
                          <span className="text-slate-600">
                            {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
                {entry.ip_address && (
                  <p className="text-[11px] text-slate-400 mt-1">IP: {entry.ip_address}</p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="px-5 py-10 text-center text-slate-300 text-sm">
            No audit entries found
          </div>
        )}
      </div>
    </div>
  );
}

export default CustomerDetail;
