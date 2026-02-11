import { useState, useEffect } from 'react';
import { getVerifications } from '../api/admin';

function Verifications() {
  const [verifications, setVerifications] = useState([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resultFilter, setResultFilter] = useState(null);

  useEffect(() => {
    async function fetchVerifications() {
      try {
        setIsLoading(true);
        const data = await getVerifications(resultFilter);
        setVerifications(data.verifications);
        setTotal(data.total);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    }
    fetchVerifications();
  }, [resultFilter]);

  const getResultBadge = (result) => {
    const styles = {
      success: 'bg-emerald-50 text-emerald-700',
      pending_review: 'bg-amber-50 text-amber-700',
      rejected: 'bg-red-50 text-red-700',
      spoof_detected: 'bg-red-50 text-red-700',
      deepfake_detected: 'bg-purple-50 text-purple-700',
      duplicate_found: 'bg-amber-50 text-amber-700',
    };
    return styles[result] || 'bg-slate-100 text-slate-600';
  };

  const getRiskBadge = (score) => {
    if (score == null) return null;
    if (score <= 30)
      return <span className="text-emerald-600 font-medium">{score}</span>;
    if (score <= 60)
      return <span className="text-amber-600 font-medium">{score}</span>;
    return <span className="text-red-600 font-medium">{score}</span>;
  };

  const filters = [
    { label: 'All', value: null },
    { label: 'Success', value: 'success' },
    { label: 'Pending Review', value: 'pending_review' },
    { label: 'Duplicate', value: 'duplicate_found' },
    { label: 'Spoof', value: 'spoof_detected' },
    { label: 'Rejected', value: 'rejected' },
  ];

  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="text-red-600 mb-4">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto">
      <div className="flex items-baseline justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold text-slate-900">Verification Log</h1>
          <p className="text-slate-400 text-sm mt-0.5">{total} total verification attempts</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-1 bg-slate-100 p-0.5 rounded-lg w-fit mb-6">
        {filters.map((f) => (
          <button
            key={f.label}
            onClick={() => setResultFilter(f.value)}
            className={`px-3 py-1 rounded-md text-[13px] font-medium transition-colors ${
              resultFilter === f.value
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="flex justify-center py-16">
          <div className="animate-spin rounded-full h-7 w-7 border-2 border-slate-200 border-t-slate-600" />
        </div>
      ) : verifications.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-xl ring-1 ring-slate-200/60">
          <p className="text-sm text-slate-400">No verifications found</p>
        </div>
      ) : (
        <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="px-4 py-3 text-left text-[11px] font-medium text-slate-400 uppercase tracking-wider">Time</th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-slate-400 uppercase tracking-wider">Session</th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-slate-400 uppercase tracking-wider">Result</th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-slate-400 uppercase tracking-wider">Risk</th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-slate-400 uppercase tracking-wider">Spoof</th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-slate-400 uppercase tracking-wider">Deepfake</th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-slate-400 uppercase tracking-wider">Flags</th>
                <th className="px-4 py-3 text-left text-[11px] font-medium text-slate-400 uppercase tracking-wider">IP</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-50">
              {verifications.map((v) => (
                <tr key={v.id} className="hover:bg-slate-50/60 transition-colors">
                  <td className="px-4 py-3 text-[13px] text-slate-600">
                    {v.created_at ? new Date(v.created_at).toLocaleString() : ''}
                  </td>
                  <td className="px-4 py-3 text-xs font-mono text-slate-400">
                    {v.session_id?.slice(0, 12)}...
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-[11px] font-medium ${getResultBadge(v.result)}`}>
                      {v.result}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-[13px]">
                    {getRiskBadge(v.risk_score) || <span className="text-slate-300">-</span>}
                  </td>
                  <td className="px-4 py-3 text-[13px] text-slate-500 font-mono">
                    {v.spoof_score != null ? v.spoof_score.toFixed(2) : <span className="text-slate-300">-</span>}
                  </td>
                  <td className="px-4 py-3 text-[13px] text-slate-500 font-mono">
                    {v.deepfake_score != null ? v.deepfake_score.toFixed(2) : <span className="text-slate-300">-</span>}
                  </td>
                  <td className="px-4 py-3">
                    {v.flags && v.flags.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {v.flags.map((flag, i) => (
                          <span key={i} className="px-1.5 py-0.5 rounded text-[11px] bg-red-50 text-red-600">
                            {flag}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-slate-300 text-[13px]">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-400 font-mono">
                    {v.ip_address || <span className="text-slate-300">-</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default Verifications;
