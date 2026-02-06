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
      success: 'bg-green-100 text-green-700',
      pending_review: 'bg-amber-100 text-amber-700',
      rejected: 'bg-red-100 text-red-700',
      spoof_detected: 'bg-red-100 text-red-700',
      deepfake_detected: 'bg-purple-100 text-purple-700',
      duplicate_found: 'bg-amber-100 text-amber-700',
    };
    return styles[result] || 'bg-gray-100 text-gray-600';
  };

  const getRiskBadge = (score) => {
    if (score == null) return null;
    if (score <= 30)
      return <span className="text-green-600 font-medium">{score}</span>;
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
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">Verification Log</h1>
        <p className="text-gray-500">{total} total verification attempts</p>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-6">
        {filters.map((f) => (
          <button
            key={f.label}
            onClick={() => setResultFilter(f.value)}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              resultFilter === f.value
                ? 'bg-gray-900 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-300 border-t-blue-600" />
        </div>
      ) : verifications.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
          <p className="text-gray-500">No verifications found</p>
        </div>
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Session</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Result</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Spoof</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Deepfake</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Flags</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">IP</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {verifications.map((v) => (
                <tr key={v.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {v.created_at ? new Date(v.created_at).toLocaleString() : ''}
                  </td>
                  <td className="px-4 py-3 text-xs font-mono text-gray-500">
                    {v.session_id?.slice(0, 12)}...
                  </td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${getResultBadge(v.result)}`}>
                      {v.result}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {getRiskBadge(v.risk_score) || <span className="text-gray-400">-</span>}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {v.spoof_score != null ? v.spoof_score.toFixed(2) : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">
                    {v.deepfake_score != null ? v.deepfake_score.toFixed(2) : '-'}
                  </td>
                  <td className="px-4 py-3">
                    {v.flags && v.flags.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {v.flags.map((flag, i) => (
                          <span key={i} className="px-1.5 py-0.5 rounded text-xs bg-red-50 text-red-600">
                            {flag}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="text-gray-400 text-sm">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-400 font-mono">
                    {v.ip_address || '-'}
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
