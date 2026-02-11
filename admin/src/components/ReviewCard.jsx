import { useState } from 'react';
import { approveReview, rejectReview } from '../api/admin';

function ReviewCard({ review, onAction }) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [notes, setNotes] = useState('');
  const [showNotes, setShowNotes] = useState(false);

  const handleApprove = async () => {
    setIsProcessing(true);
    try {
      await approveReview(review.id, notes);
      onAction?.();
    } catch (error) {
      console.error('Failed to approve:', error);
      alert('Failed to approve review');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReject = async () => {
    setIsProcessing(true);
    try {
      await rejectReview(review.id, notes);
      onAction?.();
    } catch (error) {
      console.error('Failed to reject:', error);
      alert('Failed to reject review');
    } finally {
      setIsProcessing(false);
    }
  };

  const similarityPercent = Math.round(review.similarity_score * 100);
  const isPending = review.status === 'pending';

  return (
    <div className="bg-white rounded-xl ring-1 ring-slate-200/60 shadow-sm p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-slate-400">
          {review.created_at ? new Date(review.created_at).toLocaleString() : ''}
        </span>
        <div className="flex items-center gap-2">
          {review.risk_score != null && (
            <span className={`px-2 py-0.5 rounded-md text-xs font-medium ${
              review.risk_score <= 30 ? 'bg-emerald-50 text-emerald-700' :
              review.risk_score <= 60 ? 'bg-amber-50 text-amber-700' :
              'bg-red-50 text-red-700'
            }`}>
              Risk: {review.risk_score}
            </span>
          )}
          <span
            className={`px-2 py-0.5 rounded-md text-xs font-medium ${
              similarityPercent > 80
                ? 'bg-red-50 text-red-700'
                : similarityPercent > 60
                ? 'bg-amber-50 text-amber-700'
                : 'bg-yellow-50 text-yellow-700'
            }`}
          >
            {similarityPercent}% match
          </span>
        </div>
      </div>

      {/* Telemetry scores row */}
      {(review.spoof_score != null || review.deepfake_score != null) && (
        <div className="flex gap-4 mb-4 text-xs">
          {review.spoof_score != null && (
            <div className="flex items-center gap-1">
              <span className="text-slate-400">Spoof:</span>
              <span className={review.spoof_score >= 0.55 ? 'text-emerald-600 font-medium' : 'text-red-600 font-medium'}>
                {review.spoof_score.toFixed(2)}
              </span>
            </div>
          )}
          {review.deepfake_score != null && (
            <div className="flex items-center gap-1">
              <span className="text-slate-400">Deepfake:</span>
              <span className={review.deepfake_score >= 0.4 ? 'text-emerald-600 font-medium' : 'text-red-600 font-medium'}>
                {review.deepfake_score.toFixed(2)}
              </span>
            </div>
          )}
          {review.flags && review.flags.length > 0 && (
            <div className="flex gap-1 flex-wrap">
              {review.flags.map((flag, i) => (
                <span key={i} className="px-1.5 py-0.5 rounded-md bg-red-50 text-red-600">{flag}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Face comparison */}
      <div className="grid grid-cols-2 gap-5 mb-6">
        {/* New registration */}
        <div>
          <p className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">New Registration</p>
          {review.new_face_image_url ? (
            <img
              src={review.new_face_image_url}
              alt="New face"
              className="w-full aspect-square object-cover rounded-xl bg-slate-100"
            />
          ) : (
            <div className="w-full aspect-square bg-slate-100 rounded-xl flex items-center justify-center">
              <span className="text-slate-400 text-sm">No image</span>
            </div>
          )}
          <p className="mt-2 text-xs text-slate-500 font-mono">
            {review.new_customer_id?.slice(0, 12)}...
          </p>
        </div>

        {/* Matched existing */}
        <div>
          <p className="text-xs font-semibold text-slate-500 mb-2 uppercase tracking-wider">Existing Match</p>
          {review.matched_face_image_url ? (
            <img
              src={review.matched_face_image_url}
              alt="Matched face"
              className="w-full aspect-square object-cover rounded-xl bg-slate-100"
            />
          ) : (
            <div className="w-full aspect-square bg-slate-100 rounded-xl flex items-center justify-center">
              <span className="text-slate-400 text-sm">No image</span>
            </div>
          )}
          <p className="mt-2 text-xs text-slate-500">
            {review.matched_customer_name || <span className="font-mono">{review.matched_customer_id?.slice(0, 12)}...</span>}
          </p>
        </div>
      </div>

      {/* Status badge for processed reviews */}
      {!isPending && (
        <div className={`mb-4 px-4 py-2.5 rounded-xl text-sm ${
          review.status === 'approved' ? 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200/60' : 'bg-red-50 text-red-700 ring-1 ring-red-200/60'
        }`}>
          {review.status === 'approved' ? 'Approved' : 'Rejected'} by {review.reviewed_by || 'admin'}
          {review.reviewed_at && ` on ${new Date(review.reviewed_at).toLocaleString()}`}
          {review.review_notes && <p className="mt-1 text-xs opacity-75">{review.review_notes}</p>}
        </div>
      )}

      {/* Notes input */}
      {isPending && showNotes && (
        <div className="mb-4">
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Add notes (optional)"
            className="w-full p-3 ring-1 ring-slate-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
            rows={2}
          />
        </div>
      )}

      {/* Actions - only for pending reviews */}
      {isPending && (
        <>
          <div className="flex gap-3">
            <button
              onClick={() => {
                if (showNotes) handleApprove();
                else setShowNotes(true);
              }}
              disabled={isProcessing}
              className="flex-1 py-2.5 px-4 bg-emerald-600 text-white rounded-xl font-medium hover:bg-emerald-700 disabled:bg-slate-300 transition-colors shadow-sm"
            >
              {isProcessing ? 'Processing...' : showNotes ? 'Confirm: Different People' : 'Different People'}
            </button>
            <button
              onClick={() => {
                if (showNotes) handleReject();
                else setShowNotes(true);
              }}
              disabled={isProcessing}
              className="flex-1 py-2.5 px-4 bg-red-600 text-white rounded-xl font-medium hover:bg-red-700 disabled:bg-slate-300 transition-colors shadow-sm"
            >
              {isProcessing ? 'Processing...' : showNotes ? 'Confirm: Same Person' : 'Same Person'}
            </button>
          </div>

          {showNotes && (
            <button
              onClick={() => setShowNotes(false)}
              className="w-full mt-2 py-2 text-sm text-slate-400 hover:text-slate-600 transition-colors"
            >
              Cancel
            </button>
          )}
        </>
      )}
    </div>
  );
}

export default ReviewCard;
