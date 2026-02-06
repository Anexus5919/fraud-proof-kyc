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

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-gray-500">
          {new Date(review.created_at).toLocaleString()}
        </span>
        <span
          className={`px-2 py-1 rounded text-xs font-medium ${
            similarityPercent > 80
              ? 'bg-red-100 text-red-700'
              : similarityPercent > 60
              ? 'bg-amber-100 text-amber-700'
              : 'bg-yellow-100 text-yellow-700'
          }`}
        >
          {similarityPercent}% match
        </span>
      </div>

      {/* Face comparison */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* New registration */}
        <div>
          <p className="text-sm font-medium text-gray-700 mb-2">New Registration</p>
          {review.new_face_image_url ? (
            <img
              src={review.new_face_image_url}
              alt="New face"
              className="w-full aspect-square object-cover rounded-lg bg-gray-100"
            />
          ) : (
            <div className="w-full aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
              <span className="text-gray-400">No image</span>
            </div>
          )}
          <p className="mt-2 text-sm text-gray-600">
            ID: {review.new_customer_id?.slice(0, 8)}...
          </p>
        </div>

        {/* Matched existing */}
        <div>
          <p className="text-sm font-medium text-gray-700 mb-2">Existing Match</p>
          {review.matched_face_image_url ? (
            <img
              src={review.matched_face_image_url}
              alt="Matched face"
              className="w-full aspect-square object-cover rounded-lg bg-gray-100"
            />
          ) : (
            <div className="w-full aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
              <span className="text-gray-400">No image</span>
            </div>
          )}
          <p className="mt-2 text-sm text-gray-600">
            {review.matched_customer_name || `ID: ${review.matched_customer_id?.slice(0, 8)}...`}
          </p>
        </div>
      </div>

      {/* Notes input */}
      {showNotes && (
        <div className="mb-4">
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Add notes (optional)"
            className="w-full p-3 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            rows={2}
          />
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={() => {
            if (showNotes) handleApprove();
            else setShowNotes(true);
          }}
          disabled={isProcessing}
          className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:bg-gray-400 transition-colors"
        >
          {isProcessing ? 'Processing...' : showNotes ? 'Confirm Approve' : 'Different People'}
        </button>
        <button
          onClick={() => {
            if (showNotes) handleReject();
            else setShowNotes(true);
          }}
          disabled={isProcessing}
          className="flex-1 py-2 px-4 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 disabled:bg-gray-400 transition-colors"
        >
          {isProcessing ? 'Processing...' : showNotes ? 'Confirm Reject' : 'Same Person'}
        </button>
      </div>

      {showNotes && (
        <button
          onClick={() => setShowNotes(false)}
          className="w-full mt-2 py-2 text-sm text-gray-500 hover:text-gray-700"
        >
          Cancel
        </button>
      )}
    </div>
  );
}

export default ReviewCard;
