const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Get dashboard stats
export async function getStats() {
  const response = await fetch(`${API_URL}/api/admin/stats`);
  if (!response.ok) throw new Error('Failed to fetch stats');
  return response.json();
}

// List pending reviews
export async function getReviews(status = 'pending') {
  const response = await fetch(`${API_URL}/api/admin/reviews?status=${status}`);
  if (!response.ok) throw new Error('Failed to fetch reviews');
  return response.json();
}

// Get single review details
export async function getReview(reviewId) {
  const response = await fetch(`${API_URL}/api/admin/reviews/${reviewId}`);
  if (!response.ok) throw new Error('Failed to fetch review');
  return response.json();
}

// Approve a review
export async function approveReview(reviewId, notes = '') {
  const response = await fetch(`${API_URL}/api/admin/reviews/${reviewId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes }),
  });
  if (!response.ok) throw new Error('Failed to approve review');
  return response.json();
}

// Reject a review
export async function rejectReview(reviewId, notes = '') {
  const response = await fetch(`${API_URL}/api/admin/reviews/${reviewId}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes }),
  });
  if (!response.ok) throw new Error('Failed to reject review');
  return response.json();
}

// List disputes
export async function getDisputes(status = 'pending') {
  const response = await fetch(`${API_URL}/api/admin/disputes?status=${status}`);
  if (!response.ok) throw new Error('Failed to fetch disputes');
  return response.json();
}
