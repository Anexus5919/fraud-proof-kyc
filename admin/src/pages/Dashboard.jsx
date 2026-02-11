import { useState, useEffect } from 'react';
import { getStats, getReviews } from '../api/admin';
import StatsCard from '../components/StatsCard';
import ReviewCard from '../components/ReviewCard';

function Dashboard({ onNavigate }) {
  const [stats, setStats] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statusFilter, setStatusFilter] = useState('pending');

  const fetchData = async () => {
    try {
      setIsLoading(true);
      const [statsData, reviewsData] = await Promise.all([
        getStats(),
        getReviews(statusFilter),
      ]);
      setStats(statsData);
      setReviews(reviewsData.reviews);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [statusFilter]);

  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="text-red-600 mb-4">Error: {error}</div>
        <button
          onClick={fetchData}
          className="px-4 py-2 bg-slate-900 text-white rounded-lg text-sm font-medium"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-slate-900">Dashboard</h1>
        <p className="text-slate-400 text-sm mt-0.5">Overview of KYC verification activity</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
        <div onClick={() => onNavigate?.('customers')} className="cursor-pointer">
          <StatsCard
            title="Total Customers"
            value={stats?.total_customers || 0}
            color="blue"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            }
          />
        </div>
        <StatsCard
          title="Pending Reviews"
          value={stats?.pending_reviews || 0}
          color="amber"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          }
        />
        <StatsCard
          title="Pending Disputes"
          value={stats?.pending_disputes || 0}
          color="red"
          icon={
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          }
        />
        <div onClick={() => onNavigate?.('verifications')} className="cursor-pointer">
          <StatsCard
            title="Today's Verifications"
            value={stats?.today_verifications || 0}
            color="green"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            }
          />
        </div>
      </div>

      {/* Reviews section */}
      <div>
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-semibold text-slate-900">Duplicate Reviews</h2>
          <div className="flex gap-1 bg-slate-100 p-0.5 rounded-lg">
            {['pending', 'approved', 'rejected'].map((status) => (
              <button
                key={status}
                onClick={() => setStatusFilter(status)}
                className={`px-3 py-1 rounded-md text-[13px] font-medium transition-colors ${
                  statusFilter === status
                    ? 'bg-white text-slate-900 shadow-sm'
                    : 'text-slate-500 hover:text-slate-700'
                }`}
              >
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-16">
            <div className="animate-spin rounded-full h-7 w-7 border-2 border-slate-200 border-t-slate-600" />
          </div>
        ) : reviews.length === 0 ? (
          <div className="text-center py-16 bg-white rounded-xl ring-1 ring-slate-200/60">
            <svg
              className="w-10 h-10 mx-auto text-slate-300 mb-3"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-sm text-slate-400">No {statusFilter} reviews</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {reviews.map((review) => (
              <ReviewCard
                key={review.id}
                review={review}
                onAction={fetchData}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
