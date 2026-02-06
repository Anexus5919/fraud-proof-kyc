import { useState, useEffect } from 'react';
import { getCustomers } from '../api/admin';

function Customers({ onNavigate }) {
  const [customers, setCustomers] = useState([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchCustomers() {
      try {
        setIsLoading(true);
        const data = await getCustomers('active');
        setCustomers(data.customers);
        setTotal(data.total);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    }
    fetchCustomers();
  }, []);

  const getRiskBadge = (score) => {
    if (score == null) return null;
    if (score <= 30)
      return <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-700">Low ({score})</span>;
    if (score <= 60)
      return <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700">Medium ({score})</span>;
    return <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700">High ({score})</span>;
  };

  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="text-red-600 mb-4">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-gray-900">Registered Customers</h1>
        <p className="text-gray-500">{total} total registered customers</p>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-gray-300 border-t-blue-600" />
        </div>
      ) : customers.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
          <p className="text-gray-500">No customers registered yet</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {customers.map((customer) => (
            <div
              key={customer.id}
              onClick={() => onNavigate?.('customer-detail', { customerId: customer.customer_id })}
              className="bg-white rounded-lg border border-gray-200 p-4 cursor-pointer hover:border-gray-400 hover:shadow-sm transition-all"
            >
              {/* Face image */}
              <div className="mb-3">
                {customer.face_image ? (
                  <img
                    src={customer.face_image}
                    alt="Face"
                    className="w-full aspect-square object-cover rounded-lg bg-gray-100"
                  />
                ) : (
                  <div className="w-full aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
                    <svg className="w-10 h-10 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                )}
              </div>

              {/* Info */}
              <p className="text-xs text-gray-500 font-mono truncate mb-1">
                {customer.customer_id.slice(0, 12)}...
              </p>
              <div className="flex items-center justify-between">
                <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                  customer.status === 'active' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'
                }`}>
                  {customer.status}
                </span>
                {getRiskBadge(customer.risk_score)}
              </div>
              <p className="text-xs text-gray-400 mt-1">
                {customer.created_at ? new Date(customer.created_at).toLocaleDateString() : ''}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Customers;
