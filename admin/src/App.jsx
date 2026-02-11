import { useState } from 'react';
import Dashboard from './pages/Dashboard';
import Customers from './pages/Customers';
import CustomerDetail from './pages/CustomerDetail';
import Verifications from './pages/Verifications';

function App() {
  const [page, setPage] = useState('dashboard');
  const [selectedCustomerId, setSelectedCustomerId] = useState(null);

  const navigateTo = (newPage, data) => {
    setPage(newPage);
    if (newPage === 'customer-detail' && data?.customerId) {
      setSelectedCustomerId(data.customerId);
    }
  };

  const navItems = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'customers', label: 'Customers' },
    { id: 'verifications', label: 'Verifications' },
  ];

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-slate-200/60 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <div
              className="flex items-center gap-2.5 cursor-pointer"
              onClick={() => navigateTo('dashboard')}
            >
              <div className="w-9 h-9 bg-slate-900 rounded-xl flex items-center justify-center shadow-sm">
                <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <div>
                <span className="text-[15px] font-semibold text-slate-900 leading-tight block">KYC Admin</span>
                <span className="text-[10px] text-slate-400 leading-tight">Management Console</span>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex gap-1">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => navigateTo(item.id)}
                  className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                    page === item.id || (item.id === 'customers' && page === 'customer-detail')
                      ? 'bg-slate-900 text-white shadow-sm'
                      : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main>
        {page === 'dashboard' && <Dashboard onNavigate={navigateTo} />}
        {page === 'customers' && <Customers onNavigate={navigateTo} />}
        {page === 'customer-detail' && (
          <CustomerDetail
            customerId={selectedCustomerId}
            onBack={() => navigateTo('customers')}
          />
        )}
        {page === 'verifications' && <Verifications />}
      </main>
    </div>
  );
}

export default App;
