import LivenessCheck from './components/LivenessCheck';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <h1 className="text-lg font-semibold text-gray-900">
            KYC Verification
          </h1>
        </div>
      </header>

      {/* Main content */}
      <main className="py-8">
        <LivenessCheck />
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 py-3">
        <p className="text-center text-xs text-gray-500">
          Your data is processed securely and never shared
        </p>
      </footer>
    </div>
  );
}

export default App;
