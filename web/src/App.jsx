import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import SplashScreen from './components/SplashScreen';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import VerifyEmailPage from './pages/VerifyEmailPage';
import AnalyzePage from './pages/AnalyzePage';
import HistoryPage from './pages/HistoryPage';
import ModelPage from './pages/ModelPage';

function App() {
  const [showSplash, setShowSplash] = useState(true);

  if (showSplash) {
    return <SplashScreen onFinish={() => setShowSplash(false)} />;
  }

  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route element={<Layout />}>
            {/* Public Routes */}
            <Route path="/" element={<LandingPage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/verify-email" element={<VerifyEmailPage />} />
            <Route path="/model" element={<ModelPage />} />

            {/* Protected Clinical Routes (Auth & Email Verification Required) */}
            <Route element={<ProtectedRoute requireVerified={true} />}>
              <Route path="/history" element={<HistoryPage />} />
            </Route>

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
