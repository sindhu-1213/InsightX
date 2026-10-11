import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth, getFriendlyAuthErrorMessage } from '../context/AuthContext';
import { ShieldCheck, Mail, Lock, AlertCircle, CheckCircle, RefreshCw } from 'lucide-react';

function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [error, setError] = useState(null);

  // Forgot password state
  const [showForgotModal, setShowForgotModal] = useState(false);
  const [resetEmail, setResetEmail] = useState('');
  const [resetStatus, setResetStatus] = useState(null);
  const [resetError, setResetError] = useState(null);

  const { login, signInWithGoogle, resetPassword } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    document.title = 'InsightX | Practitioner Sign In';
  }, []);

  const from = location.state?.from?.pathname || '/analyze';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const user = await login(email, password);
      if (user.emailVerified) {
        navigate(from, { replace: true });
      } else {
        navigate('/verify-email', { replace: true });
      }
    } catch (err) {
      setError(getFriendlyAuthErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setError(null);
    setGoogleLoading(true);
    try {
      await signInWithGoogle();
      navigate(from, { replace: true });
    } catch (err) {
      setError(getFriendlyAuthErrorMessage(err));
    } finally {
      setGoogleLoading(false);
    }
  };

  const handlePasswordReset = async (e) => {
    e.preventDefault();
    setResetError(null);
    setResetStatus('sending');
    try {
      await resetPassword(resetEmail);
      setResetStatus('sent');
    } catch (err) {
      setResetStatus(null);
      setResetError(getFriendlyAuthErrorMessage(err));
    }
  };

  return (
    <div
      id="login-page-container"
      style={{
        minHeight: 'calc(100vh - 160px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2.5rem 1.25rem'
      }}
    >
      <div
        className="card"
        style={{
          width: '100%',
          maxWidth: '440px',
          padding: '2.5rem 2rem',
          border: '1px solid rgba(0, 212, 255, 0.25)',
          boxShadow: '0 12px 40px rgba(0, 0, 0, 0.6)'
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div
            style={{
              width: '50px',
              height: '50px',
              borderRadius: '12px',
              backgroundColor: 'rgba(0, 212, 255, 0.1)',
              border: '1px solid var(--cyan)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1rem auto',
              color: '#00d4ff'
            }}
          >
            <ShieldCheck size={26} />
          </div>
          <h2
            id="login-heading"
            style={{
              fontSize: '1.6rem',
              fontFamily: 'var(--font-heading)',
              color: '#ffffff',
              marginBottom: '0.4rem'
            }}
          >
            Practitioner Access
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Sign in to access chest radiograph analysis
          </p>
        </div>

        {error && (
          <div
            style={{
              padding: '0.75rem 1rem',
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid var(--danger)',
              borderRadius: 'var(--radius-sm)',
              marginBottom: '1.25rem',
              fontSize: '0.85rem',
              color: '#ffffff',
              display: 'flex',
              gap: '0.5rem',
              alignItems: 'center'
            }}
          >
            <AlertCircle size={18} color="#ef4444" style={{ flexShrink: 0 }} />
            <span>{error}</span>
          </div>
        )}

        {/* Google Sign In Button */}
        <button
          type="button"
          id="google-signin-btn"
          onClick={handleGoogleSignIn}
          disabled={googleLoading}
          style={{
            width: '100%',
            padding: '0.75rem',
            backgroundColor: '#ffffff',
            color: '#1f2937',
            border: 'none',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.925rem',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.75rem',
            marginBottom: '1.25rem',
            transition: 'background-color 0.2s ease'
          }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#f3f4f6')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#ffffff')}
        >
          <svg width="18" height="18" viewBox="0 0 24 24">
            <path
              fill="#4285F4"
              d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.8-2.4 3.65v3.05h3.9c2.28-2.1 3.64-5.2 3.64-9.14z"
            />
            <path
              fill="#34A853"
              d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.9-3.05c-1.08.72-2.45 1.16-4.03 1.16-3.1 0-5.74-2.1-6.68-4.92H1.21v3.15C3.26 21.48 7.34 24 12 24z"
            />
            <path
              fill="#FBBC05"
              d="M5.32 14.28c-.24-.72-.38-1.49-.38-2.28s.14-1.56.38-2.28V6.57H1.21C.44 8.1 0 9.99 0 12s.44 3.9 1.21 5.43l4.11-3.15z"
            />
            <path
              fill="#EA4335"
              d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.34 0 3.26 2.52 1.21 6.57l4.11 3.15c.94-2.82 3.58-4.97 6.68-4.97z"
            />
          </svg>
          {googleLoading ? 'Connecting to Google...' : 'Sign in with Google'}
        </button>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            margin: '1rem 0 1.25rem 0',
            color: 'var(--text-muted)',
            fontSize: '0.8rem'
          }}
        >
          <div style={{ flex: 1, height: '1px', backgroundColor: 'var(--border)' }} />
          <span>OR SIGN IN WITH EMAIL</span>
          <div style={{ flex: 1, height: '1px', backgroundColor: 'var(--border)' }} />
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <label className="form-label" htmlFor="login-email">Clinical Email</label>
            <input
              id="login-email"
              type="email"
              className="form-input"
              placeholder="practitioner@clinic.org"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label className="form-label" htmlFor="login-password">Password</label>
              <button
                type="button"
                onClick={() => {
                  setResetEmail(email);
                  setShowForgotModal(true);
                }}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#00d4ff',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  padding: 0
                }}
              >
                Forgot Password?
              </button>
            </div>
            <input
              id="login-password"
              type="password"
              className="form-input"
              placeholder="••••••••••••"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>

          <button
            type="submit"
            id="login-submit-btn"
            className="btn-primary"
            style={{ width: '100%', marginTop: '0.5rem', padding: '0.85rem' }}
            disabled={loading}
          >
            {loading ? (
              <>
                <RefreshCw size={16} className="animate-spin" /> Authenticating...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div style={{ marginTop: '1.5rem', textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Don't have an account yet?{' '}
          <Link to="/signup" style={{ color: '#00d4ff', fontWeight: 600 }}>
            Create Account
          </Link>
        </div>
      </div>

      {/* Password Reset Modal */}
      {showForgotModal && (
        <div
          id="forgot-password-modal"
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(2, 11, 24, 0.85)',
            backdropFilter: 'blur(6px)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1rem'
          }}
        >
          <div
            className="card"
            style={{
              width: '100%',
              maxWidth: '420px',
              padding: '2rem',
              border: '1px solid rgba(0, 212, 255, 0.4)'
            }}
          >
            <h3 style={{ fontSize: '1.25rem', fontFamily: 'var(--font-heading)', marginBottom: '0.5rem' }}>
              Reset Password
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.25rem' }}>
              Enter your registered clinical email to receive a secure password reset link.
            </p>

            {resetStatus === 'sent' ? (
              <div
                style={{
                  padding: '1rem',
                  backgroundColor: 'rgba(16, 185, 129, 0.15)',
                  border: '1px solid var(--success)',
                  borderRadius: 'var(--radius-sm)',
                  marginBottom: '1.25rem',
                  fontSize: '0.85rem'
                }}
              >
                Password reset email sent! Please check your inbox.
              </div>
            ) : (
              <form onSubmit={handlePasswordReset}>
                {resetError && (
                  <div
                    style={{
                      padding: '0.75rem',
                      backgroundColor: 'rgba(239, 68, 68, 0.15)',
                      border: '1px solid var(--danger)',
                      borderRadius: 'var(--radius-sm)',
                      marginBottom: '1rem',
                      fontSize: '0.8rem'
                    }}
                  >
                    {resetError}
                  </div>
                )}
                <div style={{ marginBottom: '1.25rem' }}>
                  <label className="form-label">Email Address</label>
                  <input
                    type="email"
                    required
                    className="form-input"
                    placeholder="practitioner@clinic.org"
                    value={resetEmail}
                    onChange={(e) => setResetEmail(e.target.value)}
                  />
                </div>
                <div style={{ display: 'flex', gap: '0.75rem' }}>
                  <button
                    type="button"
                    className="btn-secondary"
                    style={{ flex: 1 }}
                    onClick={() => {
                      setShowForgotModal(false);
                      setResetStatus(null);
                      setResetError(null);
                    }}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="btn-primary"
                    style={{ flex: 1 }}
                    disabled={resetStatus === 'sending'}
                  >
                    {resetStatus === 'sending' ? 'Sending...' : 'Send Reset Link'}
                  </button>
                </div>
              </form>
            )}

            {resetStatus === 'sent' && (
              <button
                type="button"
                className="btn-primary"
                style={{ width: '100%' }}
                onClick={() => {
                  setShowForgotModal(false);
                  setResetStatus(null);
                }}
              >
                Close
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default LoginPage;
