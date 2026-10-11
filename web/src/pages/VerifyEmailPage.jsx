import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth, getFriendlyAuthErrorMessage } from '../context/AuthContext';
import { Mail, RefreshCw, CheckCircle, AlertCircle, LogOut } from 'lucide-react';

function VerifyEmailPage() {
  const { currentUser, sendVerification, reloadUser, logout } = useAuth();
  const [resendStatus, setResendStatus] = useState(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    document.title = 'InsightX | Verify Clinical Email';
  }, []);

  const handleResend = async () => {
    setError(null);
    setResendStatus('sending');
    try {
      await sendVerification();
      setResendStatus('sent');
      setTimeout(() => setResendStatus(null), 8000);
    } catch (err) {
      setResendStatus(null);
      setError(getFriendlyAuthErrorMessage(err));
    }
  };

  const handleCheckVerified = async () => {
    setError(null);
    setChecking(true);
    try {
      const user = await reloadUser();
      if (user && user.emailVerified) {
        navigate('/analyze', { replace: true });
      } else {
        setError('Email is not verified yet. Please check your inbox and click the verification link.');
      }
    } catch (err) {
      setError(getFriendlyAuthErrorMessage(err));
    } finally {
      setChecking(false);
    }
  };

  const handleSignOut = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <div
      id="verify-email-page-container"
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
          maxWidth: '520px',
          padding: '2.5rem 2rem',
          textAlign: 'center',
          border: '1px solid rgba(0, 212, 255, 0.35)',
          boxShadow: '0 12px 40px rgba(0, 0, 0, 0.6)'
        }}
      >
        <div
          style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            backgroundColor: 'rgba(0, 212, 255, 0.12)',
            border: '2px solid var(--cyan)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.5rem auto',
            color: '#00d4ff',
            boxShadow: '0 0 20px var(--cyan-glow)'
          }}
        >
          <Mail size={32} />
        </div>

        <h2
          id="verify-email-title"
          style={{
            fontSize: '1.6rem',
            fontFamily: 'var(--font-heading)',
            color: '#ffffff',
            marginBottom: '0.75rem'
          }}
        >
          Verify Your Clinical Email
        </h2>

        <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: '1.7', marginBottom: '1.5rem' }}>
          A verification link was dispatched to{' '}
          <strong style={{ color: '#00d4ff' }}>{currentUser?.email || 'your email address'}</strong>.
          Please open your inbox and click the link to activate your radiograph inference and history permissions.
        </p>

        {resendStatus === 'sent' && (
          <div
            style={{
              padding: '0.75rem 1rem',
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              border: '1px solid var(--success)',
              borderRadius: 'var(--radius-sm)',
              marginBottom: '1.5rem',
              color: '#ffffff',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}
          >
            <CheckCircle size={18} color="#10b981" />
            <span>New verification email sent! Please check your spam folder if not visible.</span>
          </div>
        )}

        {error && (
          <div
            style={{
              padding: '0.75rem 1rem',
              backgroundColor: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid var(--danger)',
              borderRadius: 'var(--radius-sm)',
              marginBottom: '1.5rem',
              color: '#ffffff',
              fontSize: '0.85rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}
          >
            <AlertCircle size={18} color="#ef4444" />
            <span>{error}</span>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', marginBottom: '1.5rem' }}>
          <button
            id="i-have-verified-btn"
            className="btn-primary"
            style={{ width: '100%', padding: '0.85rem' }}
            onClick={handleCheckVerified}
            disabled={checking}
          >
            {checking ? (
              <>
                <RefreshCw size={16} className="animate-spin" /> Checking Status...
              </>
            ) : (
              "I've Verified My Email &rarr;"
            )}
          </button>

          <button
            id="resend-verification-btn"
            className="btn-secondary"
            style={{ width: '100%', padding: '0.85rem' }}
            onClick={handleResend}
            disabled={resendStatus === 'sending'}
          >
            {resendStatus === 'sending' ? 'Sending link...' : 'Resend Verification Link'}
          </button>
        </div>

        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '1.25rem' }}>
          <button
            id="verify-email-signout-btn"
            onClick={handleSignOut}
            style={{
              background: 'transparent',
              border: 'none',
              color: 'var(--text-muted)',
              fontSize: '0.85rem',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem'
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = '#ffffff')}
            onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
          >
            <LogOut size={14} /> Sign out and switch account
          </button>
        </div>
      </div>
    </div>
  );
}

export default VerifyEmailPage;
