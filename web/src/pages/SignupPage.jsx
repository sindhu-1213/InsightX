import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth, getFriendlyAuthErrorMessage } from '../context/AuthContext';
import { UserPlus, AlertCircle, RefreshCw } from 'lucide-react';

function SignupPage() {
  const [formData, setFormData] = useState({
    displayName: '',
    email: '',
    role: 'Doctor',
    institution: '',
    password: '',
    confirmPassword: ''
  });
  const [loading, setLoading] = useState(false);
  const [googleLoading, setGoogleLoading] = useState(false);
  const [error, setError] = useState(null);

  const { signup, signInWithGoogle } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    document.title = 'InsightX | Practitioner Registration';
  }, []);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters long.');
      return;
    }

    setLoading(true);
    try {
      await signup(formData.email, formData.password, {
        displayName: formData.displayName,
        role: formData.role,
        institution: formData.institution || 'Independent Practitioner'
      });
      navigate('/verify-email', { replace: true });
    } catch (err) {
      setError(getFriendlyAuthErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignUp = async () => {
    setError(null);
    setGoogleLoading(true);
    try {
      await signInWithGoogle(formData.role, formData.institution || 'Independent Practitioner');
      navigate('/analyze', { replace: true });
    } catch (err) {
      setError(getFriendlyAuthErrorMessage(err));
    } finally {
      setGoogleLoading(false);
    }
  };

  return (
    <div
      id="signup-page-container"
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
          maxWidth: '480px',
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
            <UserPlus size={26} />
          </div>
          <h2
            id="signup-heading"
            style={{
              fontSize: '1.6rem',
              fontFamily: 'var(--font-heading)',
              color: '#ffffff',
              marginBottom: '0.4rem'
            }}
          >
            Practitioner Registration
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Create an account with email verification
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

        {/* Google Sign Up Button */}
        <button
          type="button"
          id="google-signup-btn"
          onClick={handleGoogleSignUp}
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
          {googleLoading ? 'Connecting to Google...' : 'Sign up with Google'}
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
          <span>OR REGISTER WITH EMAIL</span>
          <div style={{ flex: 1, height: '1px', backgroundColor: 'var(--border)' }} />
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <label className="form-label" htmlFor="signup-name">Practitioner Display Name</label>
            <input
              id="signup-name"
              name="displayName"
              type="text"
              maxLength={99}
              required
              className="form-input"
              placeholder="Dr. Eleanor Vance"
              value={formData.displayName}
              onChange={handleChange}
            />
          </div>

          <div>
            <label className="form-label" htmlFor="signup-email">Clinical Email Address</label>
            <input
              id="signup-email"
              name="email"
              type="email"
              required
              className="form-input"
              placeholder="eleanor.vance@clinic.org"
              value={formData.email}
              onChange={handleChange}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div>
              <label className="form-label" htmlFor="signup-role">Role</label>
              <select
                id="signup-role"
                name="role"
                className="form-select"
                value={formData.role}
                onChange={handleChange}
              >
                <option value="Doctor">Doctor</option>
                <option value="Staff">Staff</option>
                <option value="Student">Student</option>
              </select>
            </div>

            <div>
              <label className="form-label" htmlFor="signup-institution">
                Institution (Optional)
              </label>
              <input
                id="signup-institution"
                name="institution"
                type="text"
                maxLength={99}
                className="form-input"
                placeholder="e.g. University Hospital"
                value={formData.institution}
                onChange={handleChange}
              />
            </div>
          </div>

          <div>
            <label className="form-label" htmlFor="signup-password">Password</label>
            <input
              id="signup-password"
              name="password"
              type="password"
              required
              minLength={6}
              className="form-input"
              placeholder="••••••••••••"
              value={formData.password}
              onChange={handleChange}
            />
          </div>

          <div>
            <label className="form-label" htmlFor="signup-confirm-password">Confirm Password</label>
            <input
              id="signup-confirm-password"
              name="confirmPassword"
              type="password"
              required
              minLength={6}
              className="form-input"
              placeholder="••••••••••••"
              value={formData.confirmPassword}
              onChange={handleChange}
            />
          </div>

          <button
            type="submit"
            id="signup-submit-btn"
            className="btn-primary"
            style={{ width: '100%', marginTop: '0.5rem', padding: '0.85rem' }}
            disabled={loading}
          >
            {loading ? (
              <>
                <RefreshCw size={16} className="animate-spin" /> Creating Account...
              </>
            ) : (
              'Create Practitioner Account'
            )}
          </button>
        </form>

        <div style={{ marginTop: '1.5rem', textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Already have an account?{' '}
          <Link to="/login" style={{ color: '#00d4ff', fontWeight: 600 }}>
            Sign In
          </Link>
        </div>
      </div>
    </div>
  );
}

export default SignupPage;
