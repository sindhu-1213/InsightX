import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity, Menu, X, History, Brain, Sparkles, LogOut, UserCheck } from 'lucide-react';

function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();
  const { currentUser, userProfile, logout } = useAuth();

  const navLinks = [
    { label: 'Home', path: '/' },
    { label: 'Analyze', path: '/analyze', icon: Sparkles },
    { label: 'History', path: '/history', icon: History },
    { label: 'Model Card', path: '/model', icon: Brain },
  ];

  const isActive = (path) => location.pathname === path;

  const handleLogout = async () => {
    await logout();
    setMobileMenuOpen(false);
  };

  return (
    <header
      id="main-navbar"
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 1000,
        backgroundColor: 'rgba(4, 20, 40, 0.95)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid var(--border)',
        height: '80px',
        display: 'flex',
        alignItems: 'center'
      }}
    >
      <div
        className="container"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          width: '100%'
        }}
      >
        {/* Brand */}
        <Link
          to="/"
          id="nav-brand-logo"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            textDecoration: 'none'
          }}
        >
          <div
            style={{
              width: '38px',
              height: '38px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 102, 255, 0.2))',
              border: '1px solid var(--cyan)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 15px var(--cyan-glow)'
            }}
          >
            <Activity size={22} color="#00d4ff" />
          </div>
          <div>
            <span
              style={{
                fontFamily: 'var(--font-heading)',
                fontSize: '1.5rem',
                fontWeight: 900,
                color: '#00d4ff',
                letterSpacing: '0.08em',
                textShadow: '0 0 12px rgba(0, 212, 255, 0.35)'
              }}
            >
              InsightX
            </span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <nav
          className="desktop-nav"
          style={{
            display: 'none',
            alignItems: 'center',
            gap: '2rem'
          }}
        >
          {navLinks.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              id={`nav-link-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                fontSize: '0.95rem',
                fontFamily: 'var(--font-body)',
                fontWeight: 500,
                color: isActive(item.path) ? '#00d4ff' : '#8ab4c9',
                borderBottom: isActive(item.path) ? '2px solid #00d4ff' : '2px solid transparent',
                paddingBottom: '0.25rem',
                transition: 'all 0.2s ease',
                textDecoration: 'none'
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = '#ffffff')}
              onMouseLeave={(e) => (e.currentTarget.style.color = isActive(item.path) ? '#00d4ff' : '#8ab4c9')}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        {/* Desktop Auth Buttons */}
        <div
          className="desktop-auth"
          style={{
            display: 'none',
            alignItems: 'center',
            gap: '1rem'
          }}
        >
          {currentUser ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  fontSize: '0.85rem',
                  color: '#ffffff',
                  backgroundColor: 'rgba(10, 31, 53, 0.8)',
                  border: '1px solid var(--border)',
                  borderRadius: '9999px',
                  padding: '0.35rem 0.85rem'
                }}
              >
                <UserCheck size={14} color="#00d4ff" />
                <span>{currentUser.displayName || currentUser.email?.split('@')[0]}</span>
                {userProfile?.role && (
                  <span style={{ color: '#00d4ff', fontSize: '0.75rem', fontWeight: 600 }}>
                    ({userProfile.role})
                  </span>
                )}
              </div>
              <button
                id="nav-logout-btn"
                onClick={handleLogout}
                className="btn-secondary btn-sm"
                style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}
              >
                <LogOut size={14} /> Logout
              </button>
            </div>
          ) : (
            <>
              <Link
                to="/login"
                id="nav-login-btn"
                style={{
                  color: '#ffffff',
                  fontSize: '0.9rem',
                  fontWeight: 600,
                  padding: '0.5rem 1rem',
                  borderRadius: '6px',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => (e.currentTarget.style.color = '#00d4ff')}
                onMouseLeave={(e) => (e.currentTarget.style.color = '#ffffff')}
              >
                Login
              </Link>
              <Link
                to="/signup"
                id="nav-signup-btn"
                className="btn-primary btn-sm"
              >
                Sign Up
              </Link>
            </>
          )}
        </div>

        {/* Mobile Hamburger Toggle */}
        <button
          id="nav-mobile-toggle-btn"
          aria-label="Toggle Navigation"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#00d4ff',
            cursor: 'pointer',
            padding: '0.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          {mobileMenuOpen ? <X size={26} /> : <Menu size={26} />}
        </button>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div
          id="mobile-nav-drawer"
          style={{
            position: 'absolute',
            top: '80px',
            left: 0,
            width: '100%',
            backgroundColor: '#041428',
            borderBottom: '1px solid var(--border)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.7)'
          }}
        >
          {navLinks.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setMobileMenuOpen(false)}
              style={{
                color: isActive(item.path) ? '#00d4ff' : '#ffffff',
                fontFamily: 'var(--font-heading)',
                fontSize: '1rem',
                padding: '0.5rem 0',
                borderBottom: '1px solid rgba(13, 45, 74, 0.4)'
              }}
            >
              {item.label}
            </Link>
          ))}
          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
            {currentUser ? (
              <button
                onClick={handleLogout}
                className="btn-secondary btn-sm"
                style={{ flex: 1, textAlign: 'center' }}
              >
                Sign Out ({currentUser.displayName || currentUser.email?.split('@')[0]})
              </button>
            ) : (
              <>
                <Link
                  to="/login"
                  onClick={() => setMobileMenuOpen(false)}
                  className="btn-secondary btn-sm"
                  style={{ flex: 1, textAlign: 'center' }}
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  onClick={() => setMobileMenuOpen(false)}
                  className="btn-primary btn-sm"
                  style={{ flex: 1, textAlign: 'center' }}
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>
        </div>
      )}

      {/* Responsive CSS for desktop visibility */}
      <style>{`
        @media (min-width: 768px) {
          .desktop-nav { display: flex !important; }
          .desktop-auth { display: flex !important; }
          #nav-mobile-toggle-btn { display: none !important; }
        }
      `}</style>
    </header>
  );
}

export default Navbar;
