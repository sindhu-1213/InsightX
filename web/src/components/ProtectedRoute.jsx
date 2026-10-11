import React from 'react';
import { Navigate, useLocation, Outlet } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Activity } from 'lucide-react';

function ProtectedRoute({ children, requireVerified = true }) {
  const { currentUser, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div
        style={{
          minHeight: '60vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '1rem',
          color: '#00d4ff'
        }}
      >
        <Activity size={36} className="animate-pulse" />
        <span style={{ fontFamily: 'var(--font-heading)', fontSize: '0.95rem' }}>
          Verifying Clinical Authentication...
        </span>
      </div>
    );
  }

  if (!currentUser) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (requireVerified && !currentUser.emailVerified) {
    return <Navigate to="/verify-email" replace />;
  }

  return children ? children : <Outlet />;
}

export default ProtectedRoute;
