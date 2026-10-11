import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { collection, query, orderBy, getDocs } from 'firebase/firestore';
import { db } from '../firebase';
import { useAuth } from '../context/AuthContext';
import {
  History,
  Activity,
  AlertTriangle,
  RefreshCw,
  FolderOpen,
  Calendar,
  Layers,
  ArrowRight
} from 'lucide-react';

function HistoryPage() {
  const { currentUser } = useAuth();
  const [analyses, setAnalyses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('ALL');

  const fetchAnalyses = async () => {
    if (!currentUser) return;
    setLoading(true);
    setError(null);

    try {
      const analysesRef = collection(db, 'users', currentUser.uid, 'analyses');
      const q = query(analysesRef, orderBy('createdAt', 'desc'));
      const snapshot = await getDocs(q);

      const items = [];
      snapshot.forEach((doc) => {
        items.push({ id: doc.id, ...doc.data() });
      });
      setAnalyses(items);
    } catch (err) {
      console.error('Failed to load history:', err);
      setError('Unable to load clinical history from Firestore. Verify your network or permissions.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    document.title = 'InsightX | Clinical Analysis History';
    fetchAnalyses();
  }, [currentUser]);

  const filteredItems = analyses.filter((item) => {
    if (filter === 'ALL') return true;
    return (item.label || '').toUpperCase() === filter;
  });

  const formatDate = (val) => {
    if (!val) return '—';
    try {
      if (typeof val === 'string') return new Date(val).toLocaleString();
      if (val.toDate && typeof val.toDate === 'function') return val.toDate().toLocaleString();
      if (val.seconds) return new Date(val.seconds * 1000).toLocaleString();
      return String(val);
    } catch {
      return String(val);
    }
  };

  return (
    <div id="history-page-container" className="container" style={{ padding: '3rem 1.25rem 5rem 1.25rem' }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'space-between',
          alignItems: 'center',
          gap: '1rem',
          marginBottom: '2.5rem'
        }}
      >
        <div>
          <span className="badge-cyan" style={{ marginBottom: '0.5rem' }}>
            <History size={14} /> Clinical Audit Log
          </span>
          <h1 style={{ fontSize: 'clamp(1.8rem, 4vw, 2.5rem)', fontFamily: 'var(--font-heading)' }}>
            Analysis History
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
            Immutable records persisted under your practitioner profile in Firestore.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <button
            id="refresh-history-btn"
            className="btn-secondary btn-sm"
            onClick={fetchAnalyses}
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
          </button>

          {/* Filter Pills */}
          <div style={{ display: 'flex', gap: '0.35rem' }}>
            {['ALL', 'NORMAL', 'ABNORMAL'].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={filter === f ? 'btn-primary btn-sm' : 'btn-secondary btn-sm'}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <div
          style={{
            padding: '1rem',
            backgroundColor: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid var(--danger)',
            borderRadius: 'var(--radius-sm)',
            marginBottom: '1.5rem',
            color: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}
        >
          <AlertTriangle size={18} color="#ef4444" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Content Area */}
      {loading ? (
        <div
          style={{
            height: '300px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1rem',
            color: '#00d4ff'
          }}
        >
          <Activity size={32} className="animate-spin" />
          <span style={{ fontFamily: 'var(--font-heading)', fontSize: '0.9rem' }}>
            Retrieving Analyses from Firestore...
          </span>
        </div>
      ) : analyses.length === 0 ? (
        <div
          className="card"
          style={{
            textAlign: 'center',
            padding: '4rem 2rem',
            color: 'var(--text-muted)'
          }}
        >
          <FolderOpen size={48} color="#0d2d4a" style={{ margin: '0 auto 1rem auto' }} />
          <h3 style={{ fontSize: '1.25rem', color: '#ffffff', fontFamily: 'var(--font-heading)', marginBottom: '0.5rem' }}>
            No Analysis Records Found
          </h3>
          <p style={{ maxWidth: '420px', margin: '0 auto 1.5rem auto', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            You haven't run any radiograph analyses yet. Navigate to the analysis workbench to test an image.
          </p>
          <Link to="/analyze" className="btn-primary">
            Launch Analysis Workbench <ArrowRight size={16} />
          </Link>
        </div>
      ) : (
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ backgroundColor: 'rgba(4, 20, 40, 0.9)', borderBottom: '1px solid var(--border)' }}>
                  <th style={{ padding: '1rem 1.25rem', color: '#ffffff', fontFamily: 'var(--font-heading)' }}>Record ID</th>
                  <th style={{ padding: '1rem 1.25rem', color: '#ffffff', fontFamily: 'var(--font-heading)' }}>Classification</th>
                  <th style={{ padding: '1rem 1.25rem', color: '#ffffff', fontFamily: 'var(--font-heading)' }}>Calibrated Probability</th>
                  <th style={{ padding: '1rem 1.25rem', color: '#ffffff', fontFamily: 'var(--font-heading)' }}>Cluster</th>
                  <th style={{ padding: '1rem 1.25rem', color: '#ffffff', fontFamily: 'var(--font-heading)' }}>Model Version</th>
                  <th style={{ padding: '1rem 1.25rem', color: '#ffffff', fontFamily: 'var(--font-heading)' }}>Logged At</th>
                </tr>
              </thead>
              <tbody>
                {filteredItems.map((row, idx) => {
                  const prob = typeof row.probability === 'number' ? row.probability : 0;
                  return (
                    <tr
                      key={row.id}
                      style={{
                        borderBottom: '1px solid rgba(13, 45, 74, 0.5)',
                        backgroundColor: idx % 2 === 0 ? 'transparent' : 'rgba(4, 20, 40, 0.3)'
                      }}
                    >
                      <td style={{ padding: '1rem 1.25rem', fontFamily: 'monospace', color: '#00d4ff' }}>
                        {row.id.slice(0, 12)}...
                      </td>
                      <td style={{ padding: '1rem 1.25rem' }}>
                        <span
                          className="badge-cyan"
                          style={{
                            backgroundColor:
                              row.label === 'Normal' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                            color: row.label === 'Normal' ? 'var(--success)' : 'var(--danger)',
                            borderColor: row.label === 'Normal' ? 'var(--success)' : 'var(--danger)'
                          }}
                        >
                          {row.label}
                        </span>
                      </td>
                      <td style={{ padding: '1rem 1.25rem', fontWeight: 600, color: '#ffffff' }}>
                        {(prob * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '1rem 1.25rem', color: 'var(--text-secondary)' }}>
                        Cluster #{row.cluster ?? 0}
                      </td>
                      <td style={{ padding: '1rem 1.25rem', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                        {row.modelVersion}
                      </td>
                      <td style={{ padding: '1rem 1.25rem', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                        {formatDate(row.createdAt)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default HistoryPage;
