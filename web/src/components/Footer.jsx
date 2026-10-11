import React from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, ShieldAlert, Heart, ExternalLink, Activity } from 'lucide-react';

function Footer() {
  return (
    <footer
      id="main-footer"
      style={{
        backgroundColor: '#030e1d',
        borderTop: '1px solid var(--border)',
        padding: '3rem 0 2rem 0',
        marginTop: 'auto'
      }}
    >
      <div className="container">
        {/* Research Disclaimer Banner */}
        <div
          id="footer-research-disclaimer"
          style={{
            backgroundColor: 'rgba(10, 31, 53, 0.85)',
            border: '1px solid rgba(0, 212, 255, 0.3)',
            borderRadius: 'var(--radius-md)',
            padding: '1.25rem 1.5rem',
            marginBottom: '2.5rem',
            display: 'flex',
            flexDirection: 'row',
            gap: '1rem',
            alignItems: 'flex-start',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
          }}
        >
          <div style={{ color: '#00d4ff', flexShrink: 0, marginTop: '2px' }}>
            <ShieldAlert size={24} />
          </div>
          <div>
            <h4
              style={{
                color: '#00d4ff',
                fontSize: '0.95rem',
                marginBottom: '0.35rem',
                fontFamily: 'var(--font-heading)'
              }}
            >
              INVESTIGATIONAL RESEARCH DEMONSTRATION NOTICE
            </h4>
            <p
              style={{
                color: 'var(--text-secondary)',
                fontSize: '0.85rem',
                lineHeight: '1.6',
                margin: 0
              }}
            >
              INSIGHT-X is an investigational artificial intelligence decision support prototype designed strictly
              for clinical research, educational evaluation, and demonstration purposes. It does not provide medical
              diagnoses, treatment prescriptions, or definitive radiological findings, and must never replace the
              independent assessment and evaluation of a licensed clinical professional or radiologist.
            </p>
          </div>
        </div>

        {/* Links and Metadata */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '2rem',
            marginBottom: '2rem'
          }}
        >
          {/* Brand Info */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
              <Activity size={20} color="#00d4ff" />
              <span
                style={{
                  fontFamily: 'var(--font-heading)',
                  fontSize: '1.2rem',
                  fontWeight: 800,
                  color: '#00d4ff'
                }}
              >
                InsightX
              </span>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: '1.6' }}>
              Multimodal clinical decision support system fusing deep radiological vision with medical NLP
              and explainable activation heatmaps.
            </p>
          </div>

          {/* Quick Navigation */}
          <div>
            <h5 style={{ color: '#ffffff', fontSize: '0.9rem', marginBottom: '0.85rem' }}>Platform</h5>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li><Link to="/" style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Home</Link></li>
              <li><Link to="/analyze" style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Analyze Radiograph</Link></li>
              <li><Link to="/history" style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Analysis History</Link></li>
              <li><Link to="/model" style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Model Card & Verification</Link></li>
            </ul>
          </div>

          {/* Clinical & Scientific Resources */}
          <div>
            <h5 style={{ color: '#ffffff', fontSize: '0.9rem', marginBottom: '0.85rem' }}>Data & Research</h5>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li>
                <a
                  href="https://openi.nlm.nih.gov/"
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}
                >
                  Indiana University Dataset <ExternalLink size={12} />
                </a>
              </li>
              <li>
                <Link to="/model" style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                  Platt Calibration & AUROC Metrics
                </Link>
              </li>
              <li>
                <a
                  href="https://screenapp.io/app/v/CmRiOCmIsx"
                  target="_blank"
                  rel="noreferrer"
                  style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}
                >
                  Project Video Demo <ExternalLink size={12} />
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom copyright line */}
        <div
          style={{
            borderTop: '1px solid rgba(13, 45, 74, 0.5)',
            paddingTop: '1.5rem',
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '1rem',
            color: 'var(--text-muted)',
            fontSize: '0.8rem'
          }}
        >
          <span>&copy; {new Date().getFullYear()} InsightX Research Team. Strictly for investigational use.</span>
          <span>Engineered with MobileNetV2, ONNX Runtime &amp; FastAPI</span>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
