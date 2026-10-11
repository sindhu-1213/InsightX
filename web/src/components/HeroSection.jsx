import React from 'react';
import { Link } from 'react-router-dom';
import { Sparkles, ArrowRight, ShieldCheck, Cpu, Database } from 'lucide-react';

function HeroSection() {
  return (
    <section
      id="hero-section"
      style={{
        position: 'relative',
        minHeight: '80vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        textAlign: 'center',
        padding: '4rem 1.25rem 5rem 1.25rem',
        overflow: 'hidden'
      }}
    >
      {/* Background glow meshes */}
      <div
        className="mesh-glow"
        style={{
          width: '500px',
          height: '500px',
          top: '10%',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'radial-gradient(circle, rgba(0, 212, 255, 0.25) 0%, rgba(0, 102, 255, 0.1) 70%, transparent 100%)'
        }}
      />

      <div style={{ position: 'relative', zIndex: 1, maxWidth: '960px', margin: '0 auto' }}>
        {/* Top Clinical Badge */}
        <div style={{ marginBottom: '1.5rem', display: 'inline-block' }}>
          <span className="badge-cyan">
            <Sparkles size={14} /> Clinical Decision Support Prototype
          </span>
        </div>

        {/* Orbitron Heading */}
        <h1
          id="hero-main-title"
          style={{
            fontSize: 'clamp(2.2rem, 6vw, 4rem)',
            color: '#ffffff',
            marginBottom: '20px',
            fontFamily: 'Orbitron, sans-serif',
            lineHeight: 1.15,
            letterSpacing: '0.04em',
            textShadow: '0 0 30px rgba(0, 212, 255, 0.25)'
          }}
        >
          AI-Powered Chest X-Ray Analysis
        </h1>

        {/* Subtitle */}
        <p
          id="hero-subtitle"
          style={{
            maxWidth: '800px',
            margin: '0 auto',
            color: '#8ab4c9',
            fontSize: 'clamp(1rem, 2.5vw, 1.25rem)',
            lineHeight: '1.8',
            fontFamily: 'var(--font-body)'
          }}
        >
          InsightX combines deep learning, medical text analysis, and explainable AI to assist healthcare
          professionals in detecting abnormalities from chest radiographs.
        </p>

        {/* Action Buttons */}
        <div
          style={{
            marginTop: '35px',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '1rem',
            justifyContent: 'center',
            alignItems: 'center'
          }}
        >
          <Link
            to="/analyze"
            id="hero-get-started-btn"
            className="btn-primary"
            style={{
              padding: '14px 34px',
              fontSize: '1.05rem',
              borderRadius: '8px',
              fontWeight: 'bold'
            }}
          >
            Get Started <ArrowRight size={18} />
          </Link>
          <Link
            to="/model"
            id="hero-explore-model-btn"
            className="btn-secondary"
            style={{
              padding: '14px 28px',
              fontSize: '1.05rem',
              borderRadius: '8px'
            }}
          >
            Model Card &amp; AUROC
          </Link>
        </div>

        {/* Key Feature Stats Pills */}
        <div
          style={{
            marginTop: '4rem',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1.25rem',
            textAlign: 'left'
          }}
        >
          <div className="card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ color: '#00d4ff' }}><Cpu size={20} /></div>
              <span style={{ fontSize: '0.85rem', color: '#8ab4c9', fontWeight: 600 }}>MobileNetV2 CNN</span>
            </div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ffffff', fontFamily: 'var(--font-heading)' }}>
              0.706 ROC-AUC
            </div>
            <p style={{ fontSize: '0.8rem', color: '#4e738a', marginTop: '0.25rem' }}>
              Platt calibrated probability output
            </p>
          </div>

          <div className="card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ color: '#00d4ff' }}><Sparkles size={20} /></div>
              <span style={{ fontSize: '0.85rem', color: '#8ab4c9', fontWeight: 600 }}>Visual Explainability</span>
            </div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ffffff', fontFamily: 'var(--font-heading)' }}>
              Grad-CAM Heatmaps
            </div>
            <p style={{ fontSize: '0.8rem', color: '#4e738a', marginTop: '0.25rem' }}>
              Layer-targeted spatial activation overlays
            </p>
          </div>

          <div className="card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ color: '#00d4ff' }}><Database size={20} /></div>
              <span style={{ fontSize: '0.85rem', color: '#8ab4c9', fontWeight: 600 }}>Dataset Verification</span>
            </div>
            <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ffffff', fontFamily: 'var(--font-heading)' }}>
              Indiana University
            </div>
            <p style={{ fontSize: '0.8rem', color: '#4e738a', marginTop: '0.25rem' }}>
              Strict patient-level GroupShuffleSplit
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

export default HeroSection;
