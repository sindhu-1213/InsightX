import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import HeroSection from '../components/HeroSection';
import {
  UploadCloud,
  Cpu,
  Layers,
  FileText,
  ShieldCheck,
  CheckCircle,
  Eye,
  ArrowRight
} from 'lucide-react';

function LandingPage() {
  useEffect(() => {
    document.title = 'InsightX | Clinical Decision Support System';
  }, []);

  const steps = [
    {
      num: '01',
      title: 'Upload Radiograph',
      desc: 'Submit a chest X-ray in PNG or JPEG format (or test instantly with pre-indexed demo samples).',
      icon: UploadCloud
    },
    {
      num: '02',
      title: 'Neural Inference',
      desc: 'MobileNetV2 processes 224x224 normalized projections using ONNX Runtime with sub-second latency.',
      icon: Cpu
    },
    {
      num: '03',
      title: 'Platt Calibration',
      desc: 'Sigmoidal raw scores are scaled on held-out validation data to reflect reliable probability estimates.',
      icon: Layers
    },
    {
      num: '04',
      title: 'Grad-CAM Attention',
      desc: 'Visual gradient backpropagation localizes anatomical heatmaps over relevant lung and cardiac fields.',
      icon: Eye
    }
  ];

  return (
    <div id="landing-page-root">
      {/* Ported Hero Section */}
      <HeroSection />

      {/* Workflow Section */}
      <section
        id="workflow-section"
        style={{
          padding: '5rem 0',
          backgroundColor: 'var(--bg-secondary)',
          borderTop: '1px solid var(--border)',
          borderBottom: '1px solid var(--border)'
        }}
      >
        <div className="container">
          <div style={{ textAlign: 'center', maxWidth: '700px', margin: '0 auto 3.5rem auto' }}>
            <span className="badge-cyan" style={{ marginBottom: '1rem' }}>Clinical Workflow</span>
            <h2
              style={{
                fontSize: 'clamp(1.75rem, 4vw, 2.5rem)',
                fontFamily: 'var(--font-heading)',
                marginBottom: '1rem'
              }}
            >
              How InsightX Analyzes Radiographs
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem' }}>
              Four streamlined stages from initial image ingest to calibrated probability and visual localization.
            </p>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
              gap: '1.5rem'
            }}
          >
            {steps.map((s) => {
              const Icon = s.icon;
              return (
                <div
                  key={s.num}
                  className="card"
                  style={{
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: '1rem',
                      right: '1.25rem',
                      fontFamily: 'var(--font-heading)',
                      fontSize: '2rem',
                      fontWeight: 900,
                      color: 'rgba(0, 212, 255, 0.12)'
                    }}
                  >
                    {s.num}
                  </div>
                  <div
                    style={{
                      width: '46px',
                      height: '46px',
                      borderRadius: '10px',
                      backgroundColor: 'rgba(0, 212, 255, 0.1)',
                      border: '1px solid rgba(0, 212, 255, 0.3)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#00d4ff',
                      marginBottom: '1.25rem'
                    }}
                  >
                    <Icon size={22} />
                  </div>
                  <h3
                    style={{
                      fontSize: '1.2rem',
                      marginBottom: '0.65rem',
                      fontFamily: 'var(--font-heading)'
                    }}
                  >
                    {s.title}
                  </h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                    {s.desc}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Multimodal & Research Highlights */}
      <section style={{ padding: '5rem 0' }}>
        <div className="container">
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '3rem',
              alignItems: 'center'
            }}
          >
            <div>
              <span className="badge-cyan" style={{ marginBottom: '1rem' }}>Multimodal Grounding</span>
              <h2
                style={{
                  fontSize: 'clamp(1.75rem, 3.5vw, 2.4rem)',
                  marginBottom: '1.25rem',
                  lineHeight: 1.25
                }}
              >
                Why Image-Based Serving Outperforms Naive Text
              </h2>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem', lineHeight: '1.7' }}>
                While text-based models appear unrealistically high on evaluation sets because MeSH labels derive
                from the radiologist reports themselves, <strong>real-world clinical practice requires analyzing images directly</strong>.
              </p>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem', lineHeight: '1.7' }}>
                INSIGHT-X preserves critical medical negations (<em>no</em>, <em>not</em>, <em>without</em>) in its
                NLP pipeline and deploys the purely visual MobileNetV2 ONNX model to deliver independent decision support.
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginBottom: '2rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <CheckCircle size={18} color="#00d4ff" />
                  <span style={{ fontSize: '0.95rem', color: '#ffffff' }}>Strict patient-level GroupShuffleSplit (no patient overlap)</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <CheckCircle size={18} color="#00d4ff" />
                  <span style={{ fontSize: '0.95rem', color: '#ffffff' }}>Platt-calibrated probabilities for clinical interpretability</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <CheckCircle size={18} color="#00d4ff" />
                  <span style={{ fontSize: '0.95rem', color: '#ffffff' }}>Built-in Grad-CAM explainability targeting the final feature maps</span>
                </div>
              </div>
              <Link to="/analyze" className="btn-primary">
                Launch Analysis Workbench <ArrowRight size={18} />
              </Link>
            </div>

            {/* Diagnostic Card Preview */}
            <div
              className="card"
              style={{
                backgroundColor: 'var(--bg-card)',
                border: '1px solid rgba(0, 212, 255, 0.3)',
                boxShadow: '0 0 30px rgba(0, 212, 255, 0.15)',
                padding: '2rem'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <span style={{ fontFamily: 'var(--font-heading)', color: '#00d4ff', fontSize: '0.9rem' }}>
                  DEMO PREVIEW // UID-0038
                </span>
                <span className="badge-cyan" style={{ backgroundColor: 'rgba(239, 68, 68, 0.15)', color: '#ff6b6b', borderColor: '#ef4444' }}>
                  Abnormal
                </span>
              </div>
              <div
                style={{
                  height: '180px',
                  backgroundColor: '#041428',
                  borderRadius: '8px',
                  border: '1px dashed var(--border)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1.5rem',
                  position: 'relative',
                  overflow: 'hidden'
                }}
              >
                <div
                  style={{
                    position: 'absolute',
                    inset: 0,
                    background: 'radial-gradient(circle at 60% 40%, rgba(255, 65, 54, 0.4) 0%, rgba(255, 133, 27, 0.2) 40%, transparent 70%)'
                  }}
                />
                <span style={{ position: 'relative', zIndex: 1, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                  [Grad-CAM Attention Heatmap Overlay]
                </span>
              </div>
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.35rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Calibrated Probability</span>
                  <span style={{ color: '#00d4ff', fontWeight: 700 }}>68.4%</span>
                </div>
                <div style={{ height: '8px', backgroundColor: '#06172a', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: '68.4%', height: '100%', background: 'linear-gradient(90deg, #00d4ff, #ff4757)' }} />
                </div>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>
                Calibrated via Platt scaling (intercept: -0.3204, slope: 1.1093).
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default LandingPage;
