import React, { useEffect } from 'react';
import metricsData from '../data/metrics.json';
import {
  Brain,
  Cpu,
  Layers,
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Database,
  BarChart3,
  Eye,
  FileText,
  Sliders
} from 'lucide-react';

function ModelPage() {
  const metrics = metricsData;

  useEffect(() => {
    document.title = 'InsightX | Model Card & Verification';
  }, []);

  const baselineComparisons = [
    {
      modality: 'Text-Only Baseline',
      features: 'Findings TF-IDF (1,000 features)',
      accuracy: 'PENDING: filled in after real training',
      f1: 'PENDING: filled in after real training',
      rocAuc: 'PENDING: filled in after real training',
      role: 'Research NLP baseline only'
    },
    {
      modality: 'Image-Only (Served Model)',
      features: 'MobileNetV2 Fine-Tuned (GAP 1,280-dim)',
      accuracy: 'PENDING: filled in after real training',
      f1: 'PENDING: filled in after real training',
      rocAuc: 'PENDING: filled in after real training',
      role: 'Active ONNX production pipeline'
    },
    {
      modality: 'Multimodal Fused',
      features: 'Concatenated (1,280 image + 1,000 text)',
      accuracy: 'PENDING: filled in after real training',
      f1: 'PENDING: filled in after real training',
      rocAuc: 'PENDING: filled in after real training',
      role: 'Exploratory clustering benchmark'
    }
  ];

  const limitations = [
    {
      title: 'Single-Institution Cohort',
      desc: 'All training and evaluation data originates exclusively from the Indiana University hospital network. Performance under domain shift across distinct radiology scanner manufacturers (GE, Siemens, Philips) or diverse patient populations has not been established.'
    },
    {
      title: 'Label Noise from Report-Derived Annotations',
      desc: 'Target binary classes were constructed from automated indexing and MeSH extraction of clinical text narratives rather than panel-adjudicated radiologist bounding boxes or biopsy confirmations.'
    },
    {
      title: 'Mixed Frontal and Lateral Projections',
      desc: 'The dataset aggregates both frontal (PA/AP) and lateral radiograph views into a single classification pipeline without projection-specific conditioning or view-separated heads.'
    },
    {
      title: 'No Clinical or Prospective Trial Validation',
      desc: 'The system has not been evaluated in prospective clinical trials, reader studies, or emergency settings. It is not an FDA-cleared or CE-marked medical device.'
    }
  ];

  const failureModes = [
    {
      title: 'Subtle and Micro-Lesions',
      desc: 'Global Average Pooling across the 7x7 spatial grid can dilute focal abnormalities smaller than 1 cm (e.g., solitary pulmonary nodules or subtle apical pneumothoraces) within healthy lung parenchyma.'
    },
    {
      title: 'Artifacts and Foreign Objects',
      desc: 'Support hardware, pacemakers, surgical clips, ECG leads, and patient motion artifacts can cause spurious convolutional activations.'
    },
    {
      title: 'Co-Occurring Abnormalities',
      desc: 'Binary classification aggregates critical acute conditions (e.g., tension pneumothorax) and chronic benign findings (e.g., mild scoliosis, granulomatous calcification) into the same abnormal label.'
    },
    {
      title: 'Out-of-Distribution Inputs',
      desc: 'Severely underexposed/overexposed radiographs, non-chest scans, or pediatric anatomy will yield untrustworthy predictions and invalid Grad-CAM heatmaps.'
    }
  ];

  return (
    <div
      id="model-page-root"
      className="container"
      style={{
        padding: '3rem 1rem 6rem 1rem',
        maxWidth: '1100px'
      }}
    >
      {/* Header */}
      <div style={{ textAlign: 'center', maxWidth: '820px', margin: '0 auto 3rem auto' }}>
        <span className="badge-cyan" style={{ marginBottom: '0.75rem' }}>
          <Brain size={14} /> Open Science &amp; Model Verification
        </span>
        <h1
          style={{
            fontSize: 'clamp(2rem, 5vw, 3rem)',
            fontFamily: 'var(--font-heading)',
            marginBottom: '0.75rem'
          }}
        >
          Model Card &amp; Technical Architecture
        </h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: '1.05rem', lineHeight: '1.6' }}>
          Comprehensive specifications, test evaluation metrics, multimodal baselines, and safety limitations for InsightX.
        </p>
      </div>

      {/* Regulatory & Research Disclaimer Alert */}
      <div
        className="card"
        style={{
          backgroundColor: 'rgba(10, 31, 53, 0.85)',
          border: '1px solid rgba(0, 212, 255, 0.4)',
          marginBottom: '2.5rem',
          padding: '1.5rem',
          boxShadow: '0 0 25px rgba(0, 212, 255, 0.15)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <ShieldAlert size={24} color="#00d4ff" />
          <h2
            style={{
              fontSize: '1.15rem',
              color: '#00d4ff',
              fontFamily: 'var(--font-heading)',
              margin: 0
            }}
          >
            Intended Use &amp; Regulatory Notice
          </h2>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6', margin: 0 }}>
          {metrics.disclaimer}
        </p>
      </div>

      {/* SECTION 1: SYSTEM PIPELINE DIAGRAM */}
      <section style={{ marginBottom: '3.5rem' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <span className="badge-cyan" style={{ fontSize: '0.75rem', marginBottom: '0.5rem' }}>
            System Architecture
          </span>
          <h2 style={{ fontSize: '1.5rem', fontFamily: 'var(--font-heading)', color: '#ffffff' }}>
            Inference &amp; Explainability Pipeline
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            From in-memory radiograph ingest to pure NumPy spatial Grad-CAM activation and calibrated probabilities.
          </p>
        </div>

        {/* Visual Flowchart Diagram */}
        <div
          className="card"
          style={{
            backgroundColor: 'rgba(4, 20, 40, 0.95)',
            padding: '2rem 1.5rem',
            overflowX: 'auto'
          }}
        >
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '1.25rem',
              minWidth: '320px'
            }}
          >
            {/* Step 1 */}
            <div
              style={{
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                padding: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem'
              }}
            >
              <div
                style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '8px',
                  backgroundColor: 'rgba(0, 212, 255, 0.15)',
                  border: '1px solid var(--cyan)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#00d4ff',
                  fontWeight: 900,
                  fontFamily: 'var(--font-heading)',
                  flexShrink: 0
                }}
              >
                01
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '1rem', color: '#ffffff', marginBottom: '0.2rem' }}>
                  Image Acquisition &amp; Preprocessing
                </h3>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: 0 }}>
                  224×224 normalization, grayscale replication to 3 RGB channels, and MobileNetV2 pixel scaling [-1, 1]. Ingested strictly in RAM.
                </p>
              </div>
            </div>

            {/* Down Arrow */}
            <div style={{ textAlign: 'center', color: '#00d4ff', fontSize: '1.2rem' }}>&darr;</div>

            {/* Step 2 */}
            <div
              style={{
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                padding: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem'
              }}
            >
              <div
                style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '8px',
                  backgroundColor: 'rgba(0, 212, 255, 0.15)',
                  border: '1px solid var(--cyan)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#00d4ff',
                  fontWeight: 900,
                  fontFamily: 'var(--font-heading)',
                  flexShrink: 0
                }}
              >
                02
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '1rem', color: '#ffffff', marginBottom: '0.2rem' }}>
                  Dual-Output ONNX Runtime Inference
                </h3>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: 0 }}>
                  Simultaneously computes last convolutional feature map (7×7×1280) and raw linear logits via CPU-optimized <code style={{ color: '#00d4ff' }}>onnxruntime</code> without TensorFlow dependencies.
                </p>
              </div>
            </div>

            {/* Down Arrow */}
            <div style={{ textAlign: 'center', color: '#00d4ff', fontSize: '1.2rem' }}>&darr;</div>

            {/* Step 3 */}
            <div
              style={{
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                padding: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem'
              }}
            >
              <div
                style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '8px',
                  backgroundColor: 'rgba(0, 212, 255, 0.15)',
                  border: '1px solid var(--cyan)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#00d4ff',
                  fontWeight: 900,
                  fontFamily: 'var(--font-heading)',
                  flexShrink: 0
                }}
              >
                03
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '1rem', color: '#ffffff', marginBottom: '0.2rem' }}>
                  Platt Scaling Probability Calibration
                </h3>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: 0 }}>
                  Log-odds transformation: <code style={{ color: '#00d4ff' }}>P = 1 / (1 + exp(-(coef * raw_logit + intercept)))</code> using coefficients fitted on validation data, mitigating overconfidence.
                </p>
              </div>
            </div>

            {/* Down Arrow */}
            <div style={{ textAlign: 'center', color: '#00d4ff', fontSize: '1.2rem' }}>&darr;</div>

            {/* Step 4 */}
            <div
              style={{
                backgroundColor: 'var(--bg-card)',
                border: '1px solid var(--cyan)',
                borderRadius: '8px',
                padding: '1.25rem',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem',
                boxShadow: '0 0 15px rgba(0, 212, 255, 0.2)'
              }}
            >
              <div
                style={{
                  width: '42px',
                  height: '42px',
                  borderRadius: '8px',
                  backgroundColor: 'var(--cyan)',
                  color: 'var(--bg-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 900,
                  fontFamily: 'var(--font-heading)',
                  flexShrink: 0
                }}
              >
                04
              </div>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '1rem', color: '#00d4ff', marginBottom: '0.2rem' }}>
                  NumPy Grad-CAM &amp; Secure Verification Response
                </h3>
                <p style={{ fontSize: '0.825rem', color: '#ffffff', margin: 0 }}>
                  Computes ReLU(sum(w_k * A_k)) heatmap overlay and returns calibrated probability to client.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: METRICS TABLE FROM STATIC JSON */}
      <section style={{ marginBottom: '3.5rem' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <span className="badge-cyan" style={{ fontSize: '0.75rem', marginBottom: '0.5rem' }}>
            Empirical Evaluation
          </span>
          <h2 style={{ fontSize: '1.5rem', fontFamily: 'var(--font-heading)', color: '#ffffff' }}>
            Test Split Performance Metrics
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Evaluated on the held-out patient test split (N = {metrics.sample_count} images, 564 patient UIDs) from <code style={{ color: '#00d4ff' }}>reports/metrics.json</code>.
          </p>
        </div>

        {/* Key KPI Cards */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '1rem',
            marginBottom: '1.5rem'
          }}
        >
          <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ROC-AUC</div>
            <div style={{ fontSize: typeof metrics.roc_auc === 'number' ? '2rem' : '1rem', fontWeight: 900, color: '#00d4ff', fontFamily: 'var(--font-heading)' }}>
              {typeof metrics.roc_auc === 'number' ? Number(metrics.roc_auc).toFixed(4) : metrics.roc_auc}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Full discrimination capacity</div>
          </div>

          <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TEST ACCURACY</div>
            <div style={{ fontSize: typeof metrics.accuracy === 'number' ? '2rem' : '1rem', fontWeight: 900, color: '#ffffff', fontFamily: 'var(--font-heading)' }}>
              {typeof metrics.accuracy === 'number' ? `${(metrics.accuracy * 100).toFixed(2)}%` : metrics.accuracy}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Threshold = 0.50</div>
          </div>

          <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>F1-SCORE</div>
            <div style={{ fontSize: typeof metrics.f1_score === 'number' ? '2rem' : '1rem', fontWeight: 900, color: '#ffffff', fontFamily: 'var(--font-heading)' }}>
              {typeof metrics.f1_score === 'number' ? metrics.f1_score.toFixed(4) : metrics.f1_score}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Balanced precision &amp; recall</div>
          </div>

          <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>SENS. @ 90% SPEC.</div>
            <div style={{ fontSize: typeof metrics.sensitivity_at_90_specificity === 'number' ? '2rem' : '1rem', fontWeight: 900, color: '#10b981', fontFamily: 'var(--font-heading)' }}>
              {typeof metrics.sensitivity_at_90_specificity === 'number' ? Number(metrics.sensitivity_at_90_specificity).toFixed(4) : metrics.sensitivity_at_90_specificity}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Threshold: {metrics.threshold_at_90_specificity}
            </div>
          </div>
        </div>

        {/* Detailed Metrics Table */}
        <div className="card" style={{ padding: 0, overflow: 'hidden', marginBottom: '1.5rem' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
              <thead>
                <tr style={{ backgroundColor: 'rgba(4, 20, 40, 0.9)', borderBottom: '1px solid var(--border)' }}>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Metric</th>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Recorded Value</th>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Clinical Significance</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid rgba(13, 45, 74, 0.5)' }}>
                  <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600 }}>Evaluation Split</td>
                  <td style={{ padding: '0.85rem 1.25rem', color: '#00d4ff' }}>{metrics.evaluation_split.toUpperCase()} ({metrics.sample_count} images)</td>
                  <td style={{ padding: '0.85rem 1.25rem', color: 'var(--text-secondary)' }}>Strict patient-level GroupShuffleSplit without UID leakage</td>
                </tr>
                <tr style={{ borderBottom: '1px solid rgba(13, 45, 74, 0.5)' }}>
                  <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600 }}>Uncalibrated Brier Score</td>
                  <td style={{ padding: '0.85rem 1.25rem', color: '#ffffff' }}>{metrics.brier_score_uncalibrated}</td>
                  <td style={{ padding: '0.85rem 1.25rem', color: 'var(--text-secondary)' }}>Mean squared error before Platt scaling</td>
                </tr>
                <tr style={{ borderBottom: '1px solid rgba(13, 45, 74, 0.5)' }}>
                  <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600 }}>Calibrated Brier Score</td>
                  <td style={{ padding: '0.85rem 1.25rem', color: '#10b981', fontWeight: 700 }}>{metrics.brier_score_calibrated}</td>
                  <td style={{ padding: '0.85rem 1.25rem', color: 'var(--text-secondary)' }}>Reduction in probability error post-Platt calibration</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Confusion Matrix Breakdown */}
        <div className="card" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontFamily: 'var(--font-heading)', marginBottom: '1rem' }}>
            Test Split Confusion Matrix (N = 1,095)
          </h3>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '1rem'
            }}
          >
            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TRUE NEGATIVES (TN)</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--success)' }}>
                {metrics.confusion_matrix ? metrics.confusion_matrix[0][0] : 'PENDING'}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Correctly predicted Normal</div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FALSE POSITIVES (FP)</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--danger)' }}>
                {metrics.confusion_matrix ? metrics.confusion_matrix[0][1] : 'PENDING'}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Normal falsely called Abnormal</div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FALSE NEGATIVES (FN)</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--danger)' }}>
                {metrics.confusion_matrix ? metrics.confusion_matrix[1][0] : 'PENDING'}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Abnormal missed as Normal</div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TRUE POSITIVES (TP)</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--success)' }}>
                {metrics.confusion_matrix ? metrics.confusion_matrix[1][1] : 'PENDING'}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Correctly predicted Abnormal</div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 3: TEXT VS IMAGE VS FUSED COMPARISON */}
      <section style={{ marginBottom: '3.5rem' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <span className="badge-cyan" style={{ fontSize: '0.75rem', marginBottom: '0.5rem' }}>
            Modality Comparison
          </span>
          <h2 style={{ fontSize: '1.5rem', fontFamily: 'var(--font-heading)', color: '#ffffff' }}>
            Text vs Image vs Fused Baselines
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Comparison documented in <code style={{ color: '#00d4ff' }}>reports/baselines.md</code>.
          </p>
        </div>

        <div className="card" style={{ padding: 0, overflow: 'hidden', marginBottom: '1.5rem' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
              <thead>
                <tr style={{ backgroundColor: 'rgba(4, 20, 40, 0.9)', borderBottom: '1px solid var(--border)' }}>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Modality Model</th>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Feature Extraction</th>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Accuracy</th>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>F1-Score</th>
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {baselineComparisons.map((row) => (
                  <tr
                    key={row.modality}
                    style={{
                      borderBottom: '1px solid rgba(13, 45, 74, 0.5)',
                      backgroundColor: row.modality.includes('Served') ? 'rgba(0, 212, 255, 0.08)' : 'transparent'
                    }}
                  >
                    <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600, color: row.modality.includes('Served') ? '#00d4ff' : '#ffffff' }}>
                      {row.modality}
                    </td>
                    <td style={{ padding: '0.85rem 1.25rem', color: 'var(--text-secondary)' }}>{row.features}</td>
                    <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600 }}>{row.accuracy}</td>
                    <td style={{ padding: '0.85rem 1.25rem' }}>{row.f1}</td>
                    <td style={{ padding: '0.85rem 1.25rem', color: row.modality.includes('Served') ? '#00d4ff' : '#ffffff', fontWeight: 700 }}>
                      {row.rocAuc}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Narrative Callout: Why Text Baseline is High vs Why Served Model is Image-Only */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1.25rem'
          }}
        >
          <div className="card" style={{ padding: '1.25rem' }}>
            <h3 style={{ fontSize: '1rem', color: '#00d4ff', fontFamily: 'var(--font-heading)', marginBottom: '0.5rem' }}>
              Why is the Text-Only Score High?
            </h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
              Indiana University disease labels (MeSH / Problems) are extracted directly from the written radiologist reports. Vocabulary in the findings text shares direct semantic correlation with target labels. Evaluating on text from the same narrative naturally yields high performance due to lexical circularity.
            </p>
          </div>

          <div className="card" style={{ padding: '1.25rem' }}>
            <h3 style={{ fontSize: '1rem', color: '#00d4ff', fontFamily: 'var(--font-heading)', marginBottom: '0.5rem' }}>
              Why is the Served Model Image-Only?
            </h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
              In real clinical practice, an official radiology report does not exist when a radiograph is acquired. An automated decision support system must analyze raw pixel data at ingestion. Requiring a report would defeat the purpose of an AI assistant, as a human radiologist would have already completed the reading.
            </p>
          </div>
        </div>
      </section>

      {/* SECTION 4: LIMITATIONS & FAILURE MODES */}
      <section>
        <div style={{ marginBottom: '1.5rem' }}>
          <span className="badge-cyan" style={{ fontSize: '0.75rem', marginBottom: '0.5rem' }}>
            Safety &amp; Transparency
          </span>
          <h2 style={{ fontSize: '1.5rem', fontFamily: 'var(--font-heading)', color: '#ffffff' }}>
            Limitations &amp; Known Failure Modes
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Documented in compliance with <code style={{ color: '#00d4ff' }}>MODEL_CARD.md</code>.
          </p>
        </div>

        {/* Limitations Grid */}
        <h3 style={{ fontSize: '1.1rem', fontFamily: 'var(--font-heading)', marginBottom: '0.75rem', color: '#ffffff' }}>
          1. Systemic Limitations
        </h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '1rem',
            marginBottom: '2rem'
          }}
        >
          {limitations.map((item) => (
            <div key={item.title} className="card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <AlertTriangle size={18} color="#f59e0b" />
                <h4 style={{ fontSize: '0.925rem', color: '#ffffff', margin: 0 }}>{item.title}</h4>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
                {item.desc}
              </p>
            </div>
          ))}
        </div>

        {/* Failure Modes Grid */}
        <h3 style={{ fontSize: '1.1rem', fontFamily: 'var(--font-heading)', marginBottom: '0.75rem', color: '#ffffff' }}>
          2. Known Edge Cases &amp; Failure Modes
        </h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '1rem'
          }}
        >
          {failureModes.map((item) => (
            <div key={item.title} className="card" style={{ padding: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <ShieldAlert size={18} color="#ef4444" />
                <h4 style={{ fontSize: '0.925rem', color: '#ffffff', margin: 0 }}>{item.title}</h4>
              </div>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
                {item.desc}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default ModelPage;
