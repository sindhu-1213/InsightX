import React, { useEffect } from 'react';
import metricsData from '../data/metrics.json';
import {
  Brain,
  ShieldAlert,
  AlertTriangle,
  Database,
  BarChart3,
  Layers,
  Sliders
} from 'lucide-react';

function ModelPage() {
  const metrics = metricsData;

  useEffect(() => {
    document.title = 'InsightX | Model Card & Verification';
  }, []);

  const baselineComparisons = [
    {
      modality: 'Majority Class Baseline',
      features: 'Prevalence (Always Abnormal)',
      accuracy: '60.18%',
      f1: '0.7514',
      rocAuc: '0.5000',
      role: 'Class prevalence baseline'
    },
    {
      modality: 'Frozen MobileNetV2 GAP',
      features: 'Image Only (1,280 GAP + Logistic Regression, C=0.01)',
      accuracy: '64.84%',
      f1: '0.6980',
      rocAuc: '0.7182',
      role: 'Frozen transfer baseline'
    },
    {
      modality: 'Text-Only (Findings TF-IDF)',
      features: 'Text Only (1,000 TF-IDF features, C=0.01)',
      accuracy: '90.68%',
      f1: '0.9257',
      rocAuc: '0.9676',
      role: 'Report label leakage baseline'
    },
    {
      modality: 'Multimodal Fused',
      features: 'Image + Text Concatenation (C=0.01)',
      accuracy: '89.50%',
      f1: '0.9119',
      rocAuc: '0.9629',
      role: 'Multimodal benchmark'
    },
    {
      modality: 'Fine-Tuned MobileNetV2 (Served Model)',
      features: 'Image Only (Fine-tuned + Platt Scaling)',
      accuracy: '66.48%',
      f1: '0.6697',
      rocAuc: '0.7563',
      role: 'Active image-only production model'
    }
  ];

  const limitations = [
    {
      title: 'Operating Point Sensitivity Limitation',
      desc: `At the validation-selected default operating threshold (${metrics.threshold_youden ?? 0.6522}), the model achieves a sensitivity of ${(metrics.sensitivity * 100).toFixed(2)}% (0.5645) and specificity of ${(metrics.specificity * 100).toFixed(2)}% (0.8165). Approximately 43.55% of abnormal radiographs in the test split are not flagged under default settings. At the 90% specificity threshold (${metrics.threshold_at_90_specificity ?? 0.7291}), test sensitivity is ${(metrics.sensitivity_at_90_specificity * 100).toFixed(2)}% (0.4537).`
    },
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
      desc: 'The system has not been evaluated in prospective clinical trials, reader studies, or emergency settings. It is an investigational research demonstration and not a certified medical device.'
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

      {/* PROMINENT LIMITATIONS CALLOUT: SENSITIVITY AT OPERATING POINT */}
      <div
        className="card"
        style={{
          backgroundColor: 'rgba(239, 68, 68, 0.08)',
          border: '1px solid rgba(239, 68, 68, 0.4)',
          marginBottom: '3rem',
          padding: '1.5rem'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <AlertTriangle size={22} color="#ef4444" />
          <h3 style={{ fontSize: '1.1rem', color: '#ef4444', margin: 0, fontFamily: 'var(--font-heading)' }}>
            Operating Point Sensitivity Notice
          </h3>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.6', margin: 0 }}>
          At the validation-chosen decision threshold (<strong>{metrics.threshold_youden}</strong>), test sensitivity is <strong>{(metrics.sensitivity * 100).toFixed(2)}%</strong> (0.5645) and specificity is <strong>{(metrics.specificity * 100).toFixed(2)}%</strong> (0.8165). This means <strong>43.55% of abnormal radiographs</strong> in the test cohort are not flagged as abnormal at this operating point. At the 90% specificity threshold (<strong>{metrics.threshold_at_90_specificity}</strong>), sensitivity drops to <strong>{(metrics.sensitivity_at_90_specificity * 100).toFixed(2)}%</strong> (0.4537). This research prototype must not be used for patient management.
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

            <div style={{ textAlign: 'center', color: '#00d4ff', fontSize: '1.2rem' }}>&darr;</div>

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
                  Dual-Output ONNX Inference
                </h3>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: 0 }}>
                  Evaluated with ONNX Runtime on CPU (<code style={{ color: '#00d4ff' }}>insightx_gradcam.onnx</code>, 8.50 MB), outputting linear logit and (7, 7, 1280) conv feature map.
                </p>
              </div>
            </div>

            <div style={{ textAlign: 'center', color: '#00d4ff', fontSize: '1.2rem' }}>&darr;</div>

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
                  Validation-Fitted Platt Probability Scaling
                </h3>
                <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: 0 }}>
                  Transforms log-odds into calibrated empirical probabilities via logistic regression fitted on the validation partition.
                </p>
              </div>
            </div>

            <div style={{ textAlign: 'center', color: '#00d4ff', fontSize: '1.2rem' }}>&darr;</div>

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
                  NumPy Grad-CAM &amp; Response Serialization
                </h3>
                <p style={{ fontSize: '0.825rem', color: '#ffffff', margin: 0 }}>
                  Computes ReLU(sum(w_k * A_k)) heatmap overlay and returns calibrated probability to client without TensorFlow dependencies.
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
            Evaluated on held-out patient test split (N = {metrics.sample_count} images, 564 patient UIDs) from <code style={{ color: '#00d4ff' }}>reports/metrics.json</code>.
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
            <div style={{ fontSize: '2rem', fontWeight: 900, color: '#00d4ff', fontFamily: 'var(--font-heading)' }}>
              {Number(metrics.roc_auc).toFixed(4)}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              95% CI: [{metrics.roc_auc_ci_95 ? metrics.roc_auc_ci_95.join(', ') : '0.7180, 0.7921'}]
            </div>
          </div>

          <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TEST ACCURACY</div>
            <div style={{ fontSize: '2rem', fontWeight: 900, color: '#ffffff', fontFamily: 'var(--font-heading)' }}>
              {(metrics.accuracy * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              @ Youden threshold {metrics.threshold_youden}
            </div>
          </div>

          <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>F1-SCORE</div>
            <div style={{ fontSize: '2rem', fontWeight: 900, color: '#ffffff', fontFamily: 'var(--font-heading)' }}>
              {Number(metrics.f1_score).toFixed(4)}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Abnormal class positive</div>
          </div>

          <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>SENSITIVITY (@ YOUDEN)</div>
            <div style={{ fontSize: '2rem', fontWeight: 900, color: '#f59e0b', fontFamily: 'var(--font-heading)' }}>
              {(metrics.sensitivity * 100).toFixed(2)}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
              Specificity: {(metrics.specificity * 100).toFixed(2)}%
            </div>
          </div>

          <div className="card" style={{ padding: '1.25rem', textAlign: 'center' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>SENS. @ 90% SPEC.</div>
            <div style={{ fontSize: '2rem', fontWeight: 900, color: '#10b981', fontFamily: 'var(--font-heading)' }}>
              {(metrics.sensitivity_at_90_specificity * 100).toFixed(2)}%
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
                  <th style={{ padding: '0.85rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Technical Definition</th>
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
                  <td style={{ padding: '0.85rem 1.25rem', color: 'var(--text-secondary)' }}>Mean squared error of raw neural network probabilities</td>
                </tr>
                <tr style={{ borderBottom: '1px solid rgba(13, 45, 74, 0.5)' }}>
                  <td style={{ padding: '0.85rem 1.25rem', fontWeight: 600 }}>Calibrated Brier Score</td>
                  <td style={{ padding: '0.85rem 1.25rem', color: '#10b981', fontWeight: 700 }}>{metrics.brier_score_calibrated}</td>
                  <td style={{ padding: '0.85rem 1.25rem', color: 'var(--text-secondary)' }}>Reduction in probability error post-Platt scaling</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Confusion Matrix Breakdown */}
        <div className="card" style={{ padding: '1.5rem', marginBottom: '2.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontFamily: 'var(--font-heading)', marginBottom: '1rem' }}>
            Test Split Confusion Matrix (N = {metrics.sample_count}, Threshold = {metrics.threshold_youden})
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
                {metrics.confusion_matrix[0][0]}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Correctly predicted Normal</div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FALSE POSITIVES (FP)</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--danger)' }}>
                {metrics.confusion_matrix[0][1]}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Normal falsely flagged Abnormal</div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FALSE NEGATIVES (FN)</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--danger)' }}>
                {metrics.confusion_matrix[1][0]}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Abnormal missed as Normal</div>
            </div>

            <div style={{ backgroundColor: 'var(--bg-input)', padding: '1rem', borderRadius: '6px' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TRUE POSITIVES (TP)</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--success)' }}>
                {metrics.confusion_matrix[1][1]}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Correctly predicted Abnormal</div>
            </div>
          </div>
        </div>

        {/* SECTION: PROJECTION BREAKDOWN (PER-VIEW) */}
        {metrics.breakdown_by_projection && (
          <div style={{ marginBottom: '2.5rem' }}>
            <h3 style={{ fontSize: '1.15rem', fontFamily: 'var(--font-heading)', color: '#ffffff', marginBottom: '0.75rem' }}>
              Results Breakdown by Projection (View)
            </h3>
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                <thead>
                  <tr style={{ backgroundColor: 'rgba(4, 20, 40, 0.9)', borderBottom: '1px solid var(--border)' }}>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Projection</th>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Total Images</th>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Normal Count</th>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Abnormal Count</th>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>ROC-AUC</th>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>95% Bootstrap CI</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.breakdown_by_projection.map((row) => (
                    <tr key={row.projection} style={{ borderBottom: '1px solid rgba(13, 45, 74, 0.5)' }}>
                      <td style={{ padding: '0.75rem 1.25rem', fontWeight: 600, color: '#00d4ff' }}>{row.projection}</td>
                      <td style={{ padding: '0.75rem 1.25rem' }}>{row.total_count}</td>
                      <td style={{ padding: '0.75rem 1.25rem' }}>{row.normal_count}</td>
                      <td style={{ padding: '0.75rem 1.25rem' }}>{row.abnormal_count}</td>
                      <td style={{ padding: '0.75rem 1.25rem', fontWeight: 700, color: '#ffffff' }}>{row.roc_auc.toFixed(4)}</td>
                      <td style={{ padding: '0.75rem 1.25rem', color: 'var(--text-secondary)' }}>[{row.ci_95.join(', ')}]</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* SECTION: FINDING BREAKDOWN */}
        {metrics.breakdown_by_finding && (
          <div style={{ marginBottom: '2.5rem' }}>
            <h3 style={{ fontSize: '1.15rem', fontFamily: 'var(--font-heading)', color: '#ffffff', marginBottom: '0.75rem' }}>
              Results Breakdown by Top 10 Finding Tokens vs Normal
            </h3>
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                <thead>
                  <tr style={{ backgroundColor: 'rgba(4, 20, 40, 0.9)', borderBottom: '1px solid var(--border)' }}>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Finding Token</th>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Abnormal Test Images</th>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>ROC-AUC vs Normal</th>
                    <th style={{ padding: '0.75rem 1.25rem', textAlign: 'left', color: '#ffffff' }}>Reliability Status</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.breakdown_by_finding.map((row) => (
                    <tr key={row.finding} style={{ borderBottom: '1px solid rgba(13, 45, 74, 0.5)' }}>
                      <td style={{ padding: '0.75rem 1.25rem', fontWeight: 600, color: '#ffffff' }}>{row.finding}</td>
                      <td style={{ padding: '0.75rem 1.25rem' }}>{row.count}</td>
                      <td style={{ padding: '0.75rem 1.25rem', fontWeight: 700, color: '#00d4ff' }}>{row.roc_auc.toFixed(4)}</td>
                      <td style={{ padding: '0.75rem 1.25rem', color: row.trust_flag.includes('Sufficient') ? '#10b981' : '#f59e0b' }}>
                        {row.trust_flag}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
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
            Held-out test split comparison against frozen baseline models as documented in <code style={{ color: '#00d4ff' }}>reports/comparison.md</code>.
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

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1.25rem'
          }}
        >
          <div className="card" style={{ padding: '1.25rem' }}>
            <h3 style={{ fontSize: '1rem', color: '#00d4ff', fontFamily: 'var(--font-heading)', marginBottom: '0.5rem' }}>
              Why is the Text Baseline So High?
            </h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
              Indiana University disease labels (Problems / MeSH) were generated directly by cataloging the text radiology reports. The findings vocabulary contains explicit semantic overlap with the assigned abnormality categories. Evaluating a text model against labels extracted from that exact narrative naturally yields high scores due to lexical circularity.
            </p>
          </div>

          <div className="card" style={{ padding: '1.25rem' }}>
            <h3 style={{ fontSize: '1rem', color: '#00d4ff', fontFamily: 'var(--font-heading)', marginBottom: '0.5rem' }}>
              Why is the Served Model Image-Only?
            </h3>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
              In real clinical practice, an official radiology report does not exist when a radiograph is acquired. An automated decision support system must analyze raw pixel data at ingestion. Requiring an authored report would render the AI redundant, as a human clinician would already have completed the reading.
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
