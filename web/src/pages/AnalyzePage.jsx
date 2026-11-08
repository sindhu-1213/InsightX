import { useState, useEffect, useRef, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { collection, addDoc, serverTimestamp } from 'firebase/firestore';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine
} from 'recharts';
import { db } from '../firebase';
import { useAuth } from '../context/AuthContext';
import { buildAnalysisRecord } from '../utils/payloadBuilders';
import { inferenceAdapter } from '../inference/adapter';
import {
  UploadCloud,
  Camera,
  Sparkles,
  AlertTriangle,
  FileImage,
  RefreshCw,
  Eye,
  ShieldCheck,
  CheckCircle2,
  Sliders,
  Info,
  Cpu
} from 'lucide-react';

// Pre-indexed demo samples for instant guest testing without login
const DEFAULT_SAMPLES = [
  {
    id: 'sample_01_normal.png',
    title: 'Sample 1: Normal Baseline',
    desc: 'Frontal radiograph showing clear lung fields and normal cardiothoracic ratio.',
    expectedLabel: 'Normal',
    url: '/samples/sample_01_normal.png'
  },
  {
    id: 'sample_02_normal.png',
    title: 'Sample 2: Normal Projection',
    desc: 'Unremarkable pulmonary parenchyma without focal consolidation or effusion.',
    expectedLabel: 'Normal',
    url: '/samples/sample_02_normal.png'
  },
  {
    id: 'sample_03_normal.png',
    title: 'Sample 3: Normal Silhouette',
    desc: 'Indiana University radiograph without focal consolidation or effusion.',
    expectedLabel: 'Normal',
    url: '/samples/sample_03_normal.png'
  },
  {
    id: 'sample_04_abnormal.png',
    title: 'Sample 4: Focal Granuloma Case',
    desc: 'Localized opacity and dense parenchymal markings in the lung field.',
    expectedLabel: 'Abnormal',
    url: '/samples/sample_04_abnormal.png'
  }
];

/**
 * Resize image client-side to maximum 1024px while preserving aspect ratio.
 */
function resizeImageIfNeeded(file, maxDimension = 1024) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const reader = new FileReader();

    reader.onload = (e) => {
      img.src = e.target.result;
    };
    reader.onerror = () => reject(new Error('Failed to read file from storage.'));
    img.onerror = () => reject(new Error('The selected file is not a valid or readable image.'));

    img.onload = () => {
      const origWidth = img.naturalWidth || img.width;
      const origHeight = img.naturalHeight || img.height;

      if (!origWidth || !origHeight || origWidth <= 0 || origHeight <= 0) {
        reject(new Error('Invalid image dimensions'));
        return;
      }

      if (origWidth <= maxDimension && origHeight <= maxDimension) {
        resolve({
          file,
          previewUrl: img.src,
          wasResized: false,
          origDims: `${origWidth}×${origHeight}`,
          finalDims: `${origWidth}×${origHeight}`
        });
        return;
      }

      let targetWidth = origWidth;
      let targetHeight = origHeight;

      if (origWidth > origHeight) {
        targetHeight = Math.max(1, Math.round((origHeight * maxDimension) / origWidth));
        targetWidth = maxDimension;
      } else {
        targetWidth = Math.max(1, Math.round((origWidth * maxDimension) / origHeight));
        targetHeight = maxDimension;
      }

      const canvas = document.createElement('canvas');
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            resolve({
              file,
              previewUrl: img.src,
              wasResized: false,
              origDims: `${origWidth}×${origHeight}`,
              finalDims: `${origWidth}×${origHeight}`
            });
            return;
          }

          const resizedFile = new File([blob], file.name.replace(/\.[^/.]+$/, '.jpg'), {
            type: 'image/jpeg',
            lastModified: Date.now()
          });

          resolve({
            file: resizedFile,
            previewUrl: canvas.toDataURL('image/jpeg', 0.92),
            wasResized: true,
            origDims: `${origWidth}×${origHeight}`,
            finalDims: `${targetWidth}×${targetHeight}`
          });
        },
        'image/jpeg',
        0.92
      );
    };

    img.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function AnalyzePage() {
  const { currentUser, userProfile } = useAuth();

  // Model Engine State
  const [modelStatus, setModelStatus] = useState('unloaded'); // 'unloaded' | 'loading' | 'ready' | 'error'
  const [modelProgress, setModelProgress] = useState(0);
  const [modelStatusText, setModelStatusText] = useState('');
  const [modelError, setModelError] = useState(null);

  // Upload & File State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resizeMeta, setResizeMeta] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Gallery & Samples State
  const [samples, setSamples] = useState(DEFAULT_SAMPLES);
  const [activeSampleId, setActiveSampleId] = useState(null);

  // Inference & Response State
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null); // { title: string, message: string, onRetry?: function } | null
  const [savedStatus, setSavedStatus] = useState(null);

  // Visual Overlay State
  const [overlayOpacity, setOverlayOpacity] = useState(65); // 0 to 100%
  const [viewMode, setViewMode] = useState('sideBySide'); // 'sideBySide' | 'blended'

  const fileInputRef = useRef(null);
  const cameraInputRef = useRef(null);

  const initModel = useCallback(async () => {
    try {
      setModelStatus('loading');
      setModelError(null);
      await inferenceAdapter.initialize((pct, statusText) => {
        setModelProgress(pct);
        setModelStatusText(statusText);
      });
      setModelStatus('ready');
    } catch (err) {
      console.error('[InsightX Engine Initialization Error]', {
        message: err.message,
        stack: err.stack,
        cause: err.cause
      });
      setModelStatus('error');

      const msg = (err.message || '').toLowerCase();
      const isWasmIssue =
        typeof WebAssembly === 'undefined' ||
        msg.includes('wasm') ||
        msg.includes('webassembly') ||
        msg.includes('simd') ||
        msg.includes('out of memory') ||
        msg.includes('aborted') ||
        msg.includes('no available backend');

      const isNetworkIssue =
        (typeof navigator !== 'undefined' && !navigator.onLine) ||
        msg.includes('failed to fetch') ||
        msg.includes('network') ||
        msg.includes('failed to download') ||
        msg.includes('failed to load') ||
        msg.includes('http') ||
        msg.includes('asset not found') ||
        msg.includes('spa fallback');

      if (isWasmIssue) {
        setModelError({
          type: 'wasm',
          title: 'WebAssembly Engine Error',
          message:
            'WebAssembly runtime failed to initialize. Your browser may lack WebAssembly SIMD support or have memory restrictions.'
        });
      } else if (isNetworkIssue) {
        setModelError({
          type: 'network',
          title: 'Network Download Failure',
          message:
            'Network connection failed while downloading neural model assets. Please check your connection and retry.'
        });
      } else {
        setModelError({
          type: 'runtime',
          title: 'Runtime Error',
          message: `In-browser neural engine encountered an initialization error: ${err.message || 'Unknown error'}. Please retry.`
        });
      }
    }
  }, []);

  useEffect(() => {
    document.title = 'InsightX | Radiograph Analysis Workbench';
    initModel();
  }, [initModel]);

  // Load sample manifest from local static assets
  useEffect(() => {
    inferenceAdapter
      .listSamples()
      .then((items) => {
        if (Array.isArray(items) && items.length > 0) {
          setSamples(items);
        }
      })
      .catch((err) => {
        console.warn('Failed to load sample catalog, using built-in defaults:', err);
      });
  }, []);

  const processIncomingFile = async (rawFile) => {
    if (!rawFile) return;

    const lowerName = rawFile.name.toLowerCase();
    const isHeic =
      lowerName.endsWith('.heic') ||
      lowerName.endsWith('.heif') ||
      rawFile.type === 'image/heic' ||
      rawFile.type === 'image/heif';

    const validExtensions = ['.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif'];
    const hasValidExt = validExtensions.some((ext) => lowerName.endsWith(ext));
    const validMimes = [
      'image/png',
      'image/jpeg',
      'image/jpg',
      'image/webp',
      'image/heic',
      'image/heif',
      ''
    ];

    if (!validMimes.includes(rawFile.type) && !hasValidExt) {
      setError({
        title: 'Unsupported File Format',
        message: 'Please provide a valid radiograph image (PNG, JPEG, WebP, or iPhone HEIC).',
        onRetry: () => fileInputRef.current?.click()
      });
      return;
    }

    if (rawFile.size > 20 * 1024 * 1024) {
      setError({
        title: 'File Too Large',
        message: 'Original image exceeds 20 MB. Please select a smaller radiograph file.',
        onRetry: () => fileInputRef.current?.click()
      });
      return;
    }

    setError(null);
    setResult(null);
    setSavedStatus(null);
    setActiveSampleId(null);

    let fileToProcess = rawFile;
    if (isHeic) {
      try {
        const heic2anyModule = await import('heic2any');
        const heic2any = heic2anyModule.default || heic2anyModule;
        const convertedBlob = await heic2any({
          blob: rawFile,
          toType: 'image/jpeg',
          quality: 0.92
        });
        const singleBlob = Array.isArray(convertedBlob) ? convertedBlob[0] : convertedBlob;
        fileToProcess = new File(
          [singleBlob],
          rawFile.name.replace(/\.(heic|heif)$/i, '.jpg'),
          { type: 'image/jpeg', lastModified: rawFile.lastModified || 0 }
        );
      } catch (conversionErr) {
        console.error('HEIC conversion failed:', conversionErr);
        setError({
          title: 'HEIC Conversion Failed',
          message:
            'Failed to convert iPhone HEIC photo. Please take a screenshot, capture in Most Compatible (JPEG) format, or upload a PNG/JPEG scan.',
          onRetry: () => processIncomingFile(rawFile)
        });
        return;
      }
    }

    try {
      // Client-side resize to maximum 1024 px
      const resized = await resizeImageIfNeeded(fileToProcess, 1024);
      setSelectedFile(resized.file);
      setPreviewUrl(resized.previewUrl);
      setResizeMeta({
        wasResized: resized.wasResized,
        origDims: resized.origDims,
        finalDims: resized.finalDims
      });
    } catch (err) {
      console.error('Resize / decode error:', err);
      setError({
        title: 'Invalid or Corrupt Image',
        message:
          'The selected file is not a valid or readable image. Please choose a valid PNG or JPEG radiograph.',
        onRetry: () => fileInputRef.current?.click()
      });
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) processIncomingFile(file);
  };

  // Sample Selection: Works for guests and logged-in practitioners alike
  const handleSelectSample = async (sample) => {
    setError(null);
    setLoading(true);
    setActiveSampleId(sample.id);
    setSelectedFile(null);
    setResizeMeta(null);
    setSavedStatus(null);

    const sampleUrl = sample.url || `/samples/${sample.id}`;
    setPreviewUrl(sampleUrl);

    try {
      if (!inferenceAdapter.isReady()) {
        await initModel();
      }
      const data = await inferenceAdapter.predict(sampleUrl);
      setResult(data);

      // Save to Firestore audit log if practitioner is logged in with verified email
      if (currentUser && currentUser.emailVerified) {
        try {
          const analysesRef = collection(db, 'users', currentUser.uid, 'analyses');
          const record = buildAnalysisRecord({
            finding:
              data.finding ||
              (data.label === 'Abnormal' ? 'Possible abnormal finding' : 'No abnormal finding flagged'),
            probability: Number(data.calibrated_probability),
            modelVersion: data.model_version || '1.0.0',
            source: 'sample',
            createdAt: serverTimestamp()
          });
          await addDoc(analysesRef, record);
          setSavedStatus('Sample demonstration recorded to your practitioner audit log in Firestore.');
        } catch (firestoreErr) {
          console.warn('Firestore write note:', firestoreErr.message);
        }
      }
    } catch (err) {
      console.error('Sample inference error:', err);
      setError({
        title: 'Sample Analysis Error',
        message: err.message || 'Failed to analyze sample radiograph.',
        onRetry: () => handleSelectSample(sample)
      });
    } finally {
      setLoading(false);
    }
  };

  // Custom Upload Analysis: Runs in browser for guests and practitioners
  const handleUploadPredict = async () => {
    if (!selectedFile) {
      setError({
        title: 'No Radiograph Selected',
        message: 'Please choose or capture a chest radiograph first.',
        onRetry: () => fileInputRef.current?.click()
      });
      return;
    }

    setLoading(true);
    setError(null);
    setSavedStatus(null);

    try {
      if (!inferenceAdapter.isReady()) {
        await initModel();
      }
      const data = await inferenceAdapter.predict(selectedFile);
      setResult(data);

      // If practitioner is signed in with verified email, save to Firestore
      if (currentUser && currentUser.emailVerified) {
        try {
          const analysesRef = collection(db, 'users', currentUser.uid, 'analyses');
          const record = buildAnalysisRecord({
            finding:
              data.finding ||
              (data.label === 'Abnormal' ? 'Possible abnormal finding' : 'No abnormal finding flagged'),
            probability: Number(data.calibrated_probability),
            modelVersion: data.model_version || '1.0.0',
            source: 'upload',
            createdAt: serverTimestamp()
          });
          await addDoc(analysesRef, record);
          setSavedStatus('Analysis recorded to your practitioner audit log in Firestore.');
        } catch (firestoreErr) {
          console.warn('Firestore write note:', firestoreErr.message);
        }
      }
    } catch (err) {
      console.error('Inference error:', err);
      setError({
        title: 'In-Browser Inference Error',
        message: err.message || 'Failed to analyze radiograph in browser.',
        onRetry: () => handleUploadPredict()
      });
    } finally {
      setLoading(false);
    }
  };

  // Recharts Probability Chart Data
  const chartData = result
    ? [
        {
          name: 'Calibrated Probability',
          probability: Number((result.calibrated_probability * 100).toFixed(1)),
          threshold: 50
        }
      ]
    : [];

  return (
    <div
      id="analyze-workbench"
      className="container"
      style={{
        padding: '2.5rem 1rem 5rem 1rem',
        maxWidth: '1240px'
      }}
    >
      {/* Page Heading */}
      <div style={{ textAlign: 'center', maxWidth: '820px', margin: '0 auto 2.5rem auto' }}>
        <span className="badge-cyan" style={{ marginBottom: '0.75rem' }}>
          <Sparkles size={14} /> In-Browser Neural Engine
        </span>
        <h1
          style={{
            fontSize: 'clamp(1.8rem, 5vw, 2.75rem)',
            marginBottom: '0.75rem',
            fontFamily: 'var(--font-heading)'
          }}
        >
          Chest Radiograph Analysis
        </h1>
        <p
          style={{
            color: 'var(--text-secondary)',
            fontSize: 'clamp(0.9rem, 2.5vw, 1.05rem)',
            lineHeight: '1.6'
          }}
        >
          Zero-backend on-device inference using WebAssembly SIMD. Evaluate radiographs locally with Platt
          probability calibration and spatial Grad-CAM attention heatmaps.
        </p>
      </div>

      {/* Model Engine Status Card */}
      <div
        id="model-status-card"
        style={{
          backgroundColor: 'rgba(4, 20, 40, 0.85)',
          border:
            modelStatus === 'error'
              ? '1px solid var(--danger)'
              : modelStatus === 'ready'
                ? '1px solid rgba(16, 185, 129, 0.35)'
                : '1px solid rgba(0, 212, 255, 0.35)',
          borderRadius: 'var(--radius-md)',
          padding: '1rem 1.25rem',
          marginBottom: '2rem',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
        }}
      >
        {modelStatus === 'loading' && (
          <div>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '0.5rem',
                fontSize: '0.85rem'
              }}
            >
              <span style={{ color: '#00d4ff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <RefreshCw size={15} className="spin" />
                <span>{modelStatusText || 'Initializing neural model...'}</span>
              </span>
              <span style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                {modelProgress}%
              </span>
            </div>
            <div
              style={{
                height: '6px',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderRadius: '3px',
                overflow: 'hidden'
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${modelProgress}%`,
                  backgroundColor: 'var(--cyan)',
                  transition: 'width 0.3s ease'
                }}
              />
            </div>
          </div>
        )}

        {modelStatus === 'ready' && (
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '0.75rem'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <CheckCircle2 size={18} color="var(--success)" />
              <div style={{ fontSize: '0.875rem', fontWeight: 600, color: '#ffffff' }}>
                In-Browser Neural Engine Ready
              </div>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                &bull; WebAssembly SIMD &bull; Single-Threaded
              </span>
            </div>
            <span
              style={{
                fontSize: '0.725rem',
                color: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                border: '1px solid rgba(0, 212, 255, 0.25)',
                padding: '0.2rem 0.55rem',
                borderRadius: '4px'
              }}
            >
              Cached in Browser
            </span>
          </div>
        )}

        {modelStatus === 'error' && (
          <div
            id="model-error-banner"
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '0.75rem'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--danger)' }}>
              <AlertTriangle size={18} />
              <span style={{ fontSize: '0.875rem' }}>
                {typeof modelError === 'object' && modelError !== null ? modelError.message : modelError}
              </span>
            </div>
            <button
              type="button"
              id="retry-model-init-btn"
              onClick={initModel}
              className="btn-secondary btn-sm"
              style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
            >
              <RefreshCw size={13} /> Retry Initialization
            </button>
          </div>
        )}

        {modelStatus === 'unloaded' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            <Cpu size={16} /> Preparing on-device inference runtime...
          </div>
        )}
      </div>

      {/* Practitioner Status Bar */}
      {currentUser && (
        <div
          style={{
            backgroundColor: 'rgba(10, 31, 53, 0.75)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            padding: '0.9rem 1.25rem',
            marginBottom: '2rem',
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: '0.75rem'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <div
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                backgroundColor: 'rgba(0, 212, 255, 0.15)',
                border: '1px solid var(--cyan)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#00d4ff'
              }}
            >
              <ShieldCheck size={18} />
            </div>
            <div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#ffffff' }}>
                {currentUser.displayName || 'Verified Practitioner'} &bull; {userProfile?.role || 'Doctor'}
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                {currentUser.email} &bull; {userProfile?.institution || 'Independent Clinic'}
              </div>
            </div>
          </div>

          <Link to="/history" className="btn-secondary btn-sm" style={{ fontSize: '0.8rem' }}>
            Audit History &rarr;
          </Link>
        </div>
      )}

      {/* Sample Radiograph Gallery */}
      <section
        id="sample-gallery-section"
        style={{
          marginBottom: '2.5rem',
          backgroundColor: 'rgba(4, 20, 40, 0.85)',
          border: '1px solid rgba(0, 212, 255, 0.25)',
          borderRadius: 'var(--radius-md)',
          padding: '1.5rem',
          boxShadow: '0 8px 30px rgba(0, 0, 0, 0.4)'
        }}
      >
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1.25rem',
            gap: '0.75rem'
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="badge-cyan" style={{ fontSize: '0.7rem' }}>On-Device Demo</span>
              <h2
                style={{
                  fontSize: '1.15rem',
                  fontFamily: 'var(--font-heading)',
                  color: '#ffffff',
                  margin: 0
                }}
              >
                Sample Radiograph Gallery
              </h2>
            </div>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', margin: '0.25rem 0 0 0' }}>
              Instant on-device evaluation of pre-indexed Indiana University scans. Select any case to analyze locally.
            </p>
          </div>
          <span style={{ fontSize: '0.75rem', color: '#00d4ff', fontFamily: 'var(--font-heading)' }}>
            Zero Server Calls
          </span>
        </div>

        {/* Gallery Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '1rem'
          }}
        >
          {samples.map((s) => {
            const isSelected = activeSampleId === s.id;
            return (
              <div
                key={s.id}
                onClick={() => !loading && handleSelectSample(s)}
                style={{
                  backgroundColor: isSelected ? 'rgba(0, 212, 255, 0.12)' : 'var(--bg-card)',
                  border: isSelected ? '1px solid var(--cyan)' : '1px solid var(--border)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '1rem',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s ease',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between'
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) e.currentTarget.style.borderColor = 'rgba(0, 212, 255, 0.5)';
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) e.currentTarget.style.borderColor = 'var(--border)';
                }}
              >
                <div>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '0.5rem'
                    }}
                  >
                    <span
                      style={{
                        fontSize: '0.7rem',
                        fontWeight: 700,
                        color: s.expectedLabel === 'Normal' ? 'var(--success)' : 'var(--danger)',
                        backgroundColor:
                          s.expectedLabel === 'Normal'
                            ? 'rgba(16, 185, 129, 0.12)'
                            : 'rgba(239, 68, 68, 0.12)',
                        padding: '0.15rem 0.5rem',
                        borderRadius: '4px'
                      }}
                    >
                      {s.expectedLabel}
                    </span>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                      IU-XRAY
                    </span>
                  </div>

                  <h3 style={{ fontSize: '0.9rem', color: '#ffffff', marginBottom: '0.35rem' }}>
                    {s.title}
                  </h3>
                  <p style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', lineHeight: '1.4', margin: 0 }}>
                    {s.desc}
                  </p>
                </div>

                <button
                  type="button"
                  disabled={loading}
                  style={{
                    marginTop: '0.85rem',
                    width: '100%',
                    padding: '0.45rem',
                    backgroundColor: isSelected ? 'var(--cyan)' : 'rgba(0, 212, 255, 0.1)',
                    color: isSelected ? 'var(--bg-primary)' : '#00d4ff',
                    border: '1px solid var(--cyan)',
                    borderRadius: '4px',
                    fontSize: '0.775rem',
                    fontWeight: 700,
                    fontFamily: 'var(--font-heading)',
                    cursor: loading ? 'not-allowed' : 'pointer'
                  }}
                >
                  {isSelected && loading ? 'Inferring in WASM...' : isSelected ? 'Loaded in Viewer' : 'Try This Sample'}
                </button>
              </div>
            );
          })}
        </div>
      </section>

      {/* Global Error Banner with Retry */}
      {error && (
        <div
          id="analyze-error-alert"
          style={{
            backgroundColor: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid var(--danger)',
            borderRadius: 'var(--radius-sm)',
            padding: '1rem 1.25rem',
            marginBottom: '1.75rem',
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '0.85rem',
            color: '#ffffff'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <AlertTriangle size={22} color="#ef4444" style={{ flexShrink: 0 }} />
            <div>
              {error.title && (
                <div style={{ fontWeight: 700, fontSize: '0.9rem', color: '#ef4444' }}>
                  {error.title}
                </div>
              )}
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                {error.message || String(error)}
              </div>
            </div>
          </div>
          {error.onRetry && (
            <button
              type="button"
              onClick={error.onRetry}
              className="btn-secondary btn-sm"
              style={{
                flexShrink: 0,
                padding: '0.35rem 0.75rem',
                fontSize: '0.8rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem'
              }}
            >
              <RefreshCw size={13} /> Retry
            </button>
          )}
        </div>
      )}

      {/* Main Two-Column Layout */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '2rem',
          alignItems: 'start'
        }}
      >
        {/* LEFT COLUMN: Upload, Camera & Controls */}
        <div className="card">
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1.25rem'
            }}
          >
            <h2 style={{ fontSize: '1.15rem', fontFamily: 'var(--font-heading)' }}>
              Upload Custom Radiograph
            </h2>
            <span style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>
              100% Client-Side
            </span>
          </div>

          {/* Drag & Drop / Tap Target */}
          <div
            id="drag-drop-zone"
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragOver(false);
              const file = e.dataTransfer.files?.[0];
              if (file) processIncomingFile(file);
            }}
            onClick={() => fileInputRef.current?.click()}
            style={{
              padding: '2.5rem 1.25rem',
              backgroundColor: isDragOver ? 'rgba(0, 212, 255, 0.1)' : 'var(--bg-input)',
              border: isDragOver ? '2px dashed var(--cyan)' : '2px dashed var(--border)',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              textAlign: 'center',
              transition: 'all 0.2s ease',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.75rem'
            }}
          >
            <div
              style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                border: '1px solid var(--cyan)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#00d4ff'
              }}
            >
              <UploadCloud size={28} />
            </div>

            <div>
              <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#ffffff' }}>
                Tap to upload or drag radiograph here
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                Photo library, DICOM-exported PNG, or JPEG
              </div>
            </div>

            <span
              style={{
                fontSize: '0.725rem',
                color: 'var(--text-muted)',
                backgroundColor: 'rgba(4, 20, 40, 0.8)',
                padding: '0.2rem 0.6rem',
                borderRadius: '4px'
              }}
            >
              Never leaves your device &bull; Evaluated locally
            </span>

            {/* Hidden File Inputs */}
            <input
              ref={fileInputRef}
              type="file"
              id="file-upload-input"
              aria-label="Upload chest radiograph image"
              accept="image/png,image/jpeg,image/jpg,image/webp,.heic,.heif,image/heic,image/heif"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />

            <input
              ref={cameraInputRef}
              type="file"
              id="camera-capture-input"
              aria-label="Capture radiograph using camera"
              accept="image/*"
              capture="environment"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
          </div>

          {/* Camera & Library Buttons */}
          <div style={{ marginTop: '0.85rem', display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              className="btn-secondary"
              style={{
                flex: 1,
                fontSize: '0.8rem',
                padding: '0.6rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.4rem'
              }}
              onClick={(e) => {
                e.stopPropagation();
                cameraInputRef.current?.click();
              }}
            >
              <Camera size={16} /> Capture via Camera
            </button>
            <button
              type="button"
              className="btn-secondary"
              style={{
                flex: 1,
                fontSize: '0.8rem',
                padding: '0.6rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.4rem'
              }}
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              <FileImage size={16} /> Photo Library
            </button>
          </div>

          {/* Client-Side Resize Metadata Pill */}
          {resizeMeta && (
            <div
              style={{
                marginTop: '1rem',
                padding: '0.65rem 0.85rem',
                backgroundColor: 'rgba(0, 212, 255, 0.08)',
                border: '1px solid rgba(0, 212, 255, 0.25)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '0.775rem',
                color: 'var(--text-secondary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}
            >
              <span>
                {resizeMeta.wasResized
                  ? `Optimized for inference (${resizeMeta.origDims} \u2192 ${resizeMeta.finalDims} px)`
                  : `Dimensions verified (${resizeMeta.finalDims} px)`}
              </span>
              <span style={{ color: '#00d4ff', fontWeight: 600 }}>Ready</span>
            </div>
          )}

          {/* Active Preview Thumbnail */}
          {previewUrl && (
            <div style={{ marginTop: '1.25rem' }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '0.8rem',
                  color: 'var(--text-secondary)',
                  marginBottom: '0.4rem'
                }}
              >
                <span>Loaded Input Image</span>
                <span style={{ color: '#00d4ff' }}>{selectedFile?.name || activeSampleId || 'Radiograph'}</span>
              </div>
              <div
                style={{
                  height: '220px',
                  backgroundColor: '#000000',
                  borderRadius: 'var(--radius-sm)',
                  overflow: 'hidden',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: '1px solid var(--border)'
                }}
              >
                <img
                  src={previewUrl}
                  alt="Radiograph Preview"
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              </div>
            </div>
          )}

          {/* Run Inference Action Button */}
          <div style={{ marginTop: '1.5rem' }}>
            <button
              id="run-analysis-btn"
              type="button"
              className="btn-primary"
              style={{ width: '100%', padding: '0.85rem' }}
              disabled={!selectedFile || loading || modelStatus === 'loading'}
              onClick={handleUploadPredict}
            >
              {loading ? (
                <>
                  <RefreshCw size={16} className="spin" /> Evaluating in WebAssembly...
                </>
              ) : (
                'Run In-Browser Neural Analysis'
              )}
            </button>
          </div>
        </div>

        {/* RIGHT COLUMN: Results Dashboard */}
        <div className="card" id="results-dashboard">
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '1.25rem'
            }}
          >
            <h2 style={{ fontSize: '1.15rem', fontFamily: 'var(--font-heading)' }}>
              Diagnostic Dashboard
            </h2>
            {savedStatus && (
              <span
                style={{
                  fontSize: '0.75rem',
                  color: 'var(--success)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.3rem'
                }}
              >
                <CheckCircle2 size={14} /> Audit logged
              </span>
            )}
          </div>

          {loading ? (
            <div
              style={{
                height: '380px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                textAlign: 'center',
                gap: '1rem',
                color: '#00d4ff'
              }}
            >
              <RefreshCw size={36} className="spin" />
              <div>
                <div style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', color: '#ffffff' }}>
                  Processing Radiograph Locally
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
                  Running MobileNetV2 ONNX session, Platt logits, and layer-targeted Grad-CAM heatmaps...
                </div>
              </div>
            </div>
          ) : result ? (
            <div>
              {/* Verdict Card */}
              <div
                id="verdict-card"
                style={{
                  backgroundColor: 'rgba(4, 20, 40, 0.95)',
                  border:
                    result.label === 'Normal'
                      ? '1px solid var(--success)'
                      : '1px solid var(--danger)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '1.25rem',
                  marginBottom: '1.5rem',
                  boxShadow:
                    result.label === 'Normal'
                      ? '0 0 20px rgba(16, 185, 129, 0.15)'
                      : '0 0 20px rgba(239, 68, 68, 0.15)'
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    gap: '0.75rem'
                  }}
                >
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
                      CLASSIFICATION FINDING
                    </div>
                    <div
                      style={{
                        fontSize: '2rem',
                        fontFamily: 'var(--font-heading)',
                        fontWeight: 900,
                        color: result.label === 'Normal' ? 'var(--success)' : 'var(--danger)',
                        lineHeight: 1.15
                      }}
                    >
                      {result.label}
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      CALIBRATED PROBABILITY
                    </div>
                    <div
                      style={{
                        fontSize: '1.8rem',
                        fontFamily: 'var(--font-heading)',
                        fontWeight: 900,
                        color: '#00d4ff'
                      }}
                    >
                      {(result.calibrated_probability * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Privacy Badge */}
                <div
                  id="browser-privacy-badge"
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    padding: '0.25rem 0.65rem',
                    backgroundColor: 'rgba(0, 212, 255, 0.08)',
                    border: '1px solid rgba(0, 212, 255, 0.3)',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    color: '#00d4ff',
                    marginTop: '0.75rem'
                  }}
                >
                  <ShieldCheck size={14} />
                  <span>Ran in your browser. Your image never left your device.</span>
                </div>

                <div
                  style={{
                    marginTop: '0.75rem',
                    paddingTop: '0.75rem',
                    borderTop: '1px solid rgba(13, 45, 74, 0.6)',
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-between',
                    fontSize: '0.8rem',
                    color: 'var(--text-secondary)'
                  }}
                >
                  <span>
                    Finding:{' '}
                    <strong style={{ color: '#ffffff' }}>
                      {result.finding || (result.label === 'Normal' ? 'No abnormal finding flagged' : 'Possible abnormal finding')}
                    </strong>
                  </span>
                  <span>
                    Raw Model Sigmoid: <strong>{(result.uncalibrated_probability * 100).toFixed(1)}%</strong>
                  </span>
                </div>
              </div>

              {/* Guest / Unverified Practitioner Save Prompt */}
              {!currentUser && (
                <div
                  id="guest-save-prompt"
                  style={{
                    marginBottom: '1.5rem',
                    padding: '0.85rem 1.15rem',
                    backgroundColor: 'rgba(10, 31, 53, 0.85)',
                    border: '1px solid rgba(0, 212, 255, 0.35)',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexWrap: 'wrap',
                    gap: '0.75rem'
                  }}
                >
                  <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                    <strong style={{ color: '#ffffff' }}>Guest Evaluation:</strong> Sign in to save this result to your permanent clinical audit log.
                  </div>
                  <Link to="/login" className="btn-primary btn-sm" style={{ padding: '0.35rem 0.85rem', fontSize: '0.8rem' }}>
                    Sign In to Save
                  </Link>
                </div>
              )}

              {currentUser && !currentUser.emailVerified && (
                <div
                  id="unverified-save-prompt"
                  style={{
                    marginBottom: '1.5rem',
                    padding: '0.85rem 1.15rem',
                    backgroundColor: 'rgba(245, 158, 11, 0.12)',
                    border: '1px solid var(--warning)',
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexWrap: 'wrap',
                    gap: '0.75rem'
                  }}
                >
                  <div style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                    <strong style={{ color: '#f59e0b' }}>Email Verification Required:</strong> Verify your practitioner email to log analyses to Firestore.
                  </div>
                  <Link to="/verify-email" className="btn-secondary btn-sm" style={{ padding: '0.35rem 0.85rem', fontSize: '0.8rem' }}>
                    Verify Email
                  </Link>
                </div>
              )}

              {/* Recharts Probability Bar */}
              <div
                style={{
                  backgroundColor: 'var(--bg-input)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '1rem',
                  border: '1px solid var(--border)',
                  marginBottom: '1.5rem'
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '0.75rem',
                    fontSize: '0.85rem'
                  }}
                >
                  <span style={{ fontWeight: 600, color: '#ffffff' }}>
                    Platt Scaling Probability Gauge
                  </span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    Threshold: {(result.threshold ? result.threshold * 100 : 65.2).toFixed(1)}%
                  </span>
                </div>

                <div style={{ width: '100%', height: 110 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={chartData}
                      margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
                    >
                      <XAxis
                        type="number"
                        domain={[0, 100]}
                        tick={{ fill: '#8ab4c9', fontSize: 11 }}
                        unit="%"
                      />
                      <YAxis type="category" dataKey="name" hide />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#041428',
                          borderColor: '#00d4ff',
                          borderRadius: '6px',
                          color: '#ffffff',
                          fontSize: '12px'
                        }}
                        formatter={(value) => [`${value}%`, 'Abnormality Score']}
                      />
                      <ReferenceLine
                        x={Number(((result.threshold || 0.6522) * 100).toFixed(1))}
                        stroke="#ffa502"
                        strokeDasharray="3 3"
                        label={{
                          value: `Threshold (${((result.threshold || 0.6522) * 100).toFixed(1)}%)`,
                          fill: '#ffa502',
                          fontSize: 10,
                          position: 'top'
                        }}
                      />
                      <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
                        <Cell
                          fill={
                            result.calibrated_probability >= (result.threshold || 0.6522)
                              ? '#ef4444'
                              : '#10b981'
                          }
                        />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Visualizations: Side-by-Side & Opacity Slider */}
              <div style={{ marginBottom: '1.5rem' }}>
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '0.75rem'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <Eye size={16} color="#00d4ff" />
                    <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#ffffff' }}>
                      Spatial Grad-CAM Focus Heatmap
                    </span>
                  </div>

                  {/* Mode Selector */}
                  <div style={{ display: 'flex', gap: '0.35rem' }}>
                    <button
                      type="button"
                      onClick={() => setViewMode('sideBySide')}
                      className={viewMode === 'sideBySide' ? 'btn-primary btn-sm' : 'btn-secondary btn-sm'}
                      style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
                    >
                      Side by Side
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewMode('blended')}
                      className={viewMode === 'blended' ? 'btn-primary btn-sm' : 'btn-secondary btn-sm'}
                      style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
                    >
                      Interactive Overlay
                    </button>
                  </div>
                </div>

                {/* Opacity Slider Control */}
                <div
                  style={{
                    backgroundColor: 'rgba(4, 20, 40, 0.7)',
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    padding: '0.75rem 1rem',
                    marginBottom: '1rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '1rem'
                  }}
                >
                  <Sliders size={18} color="#00d4ff" style={{ flexShrink: 0 }} />
                  <div style={{ flex: 1 }}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.75rem',
                        color: 'var(--text-secondary)',
                        marginBottom: '0.25rem'
                      }}
                    >
                      <span>Heatmap Intensity Opacity</span>
                      <span style={{ color: '#00d4ff', fontWeight: 600 }}>{overlayOpacity}%</span>
                    </div>
                    <input
                      id="heatmap-opacity-slider"
                      aria-label="Adjust Grad-CAM heatmap overlay opacity percentage"
                      type="range"
                      min="0"
                      max="100"
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                      style={{
                        width: '100%',
                        cursor: 'pointer',
                        accentColor: '#00d4ff'
                      }}
                    />
                  </div>
                </div>

                {/* View: Side by Side */}
                {viewMode === 'sideBySide' ? (
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                      gap: '0.75rem'
                    }}
                  >
                    <div>
                      <div
                        style={{
                          fontSize: '0.75rem',
                          color: 'var(--text-secondary)',
                          marginBottom: '0.35rem',
                          textAlign: 'center'
                        }}
                      >
                        Original Radiograph
                      </div>
                      <div
                        style={{
                          height: '200px',
                          backgroundColor: '#000000',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--border)',
                          overflow: 'hidden',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        {previewUrl ? (
                          <img
                            src={previewUrl}
                            alt="Original Radiograph"
                            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                          />
                        ) : (
                          <FileImage size={24} color="#0d2d4a" />
                        )}
                      </div>
                    </div>

                    <div>
                      <div
                        style={{
                          fontSize: '0.75rem',
                          color: 'var(--text-secondary)',
                          marginBottom: '0.35rem',
                          textAlign: 'center'
                        }}
                      >
                        Grad-CAM Attention Map
                      </div>
                      <div
                        style={{
                          height: '200px',
                          backgroundColor: '#000000',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--border)',
                          overflow: 'hidden',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}
                      >
                        {result.gradcam_overlay ? (
                          <img
                            src={result.gradcam_overlay}
                            alt="Grad-CAM Heatmap"
                            style={{
                              width: '100%',
                              height: '100%',
                              objectFit: 'contain',
                              opacity: overlayOpacity / 100
                            }}
                          />
                        ) : (
                          <div style={{ textAlign: 'center', padding: '1rem', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                            Grad-CAM generated in real-time
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  /* View: Interactive Layered Overlay */
                  <div
                    style={{
                      position: 'relative',
                      height: '280px',
                      backgroundColor: '#000000',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border)',
                      overflow: 'hidden',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                  >
                    {previewUrl && (
                      <img
                        src={previewUrl}
                        alt="Original Base"
                        style={{
                          position: 'absolute',
                          width: '100%',
                          height: '100%',
                          objectFit: 'contain',
                          zIndex: 1
                        }}
                      />
                    )}

                    {result.gradcam_overlay && (
                      <img
                        src={result.gradcam_overlay}
                        alt="Grad-CAM Layer"
                        style={{
                          position: 'absolute',
                          width: '100%',
                          height: '100%',
                          objectFit: 'contain',
                          opacity: overlayOpacity / 100,
                          mixBlendMode: 'screen',
                          zIndex: 2,
                          transition: 'opacity 0.1s ease'
                        }}
                      />
                    )}
                  </div>
                )}
              </div>

              {/* Research Demo Disclaimer Banner */}
              <div
                id="results-disclaimer-box"
                style={{
                  backgroundColor: 'rgba(10, 31, 53, 0.8)',
                  border: '1px solid rgba(0, 212, 255, 0.3)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '1rem',
                  display: 'flex',
                  gap: '0.75rem',
                  alignItems: 'flex-start'
                }}
              >
                <Info size={18} color="#00d4ff" style={{ flexShrink: 0, marginTop: '2px' }} />
                <div style={{ fontSize: '0.775rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                  <strong style={{ color: '#00d4ff', display: 'block', marginBottom: '0.2rem' }}>
                    RESEARCH DEMONSTRATION NOTICE
                  </strong>
                  {result.disclaimer ||
                    'RESEARCH DEMO ONLY: This model output is strictly for experimental research and investigation. It is NOT a certified medical diagnosis and should never replace licensed clinical judgment.'}
                </div>
              </div>
            </div>
          ) : (
            /* Empty State */
            <div
              id="results-empty-state"
              style={{
                height: '380px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                textAlign: 'center',
                padding: '2rem 1rem',
                color: 'var(--text-muted)'
              }}
            >
              <div
                style={{
                  width: '64px',
                  height: '64px',
                  borderRadius: '50%',
                  backgroundColor: 'rgba(4, 20, 40, 0.7)',
                  border: '1px solid var(--border)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1rem',
                  color: '#4e738a'
                }}
              >
                <FileImage size={32} />
              </div>
              <h3
                style={{
                  fontSize: '1.1rem',
                  color: '#ffffff',
                  fontFamily: 'var(--font-heading)',
                  marginBottom: '0.35rem'
                }}
              >
                Awaiting Radiograph Input
              </h3>
              <p style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', maxWidth: '340px', lineHeight: '1.5' }}>
                Select a sample above to test immediately, or tap the upload box on the left to submit a custom chest radiograph. Everything runs on your device.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AnalyzePage;
