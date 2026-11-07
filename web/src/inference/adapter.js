/**
 * Inference Adapter
 *
 * Provides a unified inference interface for InsightX:
 * - isReady()
 * - initialize(onProgress)
 * - predict(file | blob | url | imageObj)
 * - listSamples()
 *
 * Backed by the "browser" backend (ClientSideInferenceEngine).
 * Structured so a "server" backend (FastAPI /predict) can be plugged in later
 * without modifying AnalyzePage.jsx or other UI components.
 */

import { ClientSideInferenceEngine } from './engine.js';

const RESEARCH_DISCLAIMER =
  'RESEARCH DEMO ONLY: This model output is strictly for experimental research and investigation. ' +
  'It is NOT a certified medical diagnosis and should never replace licensed clinical judgment.';

const FALLBACK_SAMPLES = [
  {
    id: 'sample_01_normal.png',
    sample_id: 'sample_01_normal',
    title: 'Sample 1: Normal Baseline',
    desc: 'Indiana University radiograph demonstrating clear cardiothoracic fields.',
    expectedLabel: 'Normal',
    category: 'True Normal (TN)',
    url: '/samples/sample_01_normal.png'
  },
  {
    id: 'sample_02_normal.png',
    sample_id: 'sample_02_normal',
    title: 'Sample 2: Normal Projection',
    desc: 'Indiana University radiograph with unremarkable pulmonary parenchyma.',
    expectedLabel: 'Normal',
    category: 'True Normal (TN)',
    url: '/samples/sample_02_normal.png'
  },
  {
    id: 'sample_03_normal.png',
    sample_id: 'sample_03_normal',
    title: 'Sample 3: Normal Silhouette',
    desc: 'Indiana University radiograph without focal consolidation or effusion.',
    expectedLabel: 'Normal',
    category: 'True Normal (TN)',
    url: '/samples/sample_03_normal.png'
  },
  {
    id: 'sample_04_abnormal.png',
    sample_id: 'sample_04_abnormal',
    title: 'Sample 4: Focal Granuloma Case',
    desc: 'Findings: Calcinosis, Calcified Granuloma in lung field.',
    expectedLabel: 'Abnormal',
    category: 'True Abnormal (TP)',
    url: '/samples/sample_04_abnormal.png'
  },
  {
    id: 'sample_05_abnormal.png',
    sample_id: 'sample_05_abnormal',
    title: 'Sample 5: Multi-Focal Abnormal Case',
    desc: 'Findings: Pulmonary Edema, Pleural Effusion, Calcinosis.',
    expectedLabel: 'Abnormal',
    category: 'True Abnormal (TP)',
    url: '/samples/sample_05_abnormal.png'
  },
  {
    id: 'sample_06_abnormal.png',
    sample_id: 'sample_06_abnormal',
    title: 'Sample 6: Cardiomegaly / Fibrosis',
    desc: 'Findings: Cardiomegaly, Pulmonary Fibrosis, Granulomatous Disease.',
    expectedLabel: 'Abnormal',
    category: 'True Abnormal (TP)',
    url: '/samples/sample_06_abnormal.png'
  },
  {
    id: 'sample_07_false_positive.png',
    sample_id: 'sample_07_false_positive',
    title: 'Sample 7: Diagnostic Boundary Case',
    desc: 'Prominent thoracic markings classified near operating threshold.',
    expectedLabel: 'Abnormal',
    category: 'Operating Boundary',
    url: '/samples/sample_07_false_positive.png'
  },
  {
    id: 'sample_08_false_negative.png',
    sample_id: 'sample_08_false_negative',
    title: 'Sample 8: Subtle Pathology Case',
    desc: 'Subtle parenchymal markings evaluated at validation threshold.',
    expectedLabel: 'Normal',
    category: 'Operating Boundary',
    url: '/samples/sample_08_false_negative.png'
  }
];

/**
 * Load an image from File, Blob, URL string, or raw pixel object into
 * { data: Uint8ClampedArray RGBA, width: number, height: number }.
 */
async function loadImageData(input) {
  if (input && input.data && typeof input.width === 'number' && typeof input.height === 'number') {
    return input;
  }

  let url = '';
  let needRevoke = false;

  if (typeof input === 'string') {
    url = input;
  } else if (input instanceof Blob || (typeof File !== 'undefined' && input instanceof File)) {
    url = URL.createObjectURL(input);
    needRevoke = true;
  } else {
    throw new Error('Unsupported image input format. Expected File, Blob, or URL string.');
  }

  try {
    if (typeof document !== 'undefined' && document.createElement) {
      const img = await new Promise((resolve, reject) => {
        const el = new window.Image();
        el.crossOrigin = 'anonymous';
        el.onload = () => resolve(el);
        el.onerror = () =>
          reject(new Error('Failed to decode image. The file may be corrupt or an unsupported format.'));
        el.src = url;
      });

      const width = img.naturalWidth || img.width;
      const height = img.naturalHeight || img.height;

      if (!width || !height || width <= 0 || height <= 0) {
        throw new Error('Decoded image has invalid dimensions.');
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const imgData = ctx.getImageData(0, 0, width, height);

      return {
        data: imgData.data,
        width,
        height
      };
    }
    throw new Error('Image decoding requires a browser DOM environment.');
  } finally {
    if (needRevoke && typeof URL !== 'undefined' && URL.revokeObjectURL) {
      URL.revokeObjectURL(url);
    }
  }
}

class BrowserBackend {
  constructor() {
    this.engine = new ClientSideInferenceEngine();
  }

  isReady() {
    return this.engine.isReady();
  }

  async initialize(onProgress = null) {
    return this.engine.initialize(onProgress);
  }

  async predict(input) {
    const imageObj = await loadImageData(input);
    const result = await this.engine.predict(imageObj);
    return {
      ...result,
      backend: 'browser',
      disclaimer: RESEARCH_DISCLAIMER
    };
  }

  async listSamples() {
    try {
      const manifestUrl = this.engine.resolveAssetUrl('samples/manifest.json');
      const res = await fetch(manifestUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        return data.map((s, i) => {
          const isAbnormal = s.ground_truth_label === 'Abnormal';
          return {
            id: s.id,
            sample_id: s.sample_id || `sample_${i + 1}`,
            title: `Sample ${i + 1}: ${isAbnormal ? 'Abnormal Case' : 'Normal Baseline'}`,
            desc:
              s.Problems && s.Problems !== 'normal'
                ? `Findings: ${s.Problems.replace(/;/g, ', ')}`
                : 'Clear lung fields without focal consolidation or acute cardiopulmonary abnormality.',
            expectedLabel: s.ground_truth_label || (isAbnormal ? 'Abnormal' : 'Normal'),
            category: s.category || (isAbnormal ? 'True Abnormal (TP)' : 'True Normal (TN)'),
            url: this.engine.resolveAssetUrl(`samples/${s.id}`)
          };
        });
      }
    } catch {
      // Fallback to static sample metadata if network fetch fails
    }
    return FALLBACK_SAMPLES.map((s) => ({
      ...s,
      url: this.engine.resolveAssetUrl(s.url)
    }));
  }
}

/**
 * InferenceAdapter Singleton
 *
 * Dispatches inference calls to the active backend ('browser' by default).
 */
export class InferenceAdapter {
  constructor(options = {}) {
    this.backendType = options.backend || 'browser';
    this.browserBackend = new BrowserBackend();
    // Future server backend can be instantiated here if enabled
    this.serverBackend = null;
  }

  getActiveBackend() {
    return this.backendType === 'server' && this.serverBackend
      ? this.serverBackend
      : this.browserBackend;
  }

  getBackendType() {
    return this.backendType;
  }

  isReady() {
    return this.getActiveBackend().isReady();
  }

  async initialize(onProgress = null) {
    return this.getActiveBackend().initialize(onProgress);
  }

  async predict(input) {
    return this.getActiveBackend().predict(input);
  }

  async listSamples() {
    return this.getActiveBackend().listSamples();
  }
}

// Global default instance for application-wide use
export const inferenceAdapter = new InferenceAdapter();
export default inferenceAdapter;
