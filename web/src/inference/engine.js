/**
 * Client-Side Inference Engine for InsightX.
 * Executes insightx_gradcam.onnx via onnxruntime-web (browser) or onnxruntime-node (testing).
 *
 * Architecture:
 * - Single-threaded WASM SIMD (zero COOP/COEP header dependencies).
 * - CacheStorage caching for instant subsequent loads.
 * - Progress callback for download progress.
 * - Dynamic asset base URL via VITE_ASSET_BASE_URL.
 */

import { preprocessImage } from './preprocess.js';
import { computeGradCam } from './gradcam.js';
import { calibratePrediction } from './calibrate.js';

const CACHE_NAME = 'insightx-model-v1';
const MODEL_FILENAME = 'insightx_gradcam.onnx';
const CALIB_FILENAME = 'calibration.json';
const WEIGHTS_FILENAME = 'gradcam_head_weights.json';

export class ClientSideInferenceEngine {
  constructor(options = {}) {
    this.assetBaseUrl =
      options.assetBaseUrl !== undefined
        ? options.assetBaseUrl
        : (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_ASSET_BASE_URL) || '';

    this.session = null;
    this.calibConfig = null;
    this.camWeights = null;
    this.inputName = 'image_input';
    this.isDualOutput = true;
    this._loadPromise = null;
  }

  isReady() {
    return this.session !== null && this.calibConfig !== null && this.camWeights !== null;
  }

  /**
   * Resolve an asset URL given the configured assetBaseUrl.
   */
  resolveAssetUrl(path) {
    const cleanBase = this.assetBaseUrl.replace(/\/+$/, '');
    const cleanPath = path.replace(/^\/+/, '');
    return cleanBase ? `${cleanBase}/${cleanPath}` : `/${cleanPath}`;
  }

  /**
   * Fetch a binary array buffer with progress reporting, backed by CacheStorage when available.
   * Resilient to CacheStorage failures, HTML SPA fallbacks, and missing Content-Length.
   */
  async _fetchWithCache(url, onProgress = null) {
    // 1. Try CacheStorage in browser (fail-soft: any cache error falls back to network)
    if (typeof caches !== 'undefined') {
      try {
        const cache = await caches.open(CACHE_NAME);
        const cachedRes = await cache.match(url);
        if (cachedRes) {
          const contentType = (cachedRes.headers.get('content-type') || '').toLowerCase();
          // If cached response was inadvertently an SPA fallback HTML or empty, purge it
          if (!contentType.includes('text/html')) {
            const cachedBuf = await cachedRes.arrayBuffer();
            if (cachedBuf && cachedBuf.byteLength > 1024) {
              if (onProgress) onProgress(100, 'Loaded from cache');
              return cachedBuf;
            }
          }
          console.warn('[InsightX Cache Warning] Purging corrupt or HTML cache entry for:', url);
          await cache.delete(url);
        }
      } catch (cacheErr) {
        console.warn('[InsightX Cache Warning] Cache read failed; falling back to direct network fetch:', cacheErr);
      }
    }

    // 2. Fetch from network with stream progress reporting
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to download model asset from ${url} (HTTP ${response.status})`);
    }

    const contentType = (response.headers.get('content-type') || '').toLowerCase();
    if (contentType.includes('text/html')) {
      throw new Error(
        `Server returned HTML instead of model weights for ${url} (SPA fallback). Ensure model assets are deployed in /models/.`
      );
    }

    const rawContentLength = response.headers.get('content-length');
    const contentLength = rawContentLength ? Math.max(0, Number(rawContentLength) || 0) : 0;
    let receivedBytes = 0;
    const reader = response.body ? response.body.getReader() : null;

    let buffer;
    if (reader) {
      const chunks = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        receivedBytes += value.length;
        if (onProgress) {
          if (contentLength > 0) {
            const percent = Math.min(100, Math.round((receivedBytes / contentLength) * 100));
            const mb = (receivedBytes / (1024 * 1024)).toFixed(1);
            onProgress(percent, `Downloading weights: ${percent}% (${mb} MB)`);
          } else {
            const mb = (receivedBytes / (1024 * 1024)).toFixed(1);
            onProgress(50, `Downloading weights: ${mb} MB`);
          }
        }
      }
      const allChunks = new Uint8Array(receivedBytes);
      let pos = 0;
      for (const chunk of chunks) {
        allChunks.set(chunk, pos);
        pos += chunk.length;
      }
      buffer = allChunks.buffer;
    } else {
      buffer = await response.arrayBuffer();
      if (onProgress) onProgress(100, 'Downloaded');
    }

    if (!buffer || buffer.byteLength <= 1024) {
      throw new Error(`Downloaded model buffer from ${url} is empty or invalid (${buffer?.byteLength || 0} bytes).`);
    }

    // 3. Store into CacheStorage (fail-soft: any put error is non-fatal)
    if (typeof caches !== 'undefined') {
      try {
        const cache = await caches.open(CACHE_NAME);
        await cache.put(
          url,
          new Response(buffer.slice(0), {
            headers: {
              'Content-Type': 'application/octet-stream',
              'Content-Length': String(buffer.byteLength)
            }
          })
        );
      } catch (cacheErr) {
        console.warn('[InsightX Cache Put Warning] Cache write failed; proceeding un-cached:', cacheErr);
      }
    }

    return buffer;
  }

  /**
   * Initialize and load model artifacts once.
   *
   * @param {function(number, string): void} [onProgress] Progress callback (0..100, status string)
   */
  async initialize(onProgress = null) {
    if (this.isReady()) return;
    if (this._loadPromise) return this._loadPromise;

    this._loadPromise = (async () => {
      if (onProgress) onProgress(5, 'Loading metadata...');

      // 1. Load Calibration Config
      const calibUrl = this.resolveAssetUrl(`models/${CALIB_FILENAME}`);
      const calibRes = await fetch(calibUrl);
      if (!calibRes.ok) throw new Error(`Failed to load ${calibUrl}`);
      this.calibConfig = await calibRes.json();

      // 2. Load CAM Head Weights
      const weightsUrl = this.resolveAssetUrl(`models/${WEIGHTS_FILENAME}`);
      const weightsRes = await fetch(weightsUrl);
      if (!weightsRes.ok) throw new Error(`Failed to load ${weightsUrl}`);
      const weightsData = await weightsRes.json();
      this.camWeights = new Float32Array(weightsData.weights);

      if (onProgress) onProgress(15, 'Preparing ONNX runtime...');

      // 3. Select Runtime: onnxruntime-web in browser, onnxruntime-node in Node
      let ort;
      if (typeof window !== 'undefined' || typeof self !== 'undefined') {
        // Browser environment: import wasm-only bundle
        const ortWeb = await import('onnxruntime-web/wasm');
        ort = ortWeb.default || ortWeb;
        ort.env.wasm.numThreads = 1; // single-thread SIMD: no cross-origin isolation needed
        ort.env.wasm.simd = true;
        const wasmBase = this.resolveAssetUrl('wasm/');
        const wasmClean = wasmBase.endsWith('/') ? wasmBase : `${wasmBase}/`;
        ort.env.wasm.wasmPaths = {
          wasm: `${wasmClean}ort-wasm-simd-threaded.wasm`,
          mjs: `${wasmClean}ort-wasm-simd-threaded.mjs`
        };
      } else {
        // Node environment for parity tests
        const pkg = 'onnxruntime-' + 'node';
        const ortNode = await import(/* @vite-ignore */ pkg);
        ort = ortNode.default || ortNode;
      }

      // 4. Download / load ONNX model buffer
      const modelUrl = this.resolveAssetUrl(`models/${MODEL_FILENAME}`);
      const modelBuffer = await this._fetchWithCache(modelUrl, (pct, status) => {
        if (onProgress) onProgress(15 + Math.round(pct * 0.7), status);
      });

      if (onProgress) onProgress(90, 'Compiling WebAssembly execution graph...');

      // 5. Create InferenceSession (with fail-soft cache cleanup on session failure)
      try {
        this.session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ['wasm']
        });
      } catch (sessionErr) {
        if (typeof caches !== 'undefined') {
          try {
            const cache = await caches.open(CACHE_NAME);
            await cache.delete(modelUrl);
          } catch (e) {
            console.debug('Failed to purge stale model cache:', e);
          }
        }
        throw sessionErr;
      }

      this.inputName = this.session.inputNames[0] || 'image_input';

      if (onProgress) onProgress(100, 'Model ready');
    })();

    return this._loadPromise;
  }

  /**
   * Run inference on an image object { data: Uint8Array RGBA, width, height }.
   *
   * @param {{ data: Uint8Array | Uint8ClampedArray, width: number, height: number }} imageObj
   * @param {object} [options={}]
   * @returns {Promise<object>} PredictResponse matching the FastAPI reference contract
   */
  async predict(imageObj, _options = {}) {
    if (!this.isReady()) {
      await this.initialize();
    }

    // Step 1: Preprocess to 224x224 NHWC Float32Array
    const { tensor: imgTensor, cropInfo } = preprocessImage(imageObj);

    // Step 2: Create ONNX Tensor
    let ort;
    if (typeof window !== 'undefined' || typeof self !== 'undefined') {
      const ortWeb = await import('onnxruntime-web/wasm');
      ort = ortWeb.default || ortWeb;
    } else {
      const pkg = 'onnxruntime-' + 'node';
      const ortNode = await import(/* @vite-ignore */ pkg);
      ort = ortNode.default || ortNode;
    }

    const inputTensor = new ort.Tensor('float32', imgTensor, [1, 224, 224, 3]);

    // Step 3: Run ONNX Session
    const outputs = await this.session.run({ [this.inputName]: inputTensor });

    // Step 4: Parse outputs (dual output: mobilenetv2_1.00_224 and logit_output)
    let convOut = null;
    let logitOut = null;

    for (const key of Object.keys(outputs)) {
      const out = outputs[key];
      if (out.dims && out.dims[out.dims.length - 1] === 1280) {
        convOut = out.data;
      } else if (out.dims && out.dims[out.dims.length - 1] === 1) {
        logitOut = out.data;
      }
    }

    if (!convOut || !logitOut) {
      // Fallback by key order
      const keys = Object.keys(outputs);
      convOut = outputs[keys[0]].data;
      logitOut = outputs[keys[1]].data;
    }

    const rawLogit = Number(logitOut[0]);

    // Step 5: Postprocessing & Calibration
    const calibratedResult = calibratePrediction(rawLogit, this.calibConfig);

    // Step 6: Grad-CAM Overlay
    const { overlayDataUri } = await computeGradCam(
      convOut,
      this.camWeights,
      cropInfo,
      imageObj,
      0.45
    );

    return {
      ...calibratedResult,
      gradcam_overlay: overlayDataUri,
      model_version: '1.0.0'
    };
  }
}
