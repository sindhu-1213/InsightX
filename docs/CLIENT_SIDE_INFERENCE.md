# Client-Side Inference Architecture & Implementation Plan

> **InsightX - In-Browser Chest Radiograph Abnormality Detection & Grad-CAM**  
> *Target Architecture*: Zero-backend static Progressive Web Application (PWA) hosted on Firebase Hosting, executing `insightx_gradcam.onnx` via `onnxruntime-web` (WebAssembly SIMD).  
> *Reference Implementation*: The existing FastAPI service (`api/main.py`) remains fully operational in the repository for benchmarking, API clients, and Python test suites.

---

## 1. Exact Inference Contract

This section defines the mathematical and programmatic contract extracted directly from the reference implementation in `api/main.py` and `insightx/preprocess.py`.

### 1.1 Image Preprocessing Contract
* **Source Locations**:
  * `insightx/preprocess.py`: Lines 32–84 (`resize_with_padding` and `preprocess_image_for_onnx`).
  * `api/main.py`: Line 38 (import), Line 446 (`preprocess_image_for_onnx(image)` call in `_run_prediction_pipeline`).
* **Color Mode**:
  * Input is converted to grayscale: `image.convert("L")` (`insightx/preprocess.py:47`).
  * Output values are 8-bit integers in $[0, 255]$.
* **Resize & Padding**:
  * Target canvas size: `(224, 224)` (`insightx/preprocess.py:34, 49`).
  * Aspect ratio preservation:
    $$\text{scale} = \min\left(\frac{224}{w}, \frac{224}{h}\right)$$
    $$\text{new\_w} = \max(1, \text{round}(w \cdot \text{scale})), \quad \text{new\_h} = \max(1, \text{round}(h \cdot \text{scale}))$$
    (`insightx/preprocess.py:51-53`).
  * Interpolation algorithm: `Image.Resampling.LANCZOS` (`insightx/preprocess.py:55`).
  * Canvas padding: Black canvas (`pad_color = 0`), image centered with integer floor division:
    $$\text{paste\_x} = \lfloor(224 - \text{new\_w}) / 2\rfloor, \quad \text{paste\_y} = \lfloor(224 - \text{new\_h}) / 2\rfloor$$
    (`insightx/preprocess.py:56-59`).
* **Channel Replication & Normalization**:
  * Grayscale float conversion: `arr_gray = np.asarray(canvas, dtype=np.float32)` (`insightx/preprocess.py:80`).
  * Channel replication: Replicated 3 times along channel axis:
    `arr_3ch = np.repeat(arr_gray[:, :, np.newaxis], 3, axis=-1)` (`insightx/preprocess.py:81`).
  * MobileNetV2 standard normalization:
    $$\text{tensor} = \left(\frac{\text{arr\_3ch}}{127.5}\right) - 1.0$$
    Pixel range is strictly mapped to $[-1.0, 1.0]$ (`insightx/preprocess.py:82`).
* **Tensor Layout & Dtype**:
  * Channel Order: **NHWC** (Batch, Height, Width, Channels) — **channels-last**.
  * Shape: `(1, 224, 224, 3)`.
  * Dtype: `Float32Array` (`float32`).
* **Sample Images Loading**:
  * Catalog manifest: `samples/manifest.json` (8 samples).
  * API endpoints: `api/main.py:340-401` (`/samples`, `/samples/{sample_id}/image`), `api/main.py:505-550` (`/samples/{sample_id}/predict`).
  * Images read from disk as RGB: `Image.open(target_file).convert("RGB")` (`api/main.py:542`).

---

### 1.2 Model I/O Specification
Inspected directly using `onnx` and `onnxruntime` on the active repository model files:

| Attribute | `models/insightx.onnx` | `models/insightx_gradcam.onnx` (Recommended) |
| :--- | :--- | :--- |
| **File Size** | 8,919,089 bytes (~8.51 MB) | 8,913,759 bytes (~8.50 MB) |
| **ONNX IR Version** | IR v7, Opset 13 (`('', 13)`, `('ai.onnx.ml', 2)`) | IR v7, Opset 13 (`('', 13)`, `('ai.onnx.ml', 2)`) |
| **Input Name** | `image_input` | `image_input` |
| **Input Shape & Type** | `[unk__654, 224, 224, 3]`, `tensor(float)` (float32) | `[unk__1314, 224, 224, 3]`, `tensor(float)` (float32) |
| **Output 0** | `prediction`: `[unk__655, 1]`, `tensor(float)` | `mobilenetv2_1.00_224`: `[unk__1315, 7, 7, 1280]`, `tensor(float)` |
| **Output 1** | *(None)* | `logit_output`: `[unk__1316, 1]`, `tensor(float)` |
| **Operators Used** | `Add`, `Clip`, `Conv`, `GlobalAveragePool`, `MatMul`, `Mul`, `Pad`, `Sigmoid`, `Squeeze`, `Transpose` | `Add`, `Clip`, `Conv`, `GlobalAveragePool`, `MatMul`, `Mul`, `Pad`, `Squeeze`, `Transpose` |

> **Key Architectural Takeaway**: `models/insightx_gradcam.onnx` outputs **both** the feature activations (`7x7x1280`) and the raw classification logit (`1x1`). It is a strict superset of `insightx.onnx`. Only **one** ONNX file needs to be loaded by the browser.

---

### 1.3 Postprocessing & Calibration Contract
* **Platt Scaling Source**:
  * File: `models/calibration.json` (`api/main.py:275-285`).
  * Fitted on validation split:
    * `coef` ($A$): `0.8150681805454365` (approx `0.8151`)
    * `intercept` ($B$): `0.6795524680848608` (approx `0.6796`)
    * `threshold_youden` ($T$): `0.6521837516887142` (approx `0.6522`)
* **Mathematical Calibration Steps** (`api/main.py:451-463`):
  1. Extract raw linear logit $z$ from model output:
     $$z = \text{logit\_output}[0, 0]$$
  2. Compute uncalibrated probability:
     $$p_{\text{uncal}} = \frac{1}{1 + e^{-z}}$$
  3. Apply affine Platt transformation:
     $$z_{\text{cal}} = A \cdot z + B$$
  4. Compute calibrated probability:
     $$p_{\text{cal}} = \text{clip}\left(\frac{1}{1 + e^{-z_{\text{cal}}}}, 0.0, 1.0\right)$$
* **Classification Decision & Finding Strings** (`api/main.py:464-468`):
  * Primary operating threshold: $T = 0.6522$ (Youden index optimal cutoff).
  * Prediction code:
    $$\text{prediction\_code} = \begin{cases} 1 & \text{if } p_{\text{cal}} \ge 0.6522 \\ 0 & \text{if } p_{\text{cal}} < 0.6522 \end{cases}$$
  * Labels:
    * Label: `"Abnormal"` if code is 1, else `"Normal"`
    * Finding string (strictly enforced by Firestore Security Rules):
      * If $p_{\text{cal}} \ge 0.6522$: `"Possible abnormal finding"`
      * If $p_{\text{cal}} < 0.6522$: `"No abnormal finding flagged"`
* **Uncertain Band Logic**:
  * In the core prediction engine (`api/main.py:464-468`), there is **no** ternary uncertain finding string; the API verdict is strictly binary.
  * In `models/calibration.json`, a secondary threshold is stored: `threshold_90_specificity = 0.7291`. The UI can optionally present an informational banner for probabilities between $0.50$ and $0.6522$ as "Borderline Negative (Under Research Cutoff)", but the formal `finding` property must remain `"No abnormal finding flagged"` to satisfy Firestore schema rules.

---

### 1.4 Grad-CAM Overlay Contract
* **Inputs**:
  * Convolutional features tensor: `conv_features` of shape `(7, 7, 1280)` float32 from `mobilenetv2_1.00_224`.
  * Linear head weights: 1D array of 1280 float32 weights from `models/gradcam_head_weights.json` (`api/main.py:287-293`).
* **Exact NumPy Step-by-Step Porting Guide** (`api/main.py:403-433`):
  1. **Linear Combination across Channels**:
     $$\text{cam}[i, j] = \sum_{k=0}^{1279} \text{conv\_features}[i, j, k] \cdot \text{weights}[k]$$
     *NumPy*: `cam = np.sum(conv_features * weights.reshape(1, 1, -1), axis=-1)` $\to$ Shape `(7, 7)`.
  2. **ReLU Rectification**:
     $$\text{cam}[i, j] = \max(\text{cam}[i, j], 0.0)$$
     *NumPy*: `cam = np.maximum(cam, 0.0)`.
  3. **Max Normalization**:
     $$\text{cam}_{\text{norm}}[i, j] = \frac{\text{cam}[i, j]}{\max(\text{cam}) + 10^{-10}} \quad (\text{if } \max(\text{cam}) > 0)$$
     *NumPy*: `cam = cam / (np.max(cam) + 1e-10)`. Values in $[0.0, 1.0]$.
  4. **Quantization & Aspect-Preserving Content Crop**:
     * Scaled to uint8 $[0, 255]$: `cam_uint8 = np.uint8(np.clip(cam * 255.0, 0, 255))`.
     * Upsample 7x7 to 224x224 (the full padded canvas):
       `cam_canvas = Image.fromarray(cam_uint8).resize((224, 224), Image.Resampling.BILINEAR)`.
     * **Padding Crop Fix**: The input image was centered with black padding to 224x224. Crop out the black padding region so only the unpadded content rectangle `(paste_x, paste_y, paste_x + new_w, paste_y + new_h)` is retained:
       `cam_cropped = cam_canvas.crop((paste_x, paste_y, paste_x + new_w, paste_y + new_h))`.
     * Dimension Cap: Cap longest side to 1024 px (`overlay_w`, `overlay_h`).
     * Resize `cam_cropped` to `(overlay_w, overlay_h)` with bilinear interpolation:
       `cam_resized = np.array(cam_cropped.resize((overlay_w, overlay_h), Image.Resampling.BILINEAR)) / 255.0`.
  5. **Colormap Transformation ("Jet")**:
     * Matplotlib `jet` colormap applied to float values $[0.0, 1.0]$.
     * For 100% numerical parity with Python Matplotlib, the 256-entry RGB lookup table from `plt.get_cmap("jet")` (768 bytes total) is embedded directly into `colormap.js`.
  6. **Alpha Blending**:
     * Original image converted to RGB, resized to `(overlay_w, overlay_h)` if capped, float32 in $[0.0, 1.0]$: `orig_rgb = np.asarray(base_img, dtype=np.float32) / 255.0`.
     * Blend factor: $\alpha = 0.45$.
       $$\text{blended} = (1.0 - \alpha) \cdot \text{orig\_rgb} + \alpha \cdot \text{colored\_heatmap}$$
     * Scaled to uint8 $[0, 255]$: `overlay_uint8 = np.uint8(np.clip(blended * 255.0, 0, 255))`.
  7. **Base64 PNG Export**:
     * Encoded as standard PNG and output as Data URI: `data:image/png;base64,...`.

---

## 2. Web App Inspection & Inference Adapter

### 2.1 Current Web App Implementation (`web/src/pages/AnalyzePage.jsx`)
* **API Base URL**:
  * Line 147: `const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';`
* **Network Interactions**:
  1. `GET ${apiUrl}/samples` (Lines 180–201): Fetches catalog of demo samples on mount.
  2. `GET ${apiUrl}/samples/${id}/predict` (Lines 325–331): Guest sample execution (no auth headers).
  3. `POST ${apiUrl}/predict` (Lines 403–410): Custom upload execution. Sends `FormData` with `Authorization: Bearer ${token}`.
* **Fields Read from API Response**:
  * `result.label` (Lines 1050, 1084, 1120): `"Normal"` or `"Abnormal"`.
  * `result.calibrated_probability` (Lines 448, 1100, 1196): Float $[0.0, 1.0]$.
  * `result.uncalibrated_probability` (Line 1126): Float $[0.0, 1.0]$.
  * `result.gradcam_overlay` (Lines 1358, 1408): Data URI PNG string.
  * `result.finding` / `result.finding_label` (Lines 338, 424): Passed to `buildAnalysisRecord`.
  * `result.model_version` (Lines 340, 426): Default `"1.0.0"`.
  * `result.disclaimer` (Line 1446): Rendered in research notice banner.
* **Firebase Auth & Firestore Workflow**:
  * Auth token is acquired via `const token = await getIdToken()` (`AnalyzePage.jsx:399`).
  * In the serverless architecture, **there is no backend server verifying this token during inference**.
  * Instead, client-side guards in `AnalyzePage.jsx:383-391` check `currentUser` and `currentUser.emailVerified` before allowing custom upload inference.
  * When writing the audit log to Firestore (`users/{uid}/analyses`), Firestore Security Rules independently enforce authenticated user ownership and verified email (`request.auth.token.email_verified == true`).

### 2.2 Proposed Inference Adapter Interface

```typescript
// web/src/inference/types.ts
export interface PredictResponse {
  label: 'Normal' | 'Abnormal';
  finding_label: 'Possible abnormal finding' | 'No abnormal finding flagged';
  threshold_label: 'Possible abnormal finding' | 'No abnormal finding flagged';
  finding: 'Possible abnormal finding' | 'No abnormal finding flagged';
  worded_label: 'Possible abnormal finding' | 'No abnormal finding flagged';
  prediction_code: 0 | 1;
  calibrated_probability: number;
  uncalibrated_probability: number;
  threshold_used: number;
  gradcam_overlay: string; // Base64 data:image/png;base64,...
  model_version: string;
  limitations: string;
  projection_hint: string;
  disclaimer: string;
}

export interface InferenceAdapter {
  isReady(): boolean;
  initialize(onProgress?: (percent: number, status: string) => void): Promise<void>;
  predict(imageInput: File | Blob | HTMLImageElement | string): Promise<PredictResponse>;
  listSamples(): Promise<Array<{ id: string; title: string; desc: string; expectedLabel: string; thumbnail_url: string }>>;
}
```

#### Adapter Architecture: Dual Backends Behind One Interface
* **Two Backends Supported**:
  1. `"browser"` (Default): Executes `insightx_gradcam.onnx` directly in the browser via `onnxruntime-web` (WebAssembly SIMD). Zero backend dependencies, zero egress costs.
  2. `"server"`: Existing FastAPI service. Selected **only** if `VITE_API_URL` is set **AND** `GET /health` answers within 3 seconds. Any network failure, timeout, or missing endpoint silently falls back to `"browser"` mode.
* **Unified UI Contract**:
  * Both backends return the identical `PredictResponse` shape.
  * The UI displays an execution badge: `"Ran in your browser"` or `"Ran on server"`.
* **Guest Upload Policy**:
  * Guests may analyze their own uploaded chest radiographs in browser mode with zero data sent over the network or stored.
  * Practitioner login (with verified email) is required solely to persist analyses to the Firestore audit log history.

---

## 3. Feasibility & Risk Analysis (With Real Numbers)

### 3.1 Operator Support in ONNX Runtime Web
* Both models utilize ONNX Opset 13 and only 10 fundamental operators:
  `Add`, `Clip`, `Conv`, `GlobalAveragePool`, `MatMul`, `Mul`, `Pad`, `Sigmoid`, `Squeeze`, `Transpose`.
* All 10 operators have been natively supported in `onnxruntime-web` CPU/WASM backend since ORT v1.8.
* **Unsupported Operators**: **Zero**.
* MobileNetV2 is one of the most thoroughly tested reference architectures in ONNX Runtime Web.

### 3.2 Model Sizes, Measured Compression & Asset Distribution
* **Exact Measured Sizes**:
  * `models/insightx_gradcam.onnx`: Raw = **8,913,759 bytes (8.50 MB)**, Gzip = **8,254,958 bytes (7.87 MB)**.
  * `ort-wasm-simd-threaded.wasm`: Raw = **14,848,852 bytes (14.16 MB)**, Gzip = **3,792,479 bytes (3.62 MB)**.
  * `ort-wasm-simd-threaded.mjs`: Raw = **24,738 bytes**, Gzip = **9,159 bytes**.
  * `ort.wasm.bundle.min.mjs`: Raw = **77,024 bytes**, Gzip = **25,667 bytes**.
  * `gradcam_head_weights.json`: Raw = **34,911 bytes**, Gzip = **13,186 bytes**.
  * `calibration.json`: Raw = **829 bytes**, Gzip = **438 bytes**.
* **Bandwidth & Free Hosting Budget**:
  * Total cold download per user: $7.87\text{ MB} + 3.62\text{ MB} \approx 11.5\text{ MB}$ gzipped.
  * Firebase Hosting Spark (free) plan transfer limit: **360 MB / day** (~30 cold visits/day).
  * **Solution (`VITE_ASSET_BASE_URL`)**:
    * Asset base URL is configurable via `VITE_ASSET_BASE_URL` (default: same origin `""`).
    * Large binary weights (`.onnx` and `.wasm`) and sample PNGs can be optionally hosted on free unmetered object storage or a CDN (e.g. GitHub Releases, Cloudflare R2), reserving Firebase Hosting egress exclusively for HTML/JS app traffic.
* **Workbox / Vite PWA Constraint**:
  * Default Workbox precache limit is **2 MB** (`maximumFileSizeToCacheInBytes = 2097152`).
  * `.onnx` and `.wasm` files must **never** be precached by Workbox; they are excluded from `globPatterns`.
  * Runtime caching is handled dynamically via `CacheStorage` (`caches.open('insightx-model-v1')`).

### 3.3 ONNX Runtime Web Configuration & WASM Distribution
* **Verified Dist Files**:
  * Smallest CPU-WASM entrypoint: `import * as ort from 'onnxruntime-web/wasm'` (resolves to `ort.wasm.bundle.min.mjs`, only **77 KB raw / 25 KB gzipped**; excludes WebGL and WebGPU engines).
  * Required WASM runtime: `ort-wasm-simd-threaded.wasm` (configured to run with `numThreads = 1` for single-thread SIMD execution).
* **Configuration**:
  ```javascript
  import * as ort from 'onnxruntime-web/wasm';

  const assetBase = import.meta.env.VITE_ASSET_BASE_URL || '';
  ort.env.wasm.numThreads = 1; // Single-thread SIMD: eliminates COOP/COEP header constraints
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = `${assetBase}/wasm/`;
  ```

### 3.4 Inference Latency Benchmarks (Estimated)
Based on MobileNetV2 (~300M FLOPs, 224x224x3 input):
* **Desktop / Laptop (Intel i5/i7, AMD Ryzen, Apple M1/M2/M3 - Chrome/Edge/Safari)**:
  * ONNX session execution (`ort-wasm-simd`): **35 ms – 75 ms**.
  * JavaScript Grad-CAM & canvas blending: **15 ms – 25 ms**.
  * Total time: **50 ms – 100 ms** (sub-second instantaneous feedback).
* **Mobile Devices (iPhone 12+, Pixel 6+, Galaxy S21+ - Safari Mobile / Chrome Android)**:
  * ONNX session execution: **120 ms – 260 ms**.
  * Grad-CAM & canvas blending: **30 ms – 50 ms**.
  * Total time: **150 ms – 310 ms**.
* Compared to server-side cold starts (which could take 30–60s on free tiers), in-browser inference provides a vastly superior user experience with zero hosting compute cost.

### 3.5 HEIC File Handling
* `heic2any` conversion (`AnalyzePage.jsx:264-286`) converts iPhone `.heic` photos to JPEG Blob **prior** to any analysis.
* The resulting JPEG/PNG is fed to the canvas preprocessor. HEIC conversion operates identically in the browser-only architecture.

---

## 4. Numerical Parity Testing Plan

To guarantee that client-side inference matches the Python reference implementation down to the fourth decimal place, we establish a strict parity test suite.

### 4.1 Parity Test Architecture
* Script: `tests/parity/test_client_parity.mjs` running under Node.js with `onnxruntime-node` (or headless browser via Playwright).
* Dataset: All 8 sample images in `samples/` (`sample_01_normal.png` through `sample_08_false_negative.png`).
* Gold Standard: Ground truth values precomputed by `api/main.py` and saved in `tests/data/reference_predictions.json`.

### 4.2 Tolerances & Assertions
1. **Raw Logit & Calibrated Probability**:
   $$|p_{\text{python}} - p_{\text{js}}| < 1 \times 10^{-4} \quad (0.0001)$$
2. **Finding Label**:
   $$\text{verdict}_{\text{js}} \equiv \text{verdict}_{\text{python}} \quad (\text{100\% discrete match})$$
3. **Grad-CAM Attention Map**:
   $$|CAM_{\text{python}}[i, j] - CAM_{\text{js}}[i, j]| < 1 \times 10^{-4} \quad \forall (i, j) \in [0..6] \times [0..6]$$

### 4.3 CI Workflow Integration
* Added to `.github/workflows/ci.yml` under the `web` or a new `parity` job:
  ```yaml
  - name: Run Client-Side Numerical Parity Tests
    run: npm run test:parity
  ```
* Fails build if any divergence occurs between Python and JavaScript pipelines.

---

## 5. Firebase Hosting Architecture

### 5.1 `firebase.json` Configuration
Update `firebase.json` to configure Firebase Hosting for the Single Page Application, cache policies for static AI weights, and security headers:

```json
{
  "firestore": {
    "rules": "firestore.rules",
    "indexes": "firestore.indexes.json"
  },
  "hosting": {
    "public": "web/dist",
    "ignore": [
      "firebase.json",
      "**/.*",
      "**/node_modules/**"
    ],
    "rewrites": [
      {
        "source": "**",
        "destination": "/index.html"
      }
    ],
    "headers": [
      {
        "source": "/models/**",
        "headers": [
          {
            "key": "Cache-Control",
            "value": "public, max-age=31536000, immutable"
          },
          {
            "key": "Content-Type",
            "value": "application/octet-stream"
          }
        ]
      },
      {
        "source": "/wasm/**",
        "headers": [
          {
            "key": "Cache-Control",
            "value": "public, max-age=31536000, immutable"
          },
          {
            "key": "Content-Type",
            "value": "application/wasm"
          }
        ]
      },
      {
        "source": "**",
        "headers": [
          {
            "key": "X-Content-Type-Options",
            "value": "nosniff"
          },
          {
            "key": "X-Frame-Options",
            "value": "DENY"
          },
          {
            "key": "Referrer-Policy",
            "value": "strict-origin-when-cross-origin"
          }
        ]
      }
    ]
  },
  "emulators": {
    "auth": { "port": 9099 },
    "firestore": { "port": 8080 },
    "hosting": { "port": 5000 },
    "ui": { "enabled": true, "port": 4000 },
    "singleProjectMode": true
  }
}
```

### 5.2 Firebase Auth Authorized Domains
In Firebase Console > Authentication > Settings > Authorized Domains, ensure the following are present:
1. `localhost` (Default for local development & emulators).
2. `insightx-af8f9.web.app` (Firebase Hosting live subdomain).
3. `insightx-af8f9.firebaseapp.com` (Firebase Auth default domain).

---

## 6. Phased Implementation Roadmap

Each step is self-contained, testable, and independently committable:

```
[Phase 1: Assets]   Copy ONNX models, JSON configs & sample images into web public assets
        │
[Phase 2: Engine]   Implement pure JS preprocessor, Grad-CAM & onnxruntime-web session
        │
[Phase 3: Parity]   Write and verify automated numerical parity test suite (Node vs Python)
        │
[Phase 4: Adapter]  Integrate InferenceAdapter into AnalyzePage.jsx with progress indicator
        │
[Phase 5: Hosting]  Configure Firebase Hosting in firebase.json & verify build & PWA limits
        │
[Phase 6: Docs]     Update README.md documenting both Browser-Only and FastAPI modes
```

### What Remains in the Repository
* **FastAPI Service (`api/main.py`)**: Maintained as the reference architecture, local Docker container option, and Python test harness.
* **Python Test Suite (`tests/`)**: All 45 pytest tests continue to run in CI.
* **Model Files (`models/`)**: Source weights remain the single source of truth.
* **Documentation**: The README explains both the Live WebAssembly Demo and the Python FastAPI REST API.

---

## 7. Decision Points & Considerations

1. **WASM Multi-threading vs. Single-Threaded SIMD**:
   * *Recommendation*: Use single-threaded SIMD (`ort.env.wasm.numThreads = 1`). It executes in ~50–80 ms without requiring Cross-Origin Isolation headers (`COOP`/`COEP`), which can cause issues with Firebase Auth sign-in popups.
2. **Jet Colormap Representation**:
   * *Recommendation*: Embed the precomputed 256-entry RGB lookup table (768 bytes) in `web/src/inference/colormap.js` to guarantee exact numerical parity with Matplotlib's `jet`.
3. **Asset Build Strategy**:
   * *Recommendation*: Use a small pre-build script (`npm run sync-assets`) to copy `models/insightx_gradcam.onnx`, `models/calibration.json`, `models/gradcam_head_weights.json`, and `samples/` into `web/public/` during `npm run build`, avoiding duplicate files tracked in Git.

