import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

const webPublic = path.join(REPO_ROOT, 'web', 'public');
const modelsDest = path.join(webPublic, 'models');
const samplesDest = path.join(webPublic, 'samples');
const wasmDest = path.join(webPublic, 'wasm');

// Ensure destination directories exist
fs.mkdirSync(modelsDest, { recursive: true });
fs.mkdirSync(samplesDest, { recursive: true });
fs.mkdirSync(wasmDest, { recursive: true });

function copyFileSafe(src, dest) {
  if (!fs.existsSync(src)) {
    console.warn(`[sync-assets] Warning: Source file does not exist: ${src}`);
    return;
  }
  fs.copyFileSync(src, dest);
  const stat = fs.statSync(dest);
  console.log(`[sync-assets] Copied ${path.basename(src)} -> ${path.relative(REPO_ROOT, dest)} (${(stat.size / 1024).toFixed(1)} KB)`);
}

// 1. Copy model artifacts
copyFileSafe(
  path.join(REPO_ROOT, 'models', 'insightx_gradcam.onnx'),
  path.join(modelsDest, 'insightx_gradcam.onnx')
);
copyFileSafe(
  path.join(REPO_ROOT, 'models', 'calibration.json'),
  path.join(modelsDest, 'calibration.json')
);
copyFileSafe(
  path.join(REPO_ROOT, 'models', 'gradcam_head_weights.json'),
  path.join(modelsDest, 'gradcam_head_weights.json')
);

// 2. Copy sample images and manifest
const samplesSrc = path.join(REPO_ROOT, 'samples');
if (fs.existsSync(samplesSrc)) {
  const entries = fs.readdirSync(samplesSrc);
  for (const entry of entries) {
    const srcFile = path.join(samplesSrc, entry);
    const destFile = path.join(samplesDest, entry);
    if (fs.statSync(srcFile).isFile()) {
      fs.copyFileSync(srcFile, destFile);
    }
  }
  console.log(`[sync-assets] Synced ${entries.length} sample items -> web/public/samples/`);
}

// 3. Copy onnxruntime-web wasm files from web/node_modules/onnxruntime-web/dist
const wasmSrcDir = path.join(REPO_ROOT, 'web', 'node_modules', 'onnxruntime-web', 'dist');
if (fs.existsSync(wasmSrcDir)) {
  const wasmFiles = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];
  for (const file of wasmFiles) {
    const srcFile = path.join(wasmSrcDir, file);
    if (fs.existsSync(srcFile)) {
      copyFileSafe(srcFile, path.join(wasmDest, file));
    }
  }
} else {
  console.warn(`[sync-assets] Note: onnxruntime-web/dist not found at ${wasmSrcDir}`);
}

// 4. Sync metrics.json as single source of truth
const metricsSrc = path.join(REPO_ROOT, 'reports', 'metrics.json');
const metricsDest = path.join(REPO_ROOT, 'web', 'src', 'data', 'metrics.json');
if (fs.existsSync(metricsSrc)) {
  fs.mkdirSync(path.dirname(metricsDest), { recursive: true });
  copyFileSafe(metricsSrc, metricsDest);
}

console.log('[sync-assets] Asset synchronization complete.');

