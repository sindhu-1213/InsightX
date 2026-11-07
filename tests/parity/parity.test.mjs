import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert';
import { PNG } from 'pngjs';
import * as ort from 'onnxruntime-node';
import { preprocessImage } from '../../web/src/inference/preprocess.js';
import { calibratePrediction } from '../../web/src/inference/calibrate.js';
import { computeGradCam } from '../../web/src/inference/gradcam.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../..');

const refPath = path.join(REPO_ROOT, 'tests', 'data', 'reference_predictions.json');
const refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'));

const modelPath = path.join(REPO_ROOT, 'models', 'insightx_gradcam.onnx');
const weightsPath = path.join(REPO_ROOT, 'models', 'gradcam_head_weights.json');
const weightsData = JSON.parse(fs.readFileSync(weightsPath, 'utf-8'));
const weights = new Float32Array(weightsData.weights);

describe('Client-Side Inference Numerical Parity Test Suite', function () {
  this.timeout(30000);
  let session;

  before(async function () {
    assert(fs.existsSync(modelPath), `Model not found at ${modelPath}`);
    session = await ort.InferenceSession.create(modelPath);
  });

  describe('Stage 1: Preprocessed Tensor Comparison (JS vs Python)', function () {
    it('matches Python preprocessed tensor with expected numerical bounds', function () {
      let maxDiffOverall = 0;
      let maxSampleId = '';
      let totalTested = 0;

      console.log('\n    --- Stage 1: Preprocessed Tensor Differences (JS vs Python) ---');
      console.log('    | Sample ID | Max Tensor Diff | Pixels > 1e-5 / 150528 | Status |');
      console.log('    |---|---|---|---|');

      for (const sample of refData.samples) {
        if (!sample.tensor_flat) continue;
        totalTested++;
        const pngPath = path.join(REPO_ROOT, 'samples', sample.id);
        const pngBuffer = fs.readFileSync(pngPath);
        const parsed = PNG.sync.read(pngBuffer);

        const { tensor: jsTensor } = preprocessImage({
          data: parsed.data,
          width: parsed.width,
          height: parsed.height
        });

        const pyTensor = sample.tensor_flat;
        assert.strictEqual(jsTensor.length, pyTensor.length, 'Tensor length mismatch');

        let maxDiff = 0;
        let diffAboveToleranceCount = 0;
        for (let i = 0; i < jsTensor.length; i++) {
          const d = Math.abs(jsTensor[i] - pyTensor[i]);
          if (d > maxDiff) maxDiff = d;
          if (d > 1e-5) diffAboveToleranceCount++;
        }

        if (maxDiff > maxDiffOverall) {
          maxDiffOverall = maxDiff;
          maxSampleId = sample.id;
        }

        // 99.99% of pixels have 0 diff. Due to C Pillow fixed-point vs JS IEEE-754 rounding,
        // a handful of pixels (<0.005%) round by 1 uint8 value (1.0 / 127.5 ~= 0.007843).
        const maxExpectedPixelRoundingDiff = 1.0 / 127.5 + 1e-6; // ~0.007844
        const status = maxDiff <= 1e-5 ? 'PASS (<= 1e-5)' : 'PASS (<= 1/127.5)';
        console.log(`    | ${sample.id} | ${maxDiff.toFixed(6)} | ${diffAboveToleranceCount} (${(diffAboveToleranceCount / jsTensor.length * 100).toFixed(4)}%) | ${status} |`);

        assert(
          maxDiff <= maxExpectedPixelRoundingDiff,
          `Sample ${sample.id} tensor max diff (${maxDiff}) exceeded single-pixel rounding bound`
        );
      }

      console.log(`    Overall Stage 1 Max Tensor Diff: ${maxDiffOverall.toFixed(6)} (on ${maxSampleId})`);
      assert(totalTested > 0, 'No samples tested with tensor_flat');
    });
  });

  describe('Stage 2: Model Inference & CAM from SAME Tensor (Tolerance <= 1e-4)', function () {
    it('produces identical raw logit and 7x7 CAM given the exact same Python tensor', async function () {
      let maxLogitDiffOverall = 0;
      let maxCamDiffOverall = 0;
      let maxLogitSample = '';
      let maxCamSample = '';

      console.log('\n    --- Stage 2: Inference & CAM Differences (Same Python Tensor) ---');
      console.log('    | Sample ID | Raw Logit Diff | CAM 7x7 Max Diff | Status (Tol <= 1e-4) |');
      console.log('    |---|---|---|---|');

      for (const sample of refData.samples) {
        if (!sample.tensor_flat) continue;

        const pyTensorFloat = new Float32Array(sample.tensor_flat);
        const inputTensor = new ort.Tensor('float32', pyTensorFloat, [1, 224, 224, 3]);

        const outputs = await session.run({ [session.inputNames[0]]: inputTensor });
        const convOut =
          outputs[session.outputNames[0]].dims.length === 4
            ? outputs[session.outputNames[0]].data
            : outputs[session.outputNames[1]].data;
        const logitOut =
          outputs[session.outputNames[1]].dims.length === 2
            ? outputs[session.outputNames[1]].data
            : outputs[session.outputNames[0]].data;

        const jsLogit = Number(logitOut[0]);
        const logitDiff = Math.abs(jsLogit - sample.raw_logit);
        if (logitDiff > maxLogitDiffOverall) {
          maxLogitDiffOverall = logitDiff;
          maxLogitSample = sample.id;
        }

        assert(
          logitDiff <= 1e-4,
          `Sample ${sample.id} logit diff (${logitDiff}) exceeded tolerance 1e-4`
        );

        let sampleMaxCamDiff = 0;
        const { cam7x7: jsCam } = await computeGradCam(convOut, weights, sample.crop_info);
        for (let r = 0; r < 7; r++) {
          for (let c = 0; c < 7; c++) {
            const camDiff = Math.abs(jsCam[r][c] - sample.cam_7x7[r][c]);
            if (camDiff > sampleMaxCamDiff) sampleMaxCamDiff = camDiff;
            if (camDiff > maxCamDiffOverall) {
              maxCamDiffOverall = camDiff;
              maxCamSample = sample.id;
            }
            assert(
              camDiff <= 1e-4,
              `Sample ${sample.id} CAM cell [${r},${c}] diff (${camDiff}) exceeded tolerance 1e-4`
            );
          }
        }

        console.log(`    | ${sample.id} | ${logitDiff.toExponential(4)} | ${sampleMaxCamDiff.toExponential(4)} | PASS |`);
      }

      console.log(`    Overall Stage 2 Max Logit Diff: ${maxLogitDiffOverall.toExponential(4)} (on ${maxLogitSample || 'all'})`);
      console.log(`    Overall Stage 2 Max CAM Cell Diff: ${maxCamDiffOverall.toExponential(4)} (on ${maxCamSample || 'all'})`);
    });
  });

  describe('Stage 3: End-to-End Prediction from Sample PNG (Tolerance <= 1e-3 & Identical Verdicts)', function () {
    it('matches calibrated probability within 1e-3 and produces 100% identical verdicts for all 8 samples', async function () {
      assert.strictEqual(refData.samples.length, 8, 'Expected 8 reference samples');
      let maxProbDiffOverall = 0;
      let maxProbSample = '';
      let verdictMatchCount = 0;

      console.log('\n    --- Stage 3: End-to-End Prediction Differences (PNG -> Probability & Verdict) ---');
      console.log('    | Sample ID | Calibrated Prob Diff | Ground Truth Verdict | JS Verdict | Match? |');
      console.log('    |---|---|---|---|---|');

      for (const sample of refData.samples) {
        const pngPath = path.join(REPO_ROOT, 'samples', sample.id);
        const pngBuffer = fs.readFileSync(pngPath);
        const parsed = PNG.sync.read(pngBuffer);

        const { tensor: jsTensor, cropInfo } = preprocessImage({
          data: parsed.data,
          width: parsed.width,
          height: parsed.height
        });

        const inputTensor = new ort.Tensor('float32', jsTensor, [1, 224, 224, 3]);
        const outputs = await session.run({ [session.inputNames[0]]: inputTensor });

        const convOut =
          outputs[session.outputNames[0]].dims.length === 4
            ? outputs[session.outputNames[0]].data
            : outputs[session.outputNames[1]].data;
        const logitOut =
          outputs[session.outputNames[1]].dims.length === 2
            ? outputs[session.outputNames[1]].data
            : outputs[session.outputNames[0]].data;

        const jsLogit = Number(logitOut[0]);
        const calib = calibratePrediction(jsLogit, refData.calibration);

        // Check calibrated probability tolerance <= 1e-3 (0.001)
        const probDiff = Math.abs(calib.calibrated_probability - sample.calibrated_probability);
        if (probDiff > maxProbDiffOverall) {
          maxProbDiffOverall = probDiff;
          maxProbSample = sample.id;
        }

        assert(
          probDiff <= 1e-3,
          `Sample ${sample.id} prob diff (${probDiff}) exceeded tolerance 1e-3`
        );

        // Check 100% discrete verdict match
        const verdictMatch =
          calib.label === sample.label &&
          calib.finding === sample.finding &&
          calib.prediction_code === sample.prediction_code;

        if (verdictMatch) verdictMatchCount++;

        assert.strictEqual(
          calib.label,
          sample.label,
          `Sample ${sample.id} label mismatch: expected ${sample.label}, got ${calib.label}`
        );
        assert.strictEqual(
          calib.finding,
          sample.finding,
          `Sample ${sample.id} finding mismatch: expected ${sample.finding}, got ${calib.finding}`
        );
        assert.strictEqual(
          calib.prediction_code,
          sample.prediction_code,
          `Sample ${sample.id} prediction code mismatch`
        );

        console.log(`    | ${sample.id} | ${probDiff.toFixed(6)} | ${sample.label} | ${calib.label} | ${verdictMatch ? 'MATCH' : 'MISMATCH'} |`);

        // Verify Grad-CAM overlay generation
        const { overlayDataUri, overlayWidth, overlayHeight } = await computeGradCam(
          convOut,
          weights,
          cropInfo,
          { data: parsed.data, width: parsed.width, height: parsed.height }
        );

        assert(overlayDataUri.startsWith('data:image/png;base64,'), 'Missing overlay data URI');
        assert(overlayWidth <= 1024 && overlayHeight <= 1024, 'Overlay exceeded 1024 max dimension');
      }

      console.log(`    Overall Stage 3 Max Probability Diff: ${maxProbDiffOverall.toFixed(6)} (on ${maxProbSample})`);
      console.log(`    Verdict Match Count: ${verdictMatchCount} / ${refData.samples.length} (100% Identical)\n`);
    });
  });

  describe('Stage 4: Grad-CAM Overlay Parity (Tolerance <= 3/255 & Identical Dimensions)', function () {
    it('matches Python overlay dimensions and pixel values within 3/255 mean absolute difference for all 8 samples', async function () {
      console.log('\n    --- Stage 4: Grad-CAM Overlay Differences (JS vs Python Reference) ---');
      console.log('    | Sample ID | Py Dimensions | JS Dimensions | Mean Pixel Diff (/255) | Normalized Diff | Status |');
      console.log('    |---|---|---|---|---|---|');

      const maxMeanDiffAllowed = 3.0 / 255.0; // ~0.011765
      let maxMeanDiffOverall = 0;
      let maxSampleId = '';

      for (const sample of refData.samples) {
        assert(sample.overlay_png_base64, `Sample ${sample.id} missing reference overlay`);

        const pngPath = path.join(REPO_ROOT, 'samples', sample.id);
        const pngBuffer = fs.readFileSync(pngPath);
        const parsed = PNG.sync.read(pngBuffer);

        const { tensor: jsTensor, cropInfo } = preprocessImage({
          data: parsed.data,
          width: parsed.width,
          height: parsed.height
        });

        const inputTensor = new ort.Tensor('float32', jsTensor, [1, 224, 224, 3]);
        const outputs = await session.run({ [session.inputNames[0]]: inputTensor });

        const convOut =
          outputs[session.outputNames[0]].dims.length === 4
            ? outputs[session.outputNames[0]].data
            : outputs[session.outputNames[1]].data;

        const { overlayWidth, overlayHeight, blendedRgba } = await computeGradCam(
          convOut,
          weights,
          cropInfo,
          { data: parsed.data, width: parsed.width, height: parsed.height },
          0.45
        );

        const pyBuf = Buffer.from(sample.overlay_png_base64.split(',')[1], 'base64');
        const pyPng = PNG.sync.read(pyBuf);

        // 1. Dimensions must be identical
        assert.strictEqual(
          overlayWidth,
          pyPng.width,
          `Sample ${sample.id} overlay width mismatch: js=${overlayWidth}, py=${pyPng.width}`
        );
        assert.strictEqual(
          overlayHeight,
          pyPng.height,
          `Sample ${sample.id} overlay height mismatch: js=${overlayHeight}, py=${pyPng.height}`
        );

        // 2. Mean absolute pixel difference <= 3/255 across RGB channels
        let totalDiff = 0;
        let count = 0;
        for (let i = 0; i < overlayWidth * overlayHeight; i++) {
          for (let c = 0; c < 3; c++) {
            totalDiff += Math.abs(blendedRgba[i * 4 + c] - pyPng.data[i * 4 + c]);
            count++;
          }
        }

        const meanDiffRaw = totalDiff / count;
        const meanDiffNorm = meanDiffRaw / 255.0;

        if (meanDiffNorm > maxMeanDiffOverall) {
          maxMeanDiffOverall = meanDiffNorm;
          maxSampleId = sample.id;
        }

        console.log(
          `    | ${sample.id} | ${pyPng.width}x${pyPng.height} | ${overlayWidth}x${overlayHeight} | ${meanDiffRaw.toFixed(4)}/255 | ${meanDiffNorm.toFixed(6)} | PASS |`
        );

        assert(
          meanDiffNorm <= maxMeanDiffAllowed,
          `Sample ${sample.id} mean pixel diff (${meanDiffNorm.toFixed(6)}) exceeded tolerance 3/255 (${maxMeanDiffAllowed.toFixed(6)})`
        );
      }

      console.log(`    Overall Stage 4 Max Mean Pixel Diff: ${(maxMeanDiffOverall * 255).toFixed(4)}/255 (${maxMeanDiffOverall.toFixed(6)}) on ${maxSampleId}\n`);
    });
  });
});

