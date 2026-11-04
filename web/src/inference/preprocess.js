/**
 * Preprocessing pipeline for InsightX chest radiograph inference.
 * Exact pure-JavaScript port of insightx.preprocess (Pillow-compatible).
 *
 * Runs identically in the browser and in Node.js.
 */

const PI = Math.PI;

/**
 * 3-lobe Lanczos kernel.
 * @param {number} x
 * @returns {number}
 */
function lanczos3(x) {
  const ax = Math.abs(x);
  if (ax < 1e-7) return 1.0;
  if (ax >= 3.0) return 0.0;
  const pix = PI * x;
  return (Math.sin(pix) / pix) * (Math.sin(pix / 3.0) / (pix / 3.0));
}

/**
 * Generate 1D resampling filter coefficients (support scaled for downscaling).
 * Matches Pillow's ImagingResample table generator.
 */
function getFilterCoefficients(srcSize, dstSize) {
  const scale = dstSize / srcSize;
  const filterScale = scale < 1.0 ? 1.0 / scale : 1.0;
  const support = 3.0 * filterScale;
  const coeffs = new Array(dstSize);

  for (let y = 0; y < dstSize; y++) {
    const center = (y + 0.5) / scale;
    const xmin = Math.max(0, Math.floor(center - support + 0.5));
    const xmax = Math.min(srcSize, Math.ceil(center + support + 0.5));
    const count = xmax - xmin;
    const weights = new Float64Array(count);
    let total = 0.0;

    for (let i = 0; i < count; i++) {
      const srcX = xmin + i;
      const w = lanczos3((srcX + 0.5 - center) / filterScale);
      weights[i] = w;
      total += w;
    }

    if (total !== 0.0) {
      for (let i = 0; i < count; i++) {
        weights[i] /= total;
      }
    }

    coeffs[y] = { xmin, xmax, weights };
  }

  return coeffs;
}

/**
 * Resize a 1-channel grayscale image using separable 2-pass Lanczos3.
 * Matches PIL.Image.Resampling.LANCZOS.
 *
 * @param {Uint8Array} src 1-channel grayscale source array (width * height)
 * @param {number} srcW Source width
 * @param {number} srcH Source height
 * @param {number} dstW Destination width
 * @param {number} dstH Destination height
 * @returns {Uint8Array} Resized grayscale array (dstW * dstH)
 */
export function lanczosResizeGrayscale(src, srcW, srcH, dstW, dstH) {
  const hCoeffs = getFilterCoefficients(srcW, dstW);
  const vCoeffs = getFilterCoefficients(srcH, dstH);

  // Horizontal pass: src (srcH x srcW) -> temp (srcH x dstW)
  const temp = new Uint8Array(srcH * dstW);
  for (let y = 0; y < srcH; y++) {
    const srcRowOffset = y * srcW;
    const dstRowOffset = y * dstW;
    for (let x = 0; x < dstW; x++) {
      const { xmin, weights } = hCoeffs[x];
      let sum = 0.0;
      for (let i = 0; i < weights.length; i++) {
        sum += src[srcRowOffset + xmin + i] * weights[i];
      }
      temp[dstRowOffset + x] = Math.min(255, Math.max(0, Math.round(sum)));
    }
  }

  // Vertical pass: temp (srcH x dstW) -> dst (dstH x dstW)
  const dst = new Uint8Array(dstH * dstW);
  for (let y = 0; y < dstH; y++) {
    const { xmin: ymin, weights } = vCoeffs[y];
    const dstRowOffset = y * dstW;
    for (let x = 0; x < dstW; x++) {
      let sum = 0.0;
      for (let j = 0; j < weights.length; j++) {
        sum += temp[(ymin + j) * dstW + x] * weights[j];
      }
      dst[dstRowOffset + x] = Math.min(255, Math.max(0, Math.round(sum)));
    }
  }

  return dst;
}

/**
 * Convert RGBA image data to 1-channel grayscale using Pillow's exact integer L formula:
 * L = (299 * R + 587 * G + 114 * B + 500) // 1000
 *
 * @param {Uint8Array | Uint8ClampedArray} rgbaData Flat RGBA pixel array
 * @param {number} totalPixels Number of pixels (width * height)
 * @returns {Uint8Array} Grayscale pixel array
 */
export function rgbaToGrayscale(rgbaData, totalPixels) {
  const gray = new Uint8Array(totalPixels);
  for (let i = 0; i < totalPixels; i++) {
    const offset = i * 4;
    const r = rgbaData[offset];
    const g = rgbaData[offset + 1];
    const b = rgbaData[offset + 2];
    gray[i] = ((299 * r + 587 * g + 114 * b + 500) / 1000) | 0;
  }
  return gray;
}

/**
 * Preprocess an image for MobileNetV2 ONNX inference.
 * Reuses the exact insightx.preprocess pipeline:
 * 1. Convert to grayscale.
 * 2. Aspect-preserving resize with Lanczos3.
 * 3. Centered placement on 224x224 canvas with black padding.
 * 4. Replicate grayscale to 3 channels.
 * 5. Normalize: (x / 127.5) - 1.0 -> [-1.0, 1.0].
 *
 * @param {{ data: Uint8Array | Uint8ClampedArray, width: number, height: number }} imageObj
 * @param {[number, number]} targetSize [width, height], default [224, 224]
 * @returns {{ tensor: Float32Array, cropInfo: object, canvasGray: Uint8Array }}
 */
export function preprocessImage(imageObj, targetSize = [224, 224]) {
  const { data, width, height } = imageObj;
  const [targetW, targetH] = targetSize;

  // Step 1: Grayscale
  const gray = rgbaToGrayscale(data, width * height);

  // Step 2: Aspect-preserving dimensions
  const scale = Math.min(targetW / width, targetH / height);
  const newW = Math.max(1, Math.round(width * scale));
  const newH = Math.max(1, Math.round(height * scale));

  // Step 3: Lanczos3 resize
  const resized = lanczosResizeGrayscale(gray, width, height, newW, newH);

  // Step 4: Centered black canvas padding
  const pasteX = Math.floor((targetW - newW) / 2);
  const pasteY = Math.floor((targetH - newH) / 2);

  const canvasGray = new Uint8Array(targetW * targetH); // filled with 0 (black)
  for (let y = 0; y < newH; y++) {
    const srcOffset = y * newW;
    const dstOffset = (pasteY + y) * targetW + pasteX;
    canvasGray.set(resized.subarray(srcOffset, srcOffset + newW), dstOffset);
  }

  // Step 5: 3-channel NHWC Float32Array: (x / 127.5) - 1.0
  const tensor = new Float32Array(1 * targetH * targetW * 3);
  for (let i = 0; i < targetW * targetH; i++) {
    const normVal = (canvasGray[i] / 127.5) - 1.0;
    const tOffset = i * 3;
    tensor[tOffset] = normVal;
    tensor[tOffset + 1] = normVal;
    tensor[tOffset + 2] = normVal;
  }

  const cropInfo = {
    origW: width,
    origH: height,
    newW,
    newH,
    pasteX,
    pasteY,
    targetW,
    targetH
  };

  return { tensor, cropInfo, canvasGray };
}

