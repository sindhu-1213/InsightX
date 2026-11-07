/**
 * Class Activation Mapping (Grad-CAM) pure-JavaScript implementation.
 * Exact mathematical port of api/main.py _compute_numpy_cam_overlay.
 *
 * Implements:
 * 1. Linear combination of conv features with head weights
 * 2. ReLU rectification
 * 3. Max-normalization with 1e-10 epsilon
 * 4. Pillow-compatible bilinear upsampling to 224x224
 * 5. Aspect-preserving crop (removes black padding added during preprocessing)
 * 6. Bilinear resize to overlay dimensions (capped at 1024px)
 * 7. Matplotlib 'jet' colormap lookup
 * 8. Alpha blending with original radiograph (alpha = 0.45)
 * 9. PNG encoding (Browser Canvas or Node.js pngjs)
 */

import { getJetColor } from './colormap.js';

/**
 * Resample a 2D 1-channel uint8 array using separable bilinear interpolation.
 * Matches Pillow's Image.Resampling.BILINEAR exactly.
 *
 * @param {Uint8Array} src Source array (srcW * srcH)
 * @param {number} srcW Source width
 * @param {number} srcH Source height
 * @param {number} dstW Destination width
 * @param {number} dstH Destination height
 * @returns {Uint8Array} Resampled destination array (dstW * dstH)
 */
export function bilinearResize(src, srcW, srcH, dstW, dstH) {
  const scaleX = dstW / srcW;
  const scaleY = dstH / srcH;

  // Horizontal pass: src (srcH x srcW) -> temp (srcH x dstW)
  const temp = new Uint8Array(srcH * dstW);
  for (let x = 0; x < dstW; x++) {
    const center = (x + 0.5) / scaleX;
    const x0 = Math.floor(center - 1.0 + 0.5);
    const x1 = x0 + 1;
    const x0c = Math.max(0, Math.min(srcW - 1, x0));
    const x1c = Math.max(0, Math.min(srcW - 1, x1));
    let w0 = Math.max(0.0, 1.0 - Math.abs(x0 + 0.5 - center));
    let w1 = Math.max(0.0, 1.0 - Math.abs(x1 + 0.5 - center));
    const tot = w0 + w1;
    if (tot > 0) {
      w0 /= tot;
      w1 /= tot;
    }

    for (let y = 0; y < srcH; y++) {
      const rowOffset = y * srcW;
      const v0 = src[rowOffset + x0c];
      const v1 = src[rowOffset + x1c];
      temp[y * dstW + x] = Math.min(255, Math.max(0, Math.round(v0 * w0 + v1 * w1)));
    }
  }

  // Vertical pass: temp (srcH x dstW) -> dst (dstH x dstW)
  const dst = new Uint8Array(dstH * dstW);
  for (let y = 0; y < dstH; y++) {
    const center = (y + 0.5) / scaleY;
    const y0 = Math.floor(center - 1.0 + 0.5);
    const y1 = y0 + 1;
    const y0c = Math.max(0, Math.min(srcH - 1, y0));
    const y1c = Math.max(0, Math.min(srcH - 1, y1));
    let w0 = Math.max(0.0, 1.0 - Math.abs(y0 + 0.5 - center));
    let w1 = Math.max(0.0, 1.0 - Math.abs(y1 + 0.5 - center));
    const tot = w0 + w1;
    if (tot > 0) {
      w0 /= tot;
      w1 /= tot;
    }

    const dstRowOffset = y * dstW;
    const srcRowOffset0 = y0c * dstW;
    const srcRowOffset1 = y1c * dstW;
    for (let x = 0; x < dstW; x++) {
      const v0 = temp[srcRowOffset0 + x];
      const v1 = temp[srcRowOffset1 + x];
      dst[dstRowOffset + x] = Math.min(255, Math.max(0, Math.round(v0 * w0 + v1 * w1)));
    }
  }

  return dst;
}

/**
 * Resample an RGBA image buffer using bilinear interpolation.
 *
 * @param {Uint8Array | Uint8ClampedArray} src Flat RGBA buffer
 * @param {number} srcW
 * @param {number} srcH
 * @param {number} dstW
 * @param {number} dstH
 * @returns {Uint8Array} Flat RGBA buffer
 */
export function bilinearResizeRgba(src, srcW, srcH, dstW, dstH) {
  if (srcW === dstW && srcH === dstH) {
    return new Uint8Array(src);
  }
  const dst = new Uint8Array(dstW * dstH * 4);
  const scaleX = dstW / srcW;
  const scaleY = dstH / srcH;

  for (let y = 0; y < dstH; y++) {
    const center = (y + 0.5) / scaleY;
    const y0 = Math.max(0, Math.min(srcH - 1, Math.floor(center - 1.0 + 0.5)));
    const y1 = Math.max(0, Math.min(srcH - 1, y0 + 1));
    let wy0 = Math.max(0.0, 1.0 - Math.abs(y0 + 0.5 - center));
    let wy1 = Math.max(0.0, 1.0 - Math.abs(y1 + 0.5 - center));
    const toty = wy0 + wy1 || 1.0;
    wy0 /= toty;
    wy1 /= toty;

    const dstRowOffset = y * dstW * 4;
    const rowOffset0 = y0 * srcW * 4;
    const rowOffset1 = y1 * srcW * 4;

    for (let x = 0; x < dstW; x++) {
      const centerX = (x + 0.5) / scaleX;
      const x0 = Math.max(0, Math.min(srcW - 1, Math.floor(centerX - 1.0 + 0.5)));
      const x1 = Math.max(0, Math.min(srcW - 1, x0 + 1));
      let wx0 = Math.max(0.0, 1.0 - Math.abs(x0 + 0.5 - centerX));
      let wx1 = Math.max(0.0, 1.0 - Math.abs(x1 + 0.5 - centerX));
      const totx = wx0 + wx1 || 1.0;
      wx0 /= totx;
      wx1 /= totx;

      const p00 = rowOffset0 + x0 * 4;
      const p01 = rowOffset0 + x1 * 4;
      const p10 = rowOffset1 + x0 * 4;
      const p11 = rowOffset1 + x1 * 4;

      const dstPixelOffset = dstRowOffset + x * 4;
      for (let c = 0; c < 4; c++) {
        const top = src[p00 + c] * wx0 + src[p01 + c] * wx1;
        const bot = src[p10 + c] * wx0 + src[p11 + c] * wx1;
        dst[dstPixelOffset + c] = Math.min(255, Math.max(0, Math.round(top * wy0 + bot * wy1)));
      }
    }
  }

  return dst;
}

/**
 * Compute 7x7 raw Grad-CAM activation map and blended overlay.
 *
 * @param {Float32Array} convFeatures Shape [1, 7, 7, 1280] or [7, 7, 1280]
 * @param {Float32Array | number[]} weights 1280 head weights
 * @param {object} cropInfo { origW, origH, newW, newH, pasteX, pasteY, targetW, targetH }
 * @param {object} [origImage] Optional { data: Uint8Array RGBA, width, height }
 * @param {number} [alpha=0.45] Alpha blend factor
 * @returns {Promise<{ cam7x7: number[][], overlayDataUri: string, overlayWidth: number, overlayHeight: number, blendedRgba: Uint8Array }>}
 */
export async function computeGradCam(
  convFeatures,
  weights,
  cropInfo,
  origImage = null,
  alpha = 0.45
) {
  // Step 1: Linear combination & ReLU
  const cam7x7Raw = new Float64Array(49);
  let maxVal = 0.0;

  for (let i = 0; i < 7; i++) {
    for (let j = 0; j < 7; j++) {
      const cellIdx = i * 7 + j;
      const featOffset = cellIdx * 1280;
      let sum = 0.0;
      for (let k = 0; k < 1280; k++) {
        sum += convFeatures[featOffset + k] * weights[k];
      }
      const val = Math.max(0.0, sum);
      cam7x7Raw[cellIdx] = val;
      if (val > maxVal) maxVal = val;
    }
  }

  // Step 2: Normalize and scale to uint8
  const cam7x7Uint8 = new Uint8Array(49);
  const cam7x7Normalized = [];
  for (let i = 0; i < 7; i++) {
    const row = [];
    for (let j = 0; j < 7; j++) {
      const idx = i * 7 + j;
      const normVal = maxVal > 0.0 ? cam7x7Raw[idx] / (maxVal + 1e-10) : 0.0;
      row.push(normVal);
      cam7x7Uint8[idx] = Math.min(255, Math.max(0, Math.round(normVal * 255.0)));
    }
    cam7x7Normalized.push(row);
  }

  // Step 3: Bilinear upsampling to full 224x224 padded canvas
  const camCanvas = bilinearResize(cam7x7Uint8, 7, 7, 224, 224);

  // Step 4: Crop out black padding region added in preprocessing
  const { origW, origH, newW, newH, pasteX, pasteY } = cropInfo;
  const camCropped = new Uint8Array(newW * newH);
  for (let y = 0; y < newH; y++) {
    const srcOffset = (pasteY + y) * 224 + pasteX;
    const dstOffset = y * newW;
    camCropped.set(camCanvas.subarray(srcOffset, srcOffset + newW), dstOffset);
  }

  // Step 5: Determine target overlay dimensions (capped at 1024px longest side)
  const maxOverlayDim = 1024;
  let overlayW = origW;
  let overlayH = origH;
  if (Math.max(origW, origH) > maxOverlayDim) {
    if (origW >= origH) {
      overlayW = maxOverlayDim;
      overlayH = Math.max(1, Math.round((origH * maxOverlayDim) / origW));
    } else {
      overlayH = maxOverlayDim;
      overlayW = Math.max(1, Math.round((origW * maxOverlayDim) / origH));
    }
  }

  // Step 6: Bilinear resize cropped CAM to target overlay dimensions
  const camResized = bilinearResize(camCropped, newW, newH, overlayW, overlayH);

  // Step 7: Prepare base image RGBA
  let baseRgba = null;
  if (origImage && origImage.data) {
    if (origImage.width !== overlayW || origImage.height !== overlayH) {
      baseRgba = bilinearResizeRgba(origImage.data, origImage.width, origImage.height, overlayW, overlayH);
    } else {
      baseRgba = new Uint8Array(origImage.data);
    }
  }

  // Step 8: Apply Jet Colormap and Alpha Blend
  const blendedRgba = new Uint8Array(overlayW * overlayH * 4);
  const oneMinusAlpha = 1.0 - alpha;

  for (let i = 0; i < overlayW * overlayH; i++) {
    const heatVal = camResized[i] / 255.0;
    const [jetR, jetG, jetB] = getJetColor(heatVal);
    const pixelOffset = i * 4;

    if (baseRgba) {
      const origR = baseRgba[pixelOffset];
      const origG = baseRgba[pixelOffset + 1];
      const origB = baseRgba[pixelOffset + 2];

      blendedRgba[pixelOffset] = Math.min(255, Math.max(0, Math.round(oneMinusAlpha * origR + alpha * jetR)));
      blendedRgba[pixelOffset + 1] = Math.min(255, Math.max(0, Math.round(oneMinusAlpha * origG + alpha * jetG)));
      blendedRgba[pixelOffset + 2] = Math.min(255, Math.max(0, Math.round(oneMinusAlpha * origB + alpha * jetB)));
      blendedRgba[pixelOffset + 3] = 255;
    } else {
      blendedRgba[pixelOffset] = jetR;
      blendedRgba[pixelOffset + 1] = jetG;
      blendedRgba[pixelOffset + 2] = jetB;
      blendedRgba[pixelOffset + 3] = 255;
    }
  }

  // Step 9: Encode to PNG Data URI
  let overlayDataUri = '';
  if (typeof document !== 'undefined' && document.createElement) {
    const canvas = document.createElement('canvas');
    canvas.width = overlayW;
    canvas.height = overlayH;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(overlayW, overlayH);
    imgData.data.set(blendedRgba);
    ctx.putImageData(imgData, 0, 0);
    overlayDataUri = canvas.toDataURL('image/png');
  } else if (typeof globalThis.process !== 'undefined' && globalThis.process?.versions?.node) {
    // Node.js environment: use pngjs to generate base64 data URI
    try {
      const pkg = 'png' + 'js';
      const { PNG } = await import(/* @vite-ignore */ pkg);
      const png = new PNG({ width: overlayW, height: overlayH });
      png.data = globalThis.Buffer.from(blendedRgba);
      const buffer = PNG.sync.write(png);
      overlayDataUri = `data:image/png;base64,${buffer.toString('base64')}`;
    } catch {
      // Fallback if pngjs not present
      overlayDataUri = '';
    }
  }

  return {
    cam7x7: cam7x7Normalized,
    overlayDataUri,
    overlayWidth: overlayW,
    overlayHeight: overlayH,
    blendedRgba
  };
}
