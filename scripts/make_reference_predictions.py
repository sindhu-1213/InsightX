"""Generate reference predictions for all 8 test samples using the Python pipeline.

Saves golden reference outputs to tests/data/reference_predictions.json for
client-side parity verification.
"""

import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_DIR = REPO_ROOT / "tests" / "data"
OUTPUT_FILE = OUTPUT_DIR / "reference_predictions.json"


def lanczos_pad_preprocess(image: Image.Image, target_size=(224, 224)):
    img = image.convert("L")
    w, h = img.size
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("L", target_size, color=0)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    arr_gray = np.asarray(canvas, dtype=np.float32)
    arr_3ch = np.repeat(arr_gray[:, :, np.newaxis], 3, axis=-1)
    tensor = (arr_3ch / 127.5) - 1.0
    tensor_batch = tensor[np.newaxis, ...]  # (1, 224, 224, 3)

    crop_info = {
        "orig_w": w,
        "orig_h": h,
        "new_w": new_w,
        "new_h": new_h,
        "paste_x": paste_x,
        "paste_y": paste_y,
        "target_w": target_w,
        "target_h": target_h,
    }
    return tensor_batch, crop_info


def compute_7x7_cam(conv_features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cam = np.sum(conv_features * weights.reshape(1, 1, -1), axis=-1)
    cam = np.maximum(cam, 0.0)
    max_val = np.max(cam)
    if max_val > 0:
        cam = cam / (max_val + 1e-10)
    return cam


def compute_numpy_cam_overlay(
    image: Image.Image,
    conv_features: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.45,
) -> str:
    cam = np.sum(conv_features * weights.reshape(1, 1, -1), axis=-1)
    cam = np.maximum(cam, 0.0)
    max_val = np.max(cam)
    if max_val > 0:
        cam = cam / (max_val + 1e-10)

    w, h = image.size
    scale = min(224.0 / w, 224.0 / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    paste_x = (224 - new_w) // 2
    paste_y = (224 - new_h) // 2

    cam_uint8 = np.uint8(np.clip(cam * 255.0, 0, 255))
    cam_canvas = Image.fromarray(cam_uint8).resize((224, 224), Image.Resampling.BILINEAR)
    cam_cropped = cam_canvas.crop((paste_x, paste_y, paste_x + new_w, paste_y + new_h))

    max_overlay_dim = 1024
    if max(w, h) > max_overlay_dim:
        if w >= h:
            overlay_w = max_overlay_dim
            overlay_h = max(1, int(round(h * max_overlay_dim / w)))
        else:
            overlay_h = max_overlay_dim
            overlay_w = max(1, int(round(w * max_overlay_dim / h)))
    else:
        overlay_w, overlay_h = w, h

    cam_resized = np.array(
        cam_cropped.resize((overlay_w, overlay_h), Image.Resampling.BILINEAR)
    ) / 255.0

    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(cam_resized)[..., :3]

    base_img = image.convert("RGB")
    if base_img.size != (overlay_w, overlay_h):
        base_img = base_img.resize((overlay_w, overlay_h), Image.Resampling.BILINEAR)
    orig_rgb = np.asarray(base_img, dtype=np.float32) / 255.0
    blended = (1.0 - alpha) * orig_rgb + alpha * colored_heatmap
    overlay_uint8 = np.uint8(np.clip(blended * 255.0, 0, 255))

    buf = io.BytesIO()
    Image.fromarray(overlay_uint8).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load ONNX model
    onnx_path = MODELS_DIR / "insightx_gradcam.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # 2. Load calibration and weights
    with open(MODELS_DIR / "calibration.json", encoding="utf-8") as f:
        calib_data = json.load(f)
    calib_coef = float(calib_data["coef"])
    calib_intercept = float(calib_data["intercept"])
    threshold = float(calib_data["threshold_youden"])

    with open(MODELS_DIR / "gradcam_head_weights.json", encoding="utf-8") as f:
        weights_data = json.load(f)
    weights = np.array(weights_data["weights"], dtype=np.float32)

    sample_files = sorted(SAMPLES_DIR.glob("sample_*.png"))
    print(f"Generating reference predictions for {len(sample_files)} sample files...")

    results = []
    for sf in sample_files:
        img = Image.open(sf).convert("RGB")
        tensor_batch, crop_info = lanczos_pad_preprocess(img)

        # Tensor hash
        tensor_bytes = tensor_batch.astype(np.float32).tobytes()
        t_hash = hashlib.sha256(tensor_bytes).hexdigest()

        # Run model
        outputs = sess.run(None, {input_name: tensor_batch})
        conv_out = outputs[0] if outputs[0].shape[-1] == 1280 else outputs[1]
        logit_out = outputs[1] if outputs[1].shape[-1] == 1 else outputs[0]

        raw_logit = float(logit_out[0, 0])
        uncalibrated_prob = float(1.0 / (1.0 + np.exp(-raw_logit)))
        calibrated_logit = calib_coef * raw_logit + calib_intercept
        calibrated_prob = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
        calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))

        pred_code = 1 if calibrated_prob >= threshold else 0
        pred_label = "Abnormal" if pred_code == 1 else "Normal"
        worded_label = (
            "Possible abnormal finding"
            if pred_code == 1
            else "No abnormal finding flagged"
        )

        cam7x7 = compute_7x7_cam(conv_out[0], weights)
        overlay_uri = compute_numpy_cam_overlay(img, conv_out[0], weights, alpha=0.45)

        entry = {
            "id": sf.name,
            "width": img.width,
            "height": img.height,
            "crop_info": crop_info,
            "tensor_sha256": t_hash,
            "raw_logit": raw_logit,
            "uncalibrated_probability": round(uncalibrated_prob, 4),
            "uncalibrated_probability_raw": uncalibrated_prob,
            "calibrated_probability": round(calibrated_prob, 4),
            "calibrated_probability_raw": calibrated_prob,
            "label": pred_label,
            "finding": worded_label,
            "prediction_code": pred_code,
            "cam_7x7": cam7x7.tolist(),
            "overlay_png_base64": overlay_uri,
        }

        # Include raw tensor values for sample_01 and sample_04 to allow exact stage 1 testing
        if sf.name in ["sample_01_normal.png", "sample_04_abnormal.png"]:
            entry["tensor_flat"] = tensor_batch.flatten().tolist()

        results.append(entry)
        print(
            f"  {sf.name}: logit={raw_logit:.4f}, prob={calibrated_prob:.4f}, "
            f"verdict={worded_label}"
        )

    output_payload = {
        "calibration": {
            "coef": calib_coef,
            "intercept": calib_intercept,
            "threshold_youden": threshold,
        },
        "samples": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Saved {len(results)} reference samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
