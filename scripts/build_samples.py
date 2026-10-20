"""Script to select and build 8 real test-split frontal sample images and manifest.json.

Requirements:
- TEST split only, Frontal projection only.
- Fixed random seed (seed=42).
- 3 correctly predicted Normal (TN).
- 3 correctly predicted Abnormal (TP).
- 1 false positive (FP: ground-truth Normal, predicted Abnormal).
- 1 false negative (FN: ground-truth Abnormal, predicted Normal).
- Resized to 1024 px longest side, saved as PNG.
- Write samples/manifest.json.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from insightx.preprocess import preprocess_image_for_onnx


def resize_to_max_edge(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resize image so longest side equals max_dim, preserving aspect ratio."""
    w, h = img.size
    scale = max_dim / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    calib_path = repo_root / "models" / "calibration.json"
    onnx_path = repo_root / "models" / "insightx.onnx"
    splits_path = repo_root / "data" / "processed" / "splits.csv"
    raw_dir = repo_root / "data" / "raw" / "images_normalized"
    samples_dir = repo_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with open(calib_path, encoding="utf-8") as f:
        calib = json.load(f)
    coef = float(calib["coef"])
    intercept = float(calib["intercept"])
    youden_threshold = float(calib["threshold_youden"])

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    splits_df = pd.read_csv(splits_path)
    test_frontal = splits_df[
        (splits_df["split"] == "test") & (splits_df["projection"] == "Frontal")
    ].copy()

    records = []
    print(f"Running inference on {len(test_frontal)} test frontal images...")
    for _, row in test_frontal.iterrows():
        fname = row["filename"]
        raw_file = raw_dir / fname
        with Image.open(raw_file) as img:
            _, tensor = preprocess_image_for_onnx(img)

        raw_prob = float(session.run(None, {input_name: tensor})[0][0, 0])
        p_clip = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        logit = float(np.log(p_clip / (1.0 - p_clip)))
        cal_logit = coef * logit + intercept
        cal_prob = float(1.0 / (1.0 + np.exp(-cal_logit)))
        pred = 1 if cal_prob >= youden_threshold else 0
        gt = int(row["label"])

        records.append({
            "filename": fname,
            "uid": int(row["uid"]),
            "ground_truth": gt,
            "Problems": str(row["Problems"]),
            "uncalibrated_probability": raw_prob,
            "calibrated_probability": cal_prob,
            "prediction": pred,
        })

    df = pd.DataFrame(records)

    # Subsets
    tn_df = df[(df["ground_truth"] == 0) & (df["prediction"] == 0)]
    tp_df = df[(df["ground_truth"] == 1) & (df["prediction"] == 1)]
    fp_df = df[(df["ground_truth"] == 0) & (df["prediction"] == 1)]
    fn_df = df[(df["ground_truth"] == 1) & (df["prediction"] == 0)]

    print(f"Candidate counts: TN={len(tn_df)}, TP={len(tp_df)}, FP={len(fp_df)}, FN={len(fn_df)}")

    # Sample with fixed seed 42
    rng = 42
    selected_tn = tn_df.sample(n=3, random_state=rng)
    selected_tp = tp_df.sample(n=3, random_state=rng)
    selected_fp = fp_df.sample(n=1, random_state=rng)
    selected_fn = fn_df.sample(n=1, random_state=rng)

    samples_meta = [
        {"item": selected_tn.iloc[0], "id": "sample_01_normal", "cat": "True Normal (TN)"},
        {"item": selected_tn.iloc[1], "id": "sample_02_normal", "cat": "True Normal (TN)"},
        {"item": selected_tn.iloc[2], "id": "sample_03_normal", "cat": "True Normal (TN)"},
        {"item": selected_tp.iloc[0], "id": "sample_04_abnormal", "cat": "True Abnormal (TP)"},
        {"item": selected_tp.iloc[1], "id": "sample_05_abnormal", "cat": "True Abnormal (TP)"},
        {"item": selected_tp.iloc[2], "id": "sample_06_abnormal", "cat": "True Abnormal (TP)"},
        {"item": selected_fp.iloc[0], "id": "sample_07_false_positive", "cat": "FP"},
        {"item": selected_fn.iloc[0], "id": "sample_08_false_negative", "cat": "FN"},
    ]

    manifest = []
    print("\nProcessing and saving 1024px PNG sample images:")
    for entry in samples_meta:
        row = entry["item"]
        sample_id = entry["id"]
        fname = row["filename"]
        out_filename = f"{sample_id}.png"
        out_path = samples_dir / out_filename

        with Image.open(raw_dir / fname) as raw_img:
            resized_1024 = resize_to_max_edge(raw_img.convert("RGB"), max_dim=1024)
            resized_1024.save(out_path, format="PNG")

        with Image.open(out_path) as saved_img:
            _, tensor_saved = preprocess_image_for_onnx(saved_img)
        prob_raw = float(session.run(None, {input_name: tensor_saved})[0][0, 0])
        p_clip = np.clip(prob_raw, 1e-6, 1.0 - 1e-6)
        logit = float(np.log(p_clip / (1.0 - p_clip)))
        cal_logit = coef * logit + intercept
        prob_cal = float(1.0 / (1.0 + np.exp(-cal_logit)))
        pred = 1 if prob_cal >= youden_threshold else 0

        gt_label = "Normal" if row["ground_truth"] == 0 else "Abnormal"
        cal_prob = round(prob_cal, 4)
        uncal_prob = round(prob_raw, 4)
        pred_text = (
            "Possible abnormal finding" if pred == 1
            else "No abnormal finding flagged"
        )

        manifest.append({
            "id": out_filename,
            "sample_id": sample_id,
            "source_filename": fname,
            "uid": int(row["uid"]),
            "ground_truth_label": gt_label,
            "Problems": row["Problems"],
            "calibrated_probability": cal_prob,
            "uncalibrated_probability": uncal_prob,
            "prediction_label": pred_text,
            "category": entry["cat"],
            "threshold_used": round(youden_threshold, 4),
        })
        print(f"  - Saved {out_filename} (source: {fname}, GT: {gt_label}, P_cal: {cal_prob:.4f})")

    manifest_path = samples_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
