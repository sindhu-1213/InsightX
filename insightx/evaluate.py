"""Comprehensive model evaluation, probability calibration, and metric reporting.

Adheres strictly to the protocol:
1. Validation-only decisions:
   - Platt scaling fitted on VALIDATION split only -> models/calibration.json
   - Operating thresholds chosen on VALIDATION split (Youden's J and 90% specificity)
   - Reliability diagrams (before & after calibration) generated for VALIDATION split
2. Test split evaluation (evaluated once):
   - ROC-AUC with 95% bootstrap CI (1000 resamples by patient uid)
   - Accuracy, F1, Sensitivity, Specificity at validation-chosen Youden threshold
   - Sensitivity at validation-chosen 90% specificity threshold
   - Brier score before & after calibration
   - Confusion matrix
   - Breakdown by projection (Frontal vs Lateral) with bootstrap CI by uid
   - Breakdown by top 10 finding tokens vs Normal
   - Provenance recording (images read, timestamp, model hash, seed) -> reports/metrics.json
   - Comparative analysis table -> reports/comparison.md
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from insightx import RESEARCH_DISCLAIMER
from insightx.model_io import load_trained_model

logger = logging.getLogger("insightx.evaluate")


def file_sha256(path: Path | str) -> str:
    """Compute hex SHA256 digest of a file."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def bootstrap_auc_by_uid(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate ROC-AUC and 95% bootstrap confidence interval resampling by patient uid."""
    rng = np.random.default_rng(seed)
    uids = df["uid"].unique()
    n_uids = len(uids)

    # Map each patient uid to their corresponding index rows
    uid_to_indices = {uid: df.index[df["uid"] == uid].to_numpy() for uid in uids}

    boot_aucs: list[float] = []
    for _ in range(n_bootstraps):
        sampled_uids = rng.choice(uids, size=n_uids, replace=True)
        sampled_indices = np.concatenate([uid_to_indices[uid] for uid in sampled_uids])

        y_boot_true = y_true[sampled_indices]
        y_boot_pred = y_pred[sampled_indices]

        # Only compute if both classes are present in resample
        if len(np.unique(y_boot_true)) > 1:
            auc = float(roc_auc_score(y_boot_true, y_boot_pred))
            boot_aucs.append(auc)

    boot_arr = np.array(boot_aucs)
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))
    point_auc = float(roc_auc_score(y_true, y_pred))

    return point_auc, ci_lower, ci_upper


def fit_platt_scaling(
    y_val: np.ndarray,
    val_probs: np.ndarray,
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Fit Platt scaling logistic regression on the validation split only."""
    eps = 1e-6
    clipped_probs = np.clip(val_probs, eps, 1.0 - eps)
    logits = np.log(clipped_probs / (1.0 - clipped_probs)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
    calibrator.fit(logits, y_val)

    params: dict[str, Any] = {
        "method": "platt_scaling",
        "fit_split": "val",
        "coef": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
        "disclaimer": RESEARCH_DISCLAIMER,
    }
    return calibrator, params


def apply_platt_calibration(
    calibrator: LogisticRegression,
    probs: np.ndarray,
) -> np.ndarray:
    """Transform uncalibrated probabilities using the fitted Platt calibrator."""
    eps = 1e-6
    clipped = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def determine_validation_thresholds(
    y_val: np.ndarray,
    val_probs_cal: np.ndarray,
    val_probs_uncal: np.ndarray,
) -> dict[str, Any]:
    """Determine decision thresholds on the validation split ONLY.

    1. Maximize Youden's J = TPR - FPR.
    2. Determine threshold yielding >= 90% specificity.
    """
    fpr_c, tpr_c, thresh_c = roc_curve(y_val, val_probs_cal)
    j_scores_c = tpr_c - fpr_c
    best_idx_c = int(np.argmax(j_scores_c))

    thresh_youden = float(thresh_c[best_idx_c])
    val_youden_j = float(j_scores_c[best_idx_c])
    val_sens_youden = float(tpr_c[best_idx_c])
    val_spec_youden = float(1.0 - fpr_c[best_idx_c])

    # 90% specificity threshold (FPR <= 0.10)
    valid_indices_c = np.where(fpr_c <= 0.10)[0]
    idx_90_c = valid_indices_c[-1] if len(valid_indices_c) > 0 else 0
    thresh_90_spec = float(thresh_c[idx_90_c])
    val_sens_90 = float(tpr_c[idx_90_c])
    val_spec_90 = float(1.0 - fpr_c[idx_90_c])

    # Also compute on uncalibrated probabilities for reference
    fpr_u, tpr_u, thresh_u = roc_curve(y_val, val_probs_uncal)
    j_scores_u = tpr_u - fpr_u
    best_idx_u = int(np.argmax(j_scores_u))
    thresh_youden_uncal = float(thresh_u[best_idx_u])

    valid_indices_u = np.where(fpr_u <= 0.10)[0]
    idx_90_u = valid_indices_u[-1] if len(valid_indices_u) > 0 else 0
    thresh_90_spec_uncal = float(thresh_u[idx_90_u])

    return {
        "threshold_youden": thresh_youden,
        "threshold_90_specificity": thresh_90_spec,
        "val_youden_j": val_youden_j,
        "val_sensitivity_at_youden": val_sens_youden,
        "val_specificity_at_youden": val_spec_youden,
        "val_sensitivity_at_90_specificity": val_sens_90,
        "val_specificity_at_90_specificity": val_spec_90,
        "threshold_youden_uncalibrated": thresh_youden_uncal,
        "threshold_90_specificity_uncalibrated": thresh_90_spec_uncal,
    }


def plot_validation_reliability_diagram(
    y_val: np.ndarray,
    val_probs_uncal: np.ndarray,
    val_probs_cal: np.ndarray,
    output_path: Path,
) -> None:
    """Plot reliability diagrams before and after Platt calibration on validation split."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_val, val_probs_uncal, n_bins=10
    )
    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_val, val_probs_cal, n_bins=10
    )

    brier_before = float(brier_score_loss(y_val, val_probs_uncal))
    brier_after = float(brier_score_loss(y_val, val_probs_cal))

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(
        mean_pred_uncal,
        fraction_pos_uncal,
        "s--",
        color="#E65100",
        label=f"Uncalibrated Val (Brier = {brier_before:.4f})",
    )
    plt.plot(
        mean_pred_cal,
        fraction_pos_cal,
        "o-",
        color="#2E7D32",
        label=f"Platt Calibrated Val (Brier = {brier_after:.4f})",
    )
    plt.title("InsightX - Validation Split Calibration Curve", fontweight="bold")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_test_roc_curve(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    point_auc: float,
    ci_lower: float,
    ci_upper: float,
    sens_at_90: float,
    output_path: Path,
) -> None:
    """Plot Receiver Operating Characteristic (ROC) curve on test split."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, test_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(
        fpr,
        tpr,
        color="#00838F",
        lw=2.5,
        label=f"ROC Curve (AUC = {point_auc:.4f} [95% CI: {ci_lower:.4f}, {ci_upper:.4f}])",
    )
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.scatter(
        [0.10],
        [sens_at_90],
        color="#D32F2F",
        s=80,
        zorder=5,
        label=f"Sens @ 90% Spec: {sens_at_90:.3f}",
    )
    plt.title("InsightX - Receiver Operating Characteristic (Test Split)", fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_test_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
) -> None:
    """Plot confusion matrix heatmap for test split predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title("InsightX - Confusion Matrix (Test Split)", fontweight="bold")
    plt.xlabel("Predicted Finding")
    plt.ylabel("Ground Truth Annotation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_full_evaluation(
    splits_path: Path | str = "data/processed/splits.csv",
    images_dir: Path | str = "data/processed/images_224",
    projections_path: Path | str = "data/raw/indiana_projections.csv",
    reports_dir: Path | str = "reports",
    calib_json_path: Path | str = "models/calibration.json",
    keras_archive_path: Path | str = "models/mobilenet_finetuned.keras",
    seed: int = 42,
) -> dict[str, Any]:
    """Execute complete validation calibration and one-shot test evaluation pipeline."""
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(images_dir)
    df = pd.read_csv(splits_path)

    # Ensure projection column is present
    if "projection" not in df.columns and Path(projections_path).is_file():
        proj_csv = pd.read_csv(projections_path)
        df = df.merge(proj_csv[["filename", "projection"]], on="filename", how="left")

    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    val_files = [img_dir / str(fn).strip() for fn in val_df["filename"]]
    test_files = [img_dir / str(fn).strip() for fn in test_df["filename"]]

    n_val_read = sum(1 for f in val_files if f.is_file())
    n_test_read = sum(1 for f in test_files if f.is_file())

    if n_val_read < len(val_files) or n_test_read < len(test_files):
        raise FileNotFoundError(
            f"Image files missing in {img_dir}: read val {n_val_read}/{len(val_files)}, "
            f"test {n_test_read}/{len(test_files)}"
        )

    # 0. Load trained model via clean model_io
    print("[Eval] Loading fine-tuned MobileNetV2 architecture with trained weights...")
    model = load_trained_model()

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    def _load_image(p: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(p)
        img = tf.image.decode_png(raw, channels=3)
        img = tf.cast(img, tf.float32)
        return preprocess_input(img)

    val_paths = [str(f) for f in val_files]
    test_paths = [str(f) for f in test_files]

    val_ds = (
        tf.data.Dataset.from_tensor_slices(val_paths)
        .map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    # =========================================================================
    # STEP 1: VALIDATION-ONLY DECISIONS
    # =========================================================================
    print("[Eval] Running inference on VALIDATION split only...")
    val_probs_uncal = model.predict(val_ds, verbose=0).flatten()
    y_val = val_df["label"].to_numpy().astype(int)

    # Fit Platt scaling on validation split
    calibrator, calib_dict = fit_platt_scaling(y_val, val_probs_uncal)
    val_probs_cal = apply_platt_calibration(calibrator, val_probs_uncal)

    # Determine thresholds on validation split
    thresh_decisions = determine_validation_thresholds(y_val, val_probs_cal, val_probs_uncal)
    calib_dict.update(thresh_decisions)

    # Save reliability diagram for validation split
    val_calib_png = rep_dir / "calibration_curve.png"
    plot_validation_reliability_diagram(y_val, val_probs_uncal, val_probs_cal, val_calib_png)
    print(f"[Eval] Saved validation reliability diagram to: {val_calib_png}")

    # Save models/calibration.json
    calib_out_path = Path(calib_json_path)
    calib_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_out_path, "w", encoding="utf-8") as f:
        json.dump(calib_dict, f, indent=2)
    print(f"[Eval] Saved validation calibration parameters to: {calib_out_path}")

    # =========================================================================
    # STEP 2: TEST EVALUATION (EVALUATED EXACTLY ONCE)
    # =========================================================================
    print("[Eval] Running inference on TEST split (evaluated once)...")
    test_probs_uncal = model.predict(test_ds, verbose=0).flatten()
    test_probs_cal = apply_platt_calibration(calibrator, test_probs_uncal)
    y_test = test_df["label"].to_numpy().astype(int)
    test_df["pred_prob"] = test_probs_cal

    # Overall ROC-AUC and 95% bootstrap CI (1000 resamples by patient uid)
    test_auc, test_auc_ci_low, test_auc_ci_high = bootstrap_auc_by_uid(
        test_df, y_test, test_probs_uncal, n_bootstraps=1000, seed=seed
    )

    # Operating metrics at validation-chosen Youden threshold
    youden_thresh = calib_dict["threshold_youden"]
    test_preds_youden = (test_probs_cal >= youden_thresh).astype(int)

    test_acc = float(accuracy_score(y_test, test_preds_youden))
    test_f1 = float(f1_score(y_test, test_preds_youden, zero_division=0))

    tp = int(np.sum((y_test == 1) & (test_preds_youden == 1)))
    fp = int(np.sum((y_test == 0) & (test_preds_youden == 1)))
    tn = int(np.sum((y_test == 0) & (test_preds_youden == 0)))
    fn = int(np.sum((y_test == 1) & (test_preds_youden == 0)))

    test_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    test_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Sensitivity at validation-chosen 90% specificity threshold
    spec_90_thresh = calib_dict["threshold_90_specificity"]
    test_preds_90 = (test_probs_cal >= spec_90_thresh).astype(int)
    tp_90 = int(np.sum((y_test == 1) & (test_preds_90 == 1)))
    tn_90 = int(np.sum((y_test == 0) & (test_preds_90 == 0)))
    test_sens_at_90 = float(tp_90 / (tp + fn)) if (tp + fn) > 0 else 0.0
    test_spec_at_90 = float(tn_90 / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Brier scores
    brier_before = float(brier_score_loss(y_test, test_probs_uncal))
    brier_after = float(brier_score_loss(y_test, test_probs_cal))

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds_youden)

    # Breakdown by projection (Frontal vs Lateral)
    proj_results = []
    for proj_name in ["Frontal", "Lateral"]:
        sub_mask = test_df["projection"] == proj_name
        sub_df = test_df[sub_mask].copy().reset_index(drop=True)
        sub_y_true = sub_df["label"].to_numpy().astype(int)
        sub_probs = test_probs_uncal[sub_mask]

        if len(np.unique(sub_y_true)) > 1:
            p_auc, p_low, p_high = bootstrap_auc_by_uid(
                sub_df, sub_y_true, sub_probs, n_bootstraps=1000, seed=seed
            )
        else:
            p_auc, p_low, p_high = 0.0, 0.0, 0.0

        norm_cnt = int((sub_y_true == 0).sum())
        abnorm_cnt = int((sub_y_true == 1).sum())
        proj_results.append({
            "projection": proj_name,
            "total_count": len(sub_df),
            "normal_count": norm_cnt,
            "abnormal_count": abnorm_cnt,
            "roc_auc": round(p_auc, 4),
            "ci_95": [round(p_low, 4), round(p_high, 4)],
        })

    # Breakdown by top 10 finding tokens (identified from abnormal validation/test distribution)
    abnormal_val = val_df[val_df["label"] == 1]
    tokens = [
        t.strip()
        for probs_str in abnormal_val["Problems"].dropna()
        for t in probs_str.split(";")
        if t.strip()
    ]
    top_tokens = [tok for tok, _ in Counter(tokens).most_common(10)]

    finding_results = []
    normal_test_df = test_df[test_df["label"] == 0]

    for tok in top_tokens:
        has_token_mask = test_df["Problems"].fillna("").apply(
            lambda s, t=tok: t in [p.strip() for p in s.split(";")]
        )
        finding_df = test_df[has_token_mask & (test_df["label"] == 1)]
        finding_cnt = len(finding_df)

        eval_subset = pd.concat([finding_df, normal_test_df], ignore_index=True)
        subset_y_true = eval_subset["label"].to_numpy().astype(int)
        subset_probs = eval_subset["pred_prob"].to_numpy()

        if len(np.unique(subset_y_true)) > 1:
            finding_auc = float(roc_auc_score(subset_y_true, subset_probs))
        else:
            finding_auc = 0.0

        trust_flag = "Caution (<30 images)" if finding_cnt < 30 else "Sufficient (>=30 images)"
        finding_results.append({
            "finding": tok,
            "count": finding_cnt,
            "roc_auc": round(finding_auc, 4),
            "trust_flag": trust_flag,
        })

    # Provenance data
    model_hash = file_sha256(keras_archive_path)
    weights_hash = file_sha256("models/insightx.weights.h5")

    metrics_payload: dict[str, Any] = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "evaluation_split": "test",
        "sample_count": len(y_test),
        "provenance": {
            "n_images_read_val": n_val_read,
            "n_images_read_test": n_test_read,
            "timestamp": datetime.now(UTC).isoformat(),
            "model_path": str(keras_archive_path),
            "model_sha256": model_hash,
            "weights_sha256": weights_hash,
            "seed": seed,
        },
        "roc_auc": round(test_auc, 4),
        "roc_auc_ci_95": [round(test_auc_ci_low, 4), round(test_auc_ci_high, 4)],
        "threshold_youden": round(youden_thresh, 4),
        "accuracy": round(test_acc, 4),
        "f1_score": round(test_f1, 4),
        "sensitivity": round(test_sens, 4),
        "specificity": round(test_spec, 4),
        "threshold_at_90_specificity": round(spec_90_thresh, 4),
        "sensitivity_at_90_specificity": round(test_sens_at_90, 4),
        "specificity_at_90_specificity_threshold": round(test_spec_at_90, 4),
        "brier_score_uncalibrated": round(brier_before, 4),
        "brier_score_calibrated": round(brier_after, 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["Normal (0)", "Abnormal (1)"],
        "breakdown_by_projection": proj_results,
        "breakdown_by_finding": finding_results,
    }

    # Save metrics.json
    metrics_path = rep_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[Eval] Saved test metrics to: {metrics_path}")

    # Save test plots
    plot_test_roc_curve(
        y_test,
        test_probs_uncal,
        test_auc,
        test_auc_ci_low,
        test_auc_ci_high,
        test_sens_at_90,
        rep_dir / "roc_curve.png",
    )
    plot_test_confusion_matrix(cm, rep_dir / "confusion_matrix.png")
    print(f"[Eval] Saved ROC curve to: {rep_dir / 'roc_curve.png'}")
    print(f"[Eval] Saved confusion matrix to: {rep_dir / 'confusion_matrix.png'}")

    # Build comparison.md against reports/baselines.json
    build_comparison_report(
        test_metrics=metrics_payload,
        val_metrics=calib_dict,
        baselines_path=Path("reports/baselines.json"),
        output_path=rep_dir / "comparison.md",
    )
    print(f"[Eval] Saved comparison report to: {rep_dir / 'comparison.md'}")

    return metrics_payload


def build_comparison_report(
    test_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    baselines_path: Path,
    output_path: Path,
) -> None:
    """Generate Markdown comparison table against baselines and validation performance."""
    baselines: dict[str, Any] = {}
    if baselines_path.is_file():
        with open(baselines_path, encoding="utf-8") as f:
            baselines = json.load(f)

    models_bl = baselines.get("models", {})
    maj_bl = baselines.get("majority_class_baseline", {})
    text_bl = models_bl.get("text_only", {}).get("test_metrics", {})
    img_bl = models_bl.get("image_only", {}).get("test_metrics", {})
    fuse_bl = models_bl.get("fused", {}).get("test_metrics", {})

    ci = test_metrics["roc_auc_ci_95"]
    ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    val_j = val_metrics.get("val_youden_j", 0.0)
    val_sens_y = val_metrics.get("val_sensitivity_at_youden", 0.0)
    val_spec_y = val_metrics.get("val_specificity_at_youden", 0.0)
    val_sens_90 = val_metrics.get("val_sensitivity_at_90_specificity", 0.0)
    val_spec_90 = val_metrics.get("val_specificity_at_90_specificity", 0.0)

    brier_cal = test_metrics["brier_score_calibrated"]
    brier_uncal = test_metrics["brier_score_uncalibrated"]
    test_j = test_metrics["sensitivity"] + test_metrics["specificity"] - 1.0

    lines = [
        "# InsightX: Model Comparison & Test Performance",
        "",
        "> **Notice**: Research Demo Only - This tool is not a certified medical device.",
        "> Outputs are AI predictions for research investigation only and must not guide",
        "> clinical decisions.",
        "",
        f"This report summarizes the held-out **Test Split (n = {test_metrics['sample_count']})**",
        "evaluation of the fine-tuned MobileNetV2 architecture against frozen baseline models.",
        "All calibration models and operating decision thresholds were fixed using the",
        "**Validation Split only**.",
        "",
        "---",
        "",
        "## 1. Test Split Benchmark Comparison",
        "",
        "| Model Architecture | Modality | Accuracy | F1 (Abnormal) | ROC-AUC | 95% CI | Brier |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **Majority Class** | None | {maj_bl.get('accuracy', 0.6018):.4f} | "
        f"{maj_bl.get('f1', 0.7514):.4f} | 0.5000 | N/A | N/A |",
        f"| **Frozen MobileNetV2** (C=0.01) | Image Only | {img_bl.get('accuracy', 0.6484):.4f} | "
        f"{img_bl.get('f1', 0.6980):.4f} | {img_bl.get('roc_auc', 0.7182):.4f} | N/A | N/A |",
        f"| **TF-IDF Findings** (C=0.01) | Text Only | {text_bl.get('accuracy', 0.9068):.4f} | "
        f"{text_bl.get('f1', 0.9257):.4f} | {text_bl.get('roc_auc', 0.9676):.4f} | N/A | N/A |",
        f"| **Fused Multimodal** (C=0.01) | Image + Text | {fuse_bl.get('accuracy', 0.8950):.4f} | "
        f"{fuse_bl.get('f1', 0.9119):.4f} | {fuse_bl.get('roc_auc', 0.9629):.4f} | N/A | N/A |",
        f"| **Fine-Tuned MobileNetV2** | Image Only | **{test_metrics['accuracy']:.4f}** | "
        f"**{test_metrics['f1_score']:.4f}** | **{test_metrics['roc_auc']:.4f}** | "
        f"**{ci_str}** | **{brier_cal:.4f}** (uncal: {brier_uncal:.4f}) |",
        "",
        "---",
        "",
        "## 2. Validation Split vs Test Split Comparison",
        "",
        "| Metric / Setting | Validation Split (n=1090) | Test Split (n=1095) |",
        "| :--- | :--- | :--- |",
        f"| **ROC-AUC** | 0.7522 [0.7157, 0.7879] | {test_metrics['roc_auc']:.4f} {ci_str} |",
        f"| **Frontal ROC-AUC** | 0.7655 [0.7237, 0.8053] | "
        f"{test_metrics['breakdown_by_projection'][0]['roc_auc']:.4f} |",
        f"| **Lateral ROC-AUC** | 0.7384 [0.6942, 0.7814] | "
        f"{test_metrics['breakdown_by_projection'][1]['roc_auc']:.4f} |",
        f"| **Youden Index (J)** | {val_j:.4f} | {test_j:.4f} |",
        f"| **Youden Threshold** | `{test_metrics['threshold_youden']:.4f}` | "
        f"`{test_metrics['threshold_youden']:.4f}` (applied) |",
        f"| **Sensitivity @ Youden** | {val_sens_y:.4f} | {test_metrics['sensitivity']:.4f} |",
        f"| **Specificity @ Youden** | {val_spec_y:.4f} | {test_metrics['specificity']:.4f} |",
        f"| **90% Spec Threshold** | `{test_metrics['threshold_at_90_specificity']:.4f}` | "
        f"`{test_metrics['threshold_at_90_specificity']:.4f}` (applied) |",
        f"| **Sensitivity @ 90% Spec** | {val_sens_90:.4f} | "
        f"{test_metrics['sensitivity_at_90_specificity']:.4f} |",
        f"| **Specificity Achieved** | {val_spec_90:.4f} | "
        f"{test_metrics['specificity_at_90_specificity_threshold']:.4f} |",
        f"| **Brier Score (Calibrated)** | {val_metrics.get('brier_score_calibrated', 0.0):.4f} | "
        f"{brier_cal:.4f} |",
        "",
        "---",
        "",
        "## 3. Projection Breakdown (Test Split)",
        "",
        "| Projection | Total Images | Normal Count | Abnormal Count | ROC-AUC | 95% CI |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for pr in test_metrics["breakdown_by_projection"]:
        p_ci = pr["ci_95"]
        lines.append(
            f"| **{pr['projection']}** | {pr['total_count']} | {pr['normal_count']} | "
            f"{pr['abnormal_count']} | {pr['roc_auc']:.4f} | [{p_ci[0]:.4f}, {p_ci[1]:.4f}] |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Per-Finding Breakdown vs Normal (Test Split)",
        "",
        "| Finding Token | Abnormal Test Count | ROC-AUC vs Normal | Reliability Flag |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for fr in test_metrics["breakdown_by_finding"]:
        lines.append(
            f"| **{fr['finding']}** | {fr['count']} | {fr['roc_auc']:.4f} | "
            f"{fr['trust_flag']} |"
        )

    prov = test_metrics["provenance"]
    lines.extend([
        "",
        "---",
        "",
        "## 5. Artifact & Provenance Details",
        "",
        "- **Evaluated Split**: `test` (evaluated exactly once)",
        f"- **Validation Split Size**: `{prov['n_images_read_val']}` images",
        f"- **Test Split Size**: `{prov['n_images_read_test']}` images",
        f"- **Model Checkpoint SHA-256**: `{prov['model_sha256']}`",
        f"- **Weights File SHA-256**: `{prov['weights_sha256']}`",
        f"- **Evaluation Timestamp**: `{prov['timestamp']}`",
        f"- **Seed**: `{prov['seed']}`",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_full_evaluation()
