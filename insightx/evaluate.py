"""Comprehensive model evaluation, probability calibration, and metric reporting."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def compute_sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.90,
) -> tuple[float, float]:
    """Calculate sensitivity (TPR) when specificity (1 - FPR) is at the target level.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted abnormality probabilities.
        target_specificity: Desired specificity level (default 0.90).

    Returns:
        Tuple of (sensitivity_at_target_spec, operating_threshold).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    target_fpr = 1.0 - target_specificity

    # Find closest threshold with FPR <= target_fpr
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        idx = 0
    else:
        idx = valid_indices[-1]

    # Linear interpolation for precise sensitivity estimate
    sensitivity = float(np.interp(target_fpr, fpr, tpr))
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return sensitivity, thresh


def fit_platt_scaling(
    y_val: np.ndarray,
    val_probs: np.ndarray,
) -> tuple[LogisticRegression, dict[str, Any]]:
    """Fit Platt scaling calibrator on the validation split only.

    Args:
        y_val: Validation split binary ground truth.
        val_probs: Uncalibrated probabilities from the validation split.

    Returns:
        Tuple of (fitted_logistic_model, calibration_parameters_dict).
    """
    # Clip probabilities to avoid infinite logits
    eps = 1e-6
    clipped_probs = np.clip(val_probs, eps, 1.0 - eps)
    # Log-odds (logits) representation
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


def plot_evaluation_charts(
    y_test: np.ndarray,
    test_probs_uncal: np.ndarray,
    test_probs_cal: np.ndarray,
    cm: np.ndarray,
    reports_dir: Path,
) -> None:
    """Generate ROC curve, Confusion Matrix, and Reliability Diagram plots."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, test_probs_uncal)
    auc_val = roc_auc_score(y_test, test_probs_uncal)
    sens_90, _ = compute_sensitivity_at_specificity(y_test, test_probs_uncal, 0.90)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#00838F", lw=2.5, label=f"ROC Curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    # Point at 90% specificity (FPR = 0.10)
    plt.scatter(
        [0.10], [sens_90], color="#D32F2F", s=80, zorder=5,
        label=f"Sens @ 90% Spec: {sens_90:.3f}",
    )
    plt.title("InsightX ? Receiver Operating Characteristic (ROC)", fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=150)
    plt.close()

    # 2. Confusion Matrix
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
    plt.title("InsightX ? Confusion Matrix (Test Split)", fontweight="bold")
    plt.xlabel("Predicted Finding")
    plt.ylabel("Ground Truth Annotation")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # 3. Reliability Diagram (Calibration Curve Before & After)
    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_test, test_probs_uncal, n_bins=10
    )
    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_test, test_probs_cal, n_bins=10
    )

    brier_before = brier_score_loss(y_test, test_probs_uncal)
    brier_after = brier_score_loss(y_test, test_probs_cal)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(
        mean_pred_uncal,
        fraction_pos_uncal,
        "s--",
        color="#E65100",
        label=f"Uncalibrated (Brier = {brier_before:.4f})",
    )
    plt.plot(
        mean_pred_cal,
        fraction_pos_cal,
        "o-",
        color="#2E7D32",
        label=f"Platt Calibrated (Brier = {brier_after:.4f})",
    )
    plt.title("Reliability Diagram (Validation-Fitted Calibration)", fontweight="bold")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(reports_dir / "calibration_curve.png", dpi=150)
    plt.close()


def run_full_evaluation(
    model_path: Path | str = "models/mobilenet_finetuned.keras",
    splits_path: Path | str = "data/processed/splits.csv",
    reports_dir: Path | str = "reports",
    calib_json_path: Path | str = "models/calibration.json",
) -> dict[str, Any]:
    """Execute evaluation on test split, calibrate on val split, and write reports."""
    rep_dir = Path(reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(splits_path)

    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    y_val = df.loc[val_mask, "label"].to_numpy()
    y_test = df.loc[test_mask, "label"].to_numpy()

    # Load model
    m_path = Path(model_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Model file not found at {m_path}")

    model = tf.keras.models.load_model(str(m_path))

    # Compute features/probabilities for val and test
    img_dir = Path("data/raw/images_normalized")
    val_files = [img_dir / str(fn).strip() for fn in df.loc[val_mask, "filename"]]
    test_files = [img_dir / str(fn).strip() for fn in df.loc[test_mask, "filename"]]

    val_found = sum(1 for f in val_files if f.is_file())
    test_found = sum(1 for f in test_files if f.is_file())

    if val_found < len(val_files) or test_found < len(test_files):
        total_found = val_found + test_found
        total_expected = len(val_files) + len(test_files)
        raise FileNotFoundError(
            f"Expected image directory '{img_dir}' containing {total_expected} images for splits, "
            f"but found {total_found} valid files on disk (val: {val_found}/{len(val_files)}, "
            f"test: {test_found}/{len(test_files)}). Refusing to write metrics.json."
        )

    from insightx.train import build_dataset_pipeline
    val_ds = build_dataset_pipeline(df[val_mask], img_dir, is_training=False)
    test_ds = build_dataset_pipeline(df[test_mask], img_dir, is_training=False)
    val_probs = model.predict(val_ds).flatten()
    test_probs_uncal = model.predict(test_ds).flatten()

    # 1. Fit Platt Scaling on Validation Split ONLY
    calibrator, calib_params = fit_platt_scaling(y_val, val_probs)
    test_probs_cal = apply_platt_calibration(calibrator, test_probs_uncal)

    # Save calibration parameters to models/calibration.json
    calib_out = Path(calib_json_path)
    calib_out.parent.mkdir(parents=True, exist_ok=True)
    calib_params["brier_test_before"] = float(brier_score_loss(y_test, test_probs_uncal))
    calib_params["brier_test_after"] = float(brier_score_loss(y_test, test_probs_cal))
    with open(calib_out, "w", encoding="utf-8") as f:
        json.dump(calib_params, f, indent=2)
    print(f"[OK] Calibration parameters saved to: {calib_out}")

    # 2. Test Set Metrics
    test_preds = (test_probs_cal >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, test_preds))
    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    auc = float(roc_auc_score(y_test, test_probs_uncal))
    sens_90, thresh_90 = compute_sensitivity_at_specificity(
        y_test, test_probs_uncal, 0.90
    )
    cm = confusion_matrix(y_test, test_preds)

    metrics_payload = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "evaluation_split": "test",
        "sample_count": len(y_test),
        "provenance": {
            "n_images_read_val": val_found,
            "n_images_read_test": test_found,
            "timestamp": datetime.now(UTC).isoformat(),
            "model_path": str(m_path),
            "images_dir": str(img_dir),
        },
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "sensitivity_at_90_specificity": round(sens_90, 4),
        "threshold_at_90_specificity": round(thresh_90, 4),
        "brier_score_uncalibrated": round(float(brier_score_loss(y_test, test_probs_uncal)), 4),
        "brier_score_calibrated": round(float(brier_score_loss(y_test, test_probs_cal)), 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["Normal (0)", "Abnormal (1)"],
    }

    metrics_path = rep_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[OK] Test metrics written to: {metrics_path}")

    # 3. Save Charts
    plot_evaluation_charts(
        y_test, test_probs_uncal, test_probs_cal, cm, rep_dir
    )
    print(f"[OK] ROC curve saved to: {rep_dir / 'roc_curve.png'}")
    print(f"[OK] Confusion matrix saved to: {rep_dir / 'confusion_matrix.png'}")
    print(f"[OK] Reliability diagram saved to: {rep_dir / 'calibration_curve.png'}")

    return metrics_payload


if __name__ == "__main__":
    run_full_evaluation()
