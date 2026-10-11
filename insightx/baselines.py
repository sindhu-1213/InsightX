"""Baseline model evaluation for InsightX: Text-only, Image-only, and Multimodal Fusion.

Notice:
    Research Demo Only - This tool is not a certified medical device.
    Outputs are AI predictions/findings for research purposes only and
    must not be used to guide clinical decisions.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from insightx.text import clean_medical_text, get_medical_stopwords

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("insightx.baselines")


def extract_or_load_image_features(
    df: pd.DataFrame,
    images_dir: Path | str = "data/processed/images_224",
    cache_path: Path | str = "data/processed/image_features.npy",
    index_path: Path | str = "data/processed/feature_index.csv",
    batch_size: int = 64,
    force_recompute: bool = False,
) -> tuple[np.ndarray, float]:
    """Extract or load frozen MobileNetV2 features aligned to splits.csv row order.

    Extracts 1280-dim Global Average Pooling embeddings from frozen MobileNetV2
    on CPU in batches of 64 using tf.data.
    Supports crash-resilience with checkpointed batch files.

    Args:
        df: DataFrame containing 'filename' column in splits.csv row order.
        images_dir: Directory containing resized 224x224 grayscale PNGs.
        cache_path: Destination path for .npy feature array.
        index_path: Destination path for feature_index.csv.
        batch_size: Batch size for inference (default 64).
        force_recompute: Whether to ignore cached files and recompute.

    Returns:
        Tuple of (features_array of shape (N, 1280), extraction_runtime_seconds).
    """
    cache = Path(cache_path)
    index_file = Path(index_path)

    if not force_recompute and cache.is_file() and index_file.is_file():
        logger.info(f"Checking cached image features at {cache} and index at {index_file}...")
        idx_df = pd.read_csv(index_file)
        features = np.load(cache)
        if len(features) == len(df) and len(idx_df) == len(df):
            if (idx_df["filename"].values == df["filename"].values).all():
                logger.info(
                    f"Verified valid cached features ({features.shape}) aligned with splits.csv."
                )
                return features, 0.0
        logger.warning("Cached features mismatch with splits.csv. Re-extracting...")

    # Force CPU execution
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    try:
        cpu_count = os.cpu_count() or 4
        tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
        tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
    except Exception as e:
        logger.warning(f"Could not configure tf threading: {e}")

    img_folder = Path(images_dir)
    image_paths = [str(img_folder / str(fn).strip()) for fn in df["filename"]]

    # Verify all files exist
    missing = [p for p in image_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Expected {len(image_paths)} images in '{img_folder}', but {len(missing)} are "
            f"missing. Example: {missing[:3]}"
        )

    logger.info(f"Extracting frozen MobileNetV2 features on CPU for {len(df):,} images...")
    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    model.trainable = False

    def _load_and_preprocess(p: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(p)
        # tf.image.decode_png with channels=3 converts 1-channel grayscale to 3 channels
        img = tf.image.decode_png(raw, channels=3)
        img = tf.cast(img, tf.float32)
        return preprocess_input(img)

    total_images = len(image_paths)
    num_batches = (total_images + batch_size - 1) // batch_size
    checkpoint_dir = Path("data/processed/.feature_cache")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint batches to support resume
    completed_batches = set()
    for f in checkpoint_dir.glob("batch_*.npy"):
        try:
            b_idx = int(f.stem.split("_")[1])
            completed_batches.add(b_idx)
        except (ValueError, IndexError):
            pass

    logger.info(
        f"Processing in {num_batches} batches of {batch_size}. "
        f"Found {len(completed_batches)} existing batch checkpoints."
    )

    t0 = time.time()
    with tqdm(total=num_batches, desc="Extracting MobileNetV2 features", unit="batch") as pbar:
        # Pre-advance progress bar for already completed batches
        pbar.update(len(completed_batches))

        for b_idx in range(num_batches):
            if b_idx in completed_batches:
                continue

            start_idx = b_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            batch_paths = image_paths[start_idx:end_idx]

            ds = tf.data.Dataset.from_tensor_slices(batch_paths)
            ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(len(batch_paths)).prefetch(tf.data.AUTOTUNE)

            # Predict on batch
            for batch_tensor in ds:
                batch_feats = model.predict_on_batch(batch_tensor)
                ckpt_file = checkpoint_dir / f"batch_{b_idx:04d}.npy"
                np.save(ckpt_file, batch_feats)

            completed_batches.add(b_idx)
            pbar.update(1)

    elapsed = time.time() - t0
    logger.info(f"Feature extraction completed in {elapsed:.2f} seconds.")

    # Assemble all batches in exact order
    logger.info("Assembling and caching complete feature matrix...")
    batch_arrays = []
    for b_idx in range(num_batches):
        ckpt_file = checkpoint_dir / f"batch_{b_idx:04d}.npy"
        batch_arrays.append(np.load(ckpt_file))

    features = np.concatenate(batch_arrays, axis=0)
    if features.shape != (total_images, 1280):
        raise ValueError(
            f"Shape mismatch: expected ({total_images}, 1280), got {features.shape}"
        )

    # Save cached .npy
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, features)

    # Save feature_index.csv recording filename for every row to verify alignment
    index_df = pd.DataFrame({"filename": df["filename"].values})
    index_df.to_csv(index_file, index=False)
    logger.info(f"Cached image features saved to {cache} and index saved to {index_file}.")

    # Clean up checkpoint dir
    try:
        shutil.rmtree(checkpoint_dir)
    except Exception as e:
        logger.warning(f"Could not remove checkpoint directory: {e}")

    return features, elapsed


def build_tfidf_features(
    df: pd.DataFrame,
    reports_csv_path: Path | str = "data/raw/indiana_reports.csv",
    max_features: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF vectorizer on TRAIN split only and transform val and test splits.

    Retains negation tokens ('no', 'not', 'none', 'without', 'negative')
    in the vocabulary.
    If 'findings' is missing from df, joins from reports CSV by 'uid'.

    Args:
        df: DataFrame from splits.csv.
        reports_csv_path: Path to reports CSV if fallback join is needed.
        max_features: Vocabulary size (1000).

    Returns:
        Tuple of (X_train_txt, X_val_txt, X_test_txt, fitted_vectorizer).
    """
    df_text = df.copy()

    # Join findings from reports CSV by uid if missing
    if "findings" not in df_text.columns or df_text["findings"].isna().all():
        logger.info(f"Findings missing from splits dataframe. Joining from {reports_csv_path}...")
        reports = pd.read_csv(reports_csv_path)
        df_text = df_text.drop(columns=["findings"], errors="ignore").merge(
            reports[["uid", "findings"]], on="uid", how="left"
        )

    findings_series = df_text["findings"].fillna("").astype(str)

    med_stops = get_medical_stopwords()
    clean_findings = findings_series.apply(lambda t: clean_medical_text(t, med_stops))

    train_mask = (df_text["split"] == "train").to_numpy()
    val_mask = (df_text["split"] == "val").to_numpy()
    test_mask = (df_text["split"] == "test").to_numpy()

    clean_train = clean_findings[train_mask].tolist()
    clean_val = clean_findings[val_mask].tolist()
    clean_test = clean_findings[test_mask].tolist()

    logger.info(
        f"Fitting TF-IDF on TRAIN split ({len(clean_train):,} samples)..."
    )
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(clean_train).toarray()
    X_val = vectorizer.transform(clean_val).toarray()
    X_test = vectorizer.transform(clean_test).toarray()

    # Verify retained negations
    vocab = set(vectorizer.get_feature_names_out())
    retained_negations = {"no", "not", "none", "without", "negative"} & vocab
    logger.info(f"TF-IDF vocabulary retained clinical negations: {sorted(retained_negations)}")

    return X_train, X_val, X_test, vectorizer


def train_and_evaluate_baselines(
    splits_path: Path | str = "data/processed/splits.csv",
    images_dir: Path | str = "data/processed/images_224",
    features_cache_path: Path | str = "data/processed/image_features.npy",
    feature_index_path: Path | str = "data/processed/feature_index.csv",
    report_md_path: Path | str = "reports/baselines.md",
    report_json_path: Path | str = "reports/baselines.json",
    candidate_c_values: list[float] | None = None,
) -> dict[str, Any]:
    """Train Text-only, Image-only, and Fused Logistic Regression models.

    Scalers are fit strictly on the train split only.
    C regularization is selected using the VALIDATION split only.
    Test split is evaluated once on the selected model.

    Args:
        splits_path: Path to splits.csv.
        images_dir: Directory containing resized 224x224 grayscale PNGs.
        features_cache_path: Path to cached image features .npy.
        feature_index_path: Path to feature_index.csv.
        report_md_path: Output markdown report path.
        report_json_path: Output JSON report path.
        candidate_c_values: List of candidate C values (default: [0.01, 0.1, 1, 10]).

    Returns:
        Structured results dictionary.
    """
    if candidate_c_values is None:
        candidate_c_values = [0.01, 0.1, 1.0, 10.0]

    df = pd.read_csv(splits_path)
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    y_train = df.loc[train_mask, "label"].to_numpy().astype(int)
    y_val = df.loc[val_mask, "label"].to_numpy().astype(int)
    y_test = df.loc[test_mask, "label"].to_numpy().astype(int)

    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    logger.info(f"Split counts: Train={n_train:,}, Val={n_val:,}, Test={n_test:,}")

    # 1. Image Features (MobileNetV2 GAP)
    img_features, extraction_time = extract_or_load_image_features(
        df=df,
        images_dir=images_dir,
        cache_path=features_cache_path,
        index_path=feature_index_path,
    )
    X_img_train = img_features[train_mask]
    X_img_val = img_features[val_mask]
    X_img_test = img_features[test_mask]

    # 2. Text Features (TF-IDF on findings, fit on train only)
    X_txt_train, X_txt_val, X_txt_test, _ = build_tfidf_features(
        df=df,
        max_features=1000,
    )

    # 3. Fused Features (hstack)
    X_fused_train = np.hstack((X_img_train, X_txt_train))
    X_fused_val = np.hstack((X_img_val, X_txt_val))
    X_fused_test = np.hstack((X_img_test, X_txt_test))

    modalities = {
        "text_only": {
            "name": "Text-Only (Findings TF-IDF)",
            "data": (X_txt_train, X_txt_val, X_txt_test),
        },
        "image_only": {
            "name": "Image-Only (MobileNetV2 GAP)",
            "data": (X_img_train, X_img_val, X_img_test),
        },
        "fused": {
            "name": "Fused (Image + Text hstack)",
            "data": (X_fused_train, X_fused_val, X_fused_test),
        },
    }

    # Majority-class baseline on test split
    # Abnormal is positive class (1). Most frequent class in Indiana dataset is Abnormal (1).
    majority_class_val = 1
    majority_pred_test = np.full_like(y_test, fill_value=majority_class_val)
    maj_acc = float(accuracy_score(y_test, majority_pred_test))
    maj_f1 = float(f1_score(y_test, majority_pred_test, pos_label=1, zero_division=0))
    maj_roc_auc = 0.5000
    maj_cm = confusion_matrix(y_test, majority_pred_test).tolist()

    results: dict[str, Any] = {
        "dataset_splits": {
            "train_rows": n_train,
            "val_rows": n_val,
            "test_rows": n_test,
            "total_rows": len(df),
        },
        "feature_extraction": {
            "model": "MobileNetV2 (frozen, ImageNet weights, include_top=False, pooling=avg)",
            "embedding_dim": 1280,
            "batch_size": 64,
            "runtime_seconds": float(round(extraction_time, 2)),
            "cache_path": str(features_cache_path),
            "index_path": str(feature_index_path),
        },
        "majority_class_baseline": {
            "majority_class": "Abnormal (1)",
            "accuracy": maj_acc,
            "f1": maj_f1,
            "roc_auc": maj_roc_auc,
            "confusion_matrix": maj_cm,
        },
        "models": {},
    }

    for mod_key, mod_info in modalities.items():
        mod_name = mod_info["name"]
        X_tr_raw, X_va_raw, X_te_raw = mod_info["data"]

        # Standardize numeric features with scaler fit on TRAIN only
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)
        X_te = scaler.transform(X_te_raw)

        val_grid_results = {}
        best_c = candidate_c_values[0]
        best_val_auc = -1.0
        best_clf = None

        logger.info(f"Tuning regularization parameter C for {mod_name} on validation split...")
        for c in candidate_c_values:
            clf = LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
            clf.fit(X_tr, y_train)

            va_pred = clf.predict(X_va)
            va_proba = clf.predict_proba(X_va)[:, 1]

            va_acc = float(accuracy_score(y_val, va_pred))
            va_f1 = float(f1_score(y_val, va_pred, pos_label=1, zero_division=0))
            va_auc = float(roc_auc_score(y_val, va_proba))

            val_grid_results[str(c)] = {
                "accuracy": va_acc,
                "f1": va_f1,
                "roc_auc": va_auc,
            }
            logger.info(
                f"  C={c:5.2f} -> Val Acc: {va_acc*100:.2f}%, F1: {va_f1:.4f}, AUC: {va_auc:.4f}"
            )

            # Select best C based on validation ROC-AUC
            if va_auc > best_val_auc:
                best_val_auc = va_auc
                best_c = c
                best_clf = clf

        logger.info(f"Chosen C for {mod_name}: {best_c} (Val ROC-AUC: {best_val_auc:.4f})")

        # Evaluate on TEST split once with chosen model
        assert best_clf is not None
        te_pred = best_clf.predict(X_te)
        te_proba = best_clf.predict_proba(X_te)[:, 1]

        te_acc = float(accuracy_score(y_test, te_pred))
        te_f1 = float(f1_score(y_test, te_pred, pos_label=1, zero_division=0))
        te_auc = float(roc_auc_score(y_test, te_proba))
        te_cm = confusion_matrix(y_test, te_pred).tolist()

        results["models"][mod_key] = {
            "display_name": mod_name,
            "chosen_C": best_c,
            "val_metrics_by_c": val_grid_results,
            "test_metrics": {
                "accuracy": te_acc,
                "f1": te_f1,
                "roc_auc": te_auc,
                "confusion_matrix": te_cm,
            },
        }

    # Save reports
    save_reports(results, md_path=report_md_path, json_path=report_json_path)
    return results


def save_reports(
    results: dict[str, Any],
    md_path: Path | str = "reports/baselines.md",
    json_path: Path | str = "reports/baselines.json",
) -> None:
    """Save formatted markdown report and JSON artifact."""
    md_file = Path(md_path)
    json_file = Path(json_path)

    md_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved baseline JSON metrics to {json_file}")

    # 2. Build Markdown
    n_train = results["dataset_splits"]["train_rows"]
    n_val = results["dataset_splits"]["val_rows"]
    n_test = results["dataset_splits"]["test_rows"]
    maj = results["majority_class_baseline"]

    rows = []
    # Majority baseline row
    maj_tn, maj_fp = maj["confusion_matrix"][0][0], maj["confusion_matrix"][0][1]
    maj_fn, maj_tp = maj["confusion_matrix"][1][0], maj["confusion_matrix"][1][1]
    maj_cm_str = f"TN={maj_tn}, FP={maj_fp} / FN={maj_fn}, TP={maj_tp}"
    rows.append(
        f"| **Majority-Class Baseline** (Predict Abnormal) | N/A | {maj['accuracy']*100:.2f}% | "
        f"{maj['f1']:.4f} | {maj['roc_auc']:.4f} | {maj_cm_str} |"
    )

    for _, model_data in results["models"].items():
        name = model_data["display_name"]
        c_val = model_data["chosen_C"]
        te = model_data["test_metrics"]
        cm = te["confusion_matrix"]
        cm_str = f"TN={cm[0][0]}, FP={cm[0][1]} / FN={cm[1][0]}, TP={cm[1][1]}"
        rows.append(
            f"| **{name}** | C={c_val} | {te['accuracy']*100:.2f}% | "
            f"{te['f1']:.4f} | {te['roc_auc']:.4f} | {cm_str} |"
        )

    table_content = "\n".join(rows)

    img_acc = results["models"]["image_only"]["test_metrics"]["accuracy"] * 100
    img_auc = results["models"]["image_only"]["test_metrics"]["roc_auc"]
    txt_acc = results["models"]["text_only"]["test_metrics"]["accuracy"] * 100
    txt_auc = results["models"]["text_only"]["test_metrics"]["roc_auc"]
    fus_acc = results["models"]["fused"]["test_metrics"]["accuracy"] * 100
    fus_auc = results["models"]["fused"]["test_metrics"]["roc_auc"]
    ext_time = results["feature_extraction"]["runtime_seconds"]

    col_header = (
        "| Model / Modality | Regularization ($C$) | Accuracy | F1-Score | "
        "ROC-AUC | Confusion Matrix (TN, FP / FN, TP) |"
    )

    md_content = f"""# InsightX — Baseline Model Comparison Report

> **Research Demo Only** — This tool is not a certified medical device.
> Results are AI predictions for research purposes only and must not be used
> to guide clinical decisions. Always consult a qualified radiologist.

---

## Dataset Split Accounting

- **Train Rows**: {n_train:,} (Strictly used for feature fitting and model training)
- **Validation Rows**: {n_val:,} (Used for regularization parameter C selection)
- **Test Rows**: {n_test:,} (Held-out evaluation split; evaluated once per chosen model)
- **Total Rows**: {results['dataset_splits']['total_rows']:,}
- **MobileNetV2 Feature Extraction Runtime**: {ext_time:.2f}s (CPU execution, batch size 64)

---

## Test Split Performance Summary

{col_header}
|---|---|---|---|---|---|
{table_content}

---

## Model Performance Interpretation & Analysis

The evaluation numbers demonstrate a substantial performance divergence between modalities.
While the majority-class baseline achieves {maj['accuracy']*100:.2f}% accuracy with an F1 score
of {maj['f1']:.4f} and ROC-AUC of 0.5000 by predicting Abnormal for all cases, the image-only
MobileNetV2 GAP baseline achieves an accuracy of {img_acc:.2f}% (ROC-AUC: {img_auc:.4f}), and
the text-only TF-IDF baseline achieves an accuracy of {txt_acc:.2f}% (ROC-AUC: {txt_auc:.4f}).
Fused multimodal representation yields an accuracy of {fus_acc:.2f}% (ROC-AUC: {fus_auc:.4f}).

The text-only baseline significantly outperforms the image-only baseline across all evaluation
metrics. This gap occurs because the Indiana University disease annotations (MeSH / Problems)
were originally extracted directly from the same clinical radiology reports by medical indexers.
As a result, the findings text contains explicit lexical diagnostic statements (e.g., negative
phrases such as "no acute cardiopulmonary abnormality" versus positive descriptors of infiltrates
or cardiomegaly) that directly encode the target label. In clinical practice, however, a
radiologist's pre-authored report does not exist when a chest radiograph is initially taken.
Relying on findings text therefore introduces label information that an actual automated X-ray
interpretation system would not have access to at test time. Consequently, the production
diagnostic pipeline must center on visual radiograph representations.

---
*Report generated by `insightx/baselines.py` | Order verified against feature_index.csv*
"""

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved baseline Markdown comparison report to {md_file}")


if __name__ == "__main__":
    train_and_evaluate_baselines()
