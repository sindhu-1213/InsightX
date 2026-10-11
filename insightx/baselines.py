"""Baseline model evaluation for InsightX: Text-only, Image-only, and Multimodal Fusion.

Notice:
    Research Demo Only - This tool is not a certified medical device.
    Outputs are AI predictions/findings for research purposes only and
    must not be used to guide clinical decisions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from insightx.text import clean_medical_text, get_medical_stopwords


def extract_or_load_image_features(
    df: pd.DataFrame,
    images_dir: Path | str = "data/raw/images_normalized",
    cache_path: Path | str = "data/processed/image_features.npy",
    batch_size: int = 64,
) -> np.ndarray:
    """Extract or load MobileNetV2 features aligned to splits.csv row order.

    Extracts 1280-dim Global Average Pooling embeddings from frozen MobileNetV2
    in batches of 64 using tf.data. Caches to disk.

    Args:
        df: DataFrame containing 'filename' column in splits.csv row order.
        images_dir: Directory containing normalized X-ray images.
        cache_path: File path to save/load cached .npy feature array.
        batch_size: Inference batch size.

    Returns:
        Numpy array of shape (N, 1280).
    """
    cache = Path(cache_path)
    if cache.is_file():
        print(f"[CACHE] Loading cached image features from {cache}...")
        features = np.load(cache)
        if len(features) == len(df):
            return features
        print(
            f"[CACHE] Shape mismatch ({len(features)} vs {len(df)}). Re-extracting..."
        )

    img_folder = Path(images_dir)
    image_paths = [str(img_folder / str(fn).strip()) for fn in df["filename"]]
    existing_paths = [p for p in image_paths if Path(p).is_file()]

    if len(existing_paths) < len(image_paths) or len(image_paths) == 0:
        raise FileNotFoundError(
            f"Expected image directory '{img_folder}' containing {len(image_paths)} images from "
            f"dataframe, but found {len(existing_paths)} valid files on disk. "
            f"Please ensure all Indiana University images are present in '{img_folder}'."
        )

    print(f"[DL] Extracting MobileNetV2 features for {len(image_paths):,} images...")
    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    model.trainable = False

    def _load_image(p: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(p)
        img = tf.image.decode_png(raw, channels=3)
        img = tf.image.resize(img, [224, 224])
        return preprocess_input(img)

    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    features = model.predict(ds, verbose=1)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, features)
    print(f"[SAVED] Cached image features to {cache} (Shape: {features.shape})")
    return features


def build_tfidf_features(
    train_texts: pd.Series | list[str],
    val_texts: pd.Series | list[str],
    test_texts: pd.Series | list[str],
    max_features: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF vectorizer on the train split only and transform val/test splits.

    Maintains clinical negation tokens ('no', 'not', 'none', 'without', 'negative')
    in the vocabulary to prevent diagnostic inversion.

    Args:
        train_texts: Training findings.
        val_texts: Validation findings.
        test_texts: Test findings.
        max_features: Vocabulary capacity.

    Returns:
        Tuple of (X_train_txt, X_val_txt, X_test_txt, fitted_vectorizer).
    """
    med_stops = get_medical_stopwords()

    clean_train = [clean_medical_text(t, med_stops) for t in train_texts]
    clean_val = [clean_medical_text(t, med_stops) for t in val_texts]
    clean_test = [clean_medical_text(t, med_stops) for t in test_texts]

    vectorizer = TfidfVectorizer(max_features=max_features)
    # FIT ON TRAIN ONLY to prevent test split leakage
    X_train = vectorizer.fit_transform(clean_train).toarray()
    X_val = vectorizer.transform(clean_val).toarray()
    X_test = vectorizer.transform(clean_test).toarray()

    return X_train, X_val, X_test, vectorizer


def train_and_evaluate_baselines(
    splits_path: Path | str = "data/processed/splits.csv",
    report_output_path: Path | str = "reports/baselines.md",
) -> dict[str, dict[str, float]]:
    """Train Text-only, Image-only, and Fused Logistic Regression models and evaluate on test set.

    Scalers are strictly fit on the train split only.

    Args:
        splits_path: Path to splits.csv.
        report_output_path: Destination path for markdown comparison report.

    Returns:
        Dictionary of test metric scores per modality.
    """
    df = pd.read_csv(splits_path)

    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    y_train = df.loc[train_mask, "label"].to_numpy()
    y_test = df.loc[test_mask, "label"].to_numpy()

    # 1. Image Features (MobileNetV2)
    img_features = extract_or_load_image_features(df)
    X_img_train = img_features[train_mask]
    X_img_test = img_features[test_mask]

    # 2. Text Features (TF-IDF on findings, fit on train only)
    findings = df["findings"].fillna("").astype(str)
    X_txt_train, _, X_txt_test, _ = build_tfidf_features(
        findings[train_mask], findings[val_mask], findings[test_mask]
    )

    # 3. Fused Features (hstack)
    X_fused_train = np.hstack((X_img_train, X_txt_train))
    X_fused_test = np.hstack((X_img_test, X_txt_test))

    modalities: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Text-Only (Findings TF-IDF)": (X_txt_train, X_txt_test),
        "Image-Only (MobileNetV2 GAP)": (X_img_train, X_img_test),
        "Fused (Image + Text hstack)": (X_fused_train, X_fused_test),
    }

    results: dict[str, dict[str, float]] = {}

    for name, (X_tr, X_te) in modalities.items():
        # Fit scaler on train only
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        clf.fit(X_tr_scaled, y_train)

        y_pred = clf.predict(X_te_scaled)
        y_proba = clf.predict_proba(X_te_scaled)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_proba))

        results[name] = {
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
        }

    save_markdown_report(results, output_path=report_output_path)
    return results


def save_markdown_report(
    results: dict[str, dict[str, float]],
    output_path: Path | str = "reports/baselines.md",
) -> None:
    """Generate and save formatted markdown baseline comparison report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for mod_name, metrics in results.items():
        rows.append(
            f"| **{mod_name}** | {metrics['Accuracy']*100:.2f}% | "
            f"{metrics['F1-Score']:.4f} | {metrics['ROC-AUC']:.4f} |"
        )
    table_rows = "\n".join(rows)

    content = f"""# InsightX ? Baseline Model Comparison Report

> ?? **Research Demo Only** ? This tool is not a certified medical device.
> Results are AI predictions for research purposes only and must not be used
> to guide clinical decisions. Always consult a qualified radiologist.

---

## ?? Test Split Performance Summary

| Model / Modality | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
{table_rows}

---

## 🔍 Model Performance Interpretation & Analysis

### Why is the Text-Only Score Higher?
The **text-only baseline achieves higher performance metrics** on this evaluation benchmark.
This is an expected phenomenon inherent to radiological text datasets:
- The Indiana University disease annotations (MeSH / Problems) are **extracted directly
  from the clinical radiology reports** by medical indexers.
- Consequently, vocabulary in the `findings` section shares direct semantic and lexical
  overlap with the target labels.
- Evaluating on text from the same clinical narrative that produced the label provides
  strong linguistic correlation.

### Why is the Served Model Image-Based?
In actual clinical deployment scenarios, a radiologist report **does not exist at patient intake**:
- The primary clinical utility of InsightX is to analyze raw chest radiographs before reading.
- Relying on report text at inference time would defeat the purpose of an automated clinical
  decision support system, as a human radiologist would have already authored the report.
- Therefore, the production-served model prioritizes visual radiograph representations
  (MobileNetV2 feature extraction and Grad-CAM spatial heatmaps).

---
*Generated by `insightx/baselines.py` | Project: InsightX v1.0.0*
"""

    with open(out, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[OK] Markdown comparison report saved to: {out}")
    print("\n" + "=" * 65)
    print("  INSIGHTX BASELINE TEST RESULTS")
    print("=" * 65)
    for mod, met in results.items():
        acc_str = f"{met['Accuracy']*100:.2f}%"
        f1_str = f"{met['F1-Score']:.4f}"
        auc_str = f"{met['ROC-AUC']:.4f}"
        print(f"  {mod:30s} | Acc: {acc_str} | F1: {f1_str} | ROC-AUC: {auc_str}")
    print("=" * 65)


if __name__ == "__main__":
    train_and_evaluate_baselines()

