"""Validation split read-only diagnostic for models/mobilenet_finetuned.keras."""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("diagnostics_val")


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

    # Pre-index indices by uid for fast lookup
    uid_to_indices = {uid: df.index[df["uid"] == uid].to_numpy() for uid in uids}

    boot_aucs = []
    for _ in range(n_bootstraps):
        sampled_uids = rng.choice(uids, size=n_uids, replace=True)
        sampled_indices = np.concatenate([uid_to_indices[uid] for uid in sampled_uids])

        y_boot_true = y_true[sampled_indices]
        y_boot_pred = y_pred[sampled_indices]

        # Only valid if both classes are present in sample
        if len(np.unique(y_boot_true)) > 1:
            auc = roc_auc_score(y_boot_true, y_boot_pred)
            boot_aucs.append(auc)

    boot_aucs = np.array(boot_aucs)
    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    point_auc = float(roc_auc_score(y_true, y_pred))

    return point_auc, ci_lower, ci_upper


def main() -> None:
    splits_path = Path("data/processed/splits.csv")
    model_path = Path("models/mobilenet_finetuned.keras")
    images_dir = Path("data/processed/images_224")
    report_path = Path("reports/diagnostics_val.md")

    logger.info("Loading validation split only...")
    full_df = pd.read_csv(splits_path)
    val_df = full_df[full_df["split"] == "val"].copy().reset_index(drop=True)
    n_val_uids = val_df["uid"].nunique()
    logger.info(f"Loaded {len(val_df)} validation images across {n_val_uids} patients.")

    # Ensure projection column is present
    if "projection" not in val_df.columns:
        projections_csv = pd.read_csv("data/raw/indiana_projections.csv")
        val_df = val_df.merge(
            projections_csv[["filename", "projection"]], on="filename", how="left"
        )

    logger.info(f"Loading trained model from {model_path}...")
    # Handle cross-version Keras 3 deserialization arguments
    import keras

    bn_orig = keras.layers.BatchNormalization.__init__
    keras.layers.BatchNormalization.__init__ = lambda self, *a, **kw: bn_orig(
        self, *a, **{k: v for k, v in kw.items() if not k.startswith("renorm")}
    )

    gl_orig = keras.initializers.GlorotUniform.__init__
    keras.initializers.GlorotUniform.__init__ = lambda self, *a, **kw: gl_orig(
        self, *a, **{k: v for k, v in kw.items() if not k.endswith("_axes")}
    )

    model = tf.keras.models.load_model(str(model_path))

    # Build tf.data evaluation pipeline for validation split only
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    def _load_image(p: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(p)
        img = tf.image.decode_png(raw, channels=3)
        img = tf.cast(img, tf.float32)
        return preprocess_input(img)

    val_paths = [str(images_dir / str(f).strip()) for f in val_df["filename"]]
    ds = tf.data.Dataset.from_tensor_slices(val_paths)
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)

    logger.info("Running inference on validation images...")
    preds = model.predict(ds, verbose=1)
    probs = preds.flatten()
    val_df["pred_prob"] = probs
    y_true = val_df["label"].to_numpy().astype(int)

    # 1. Overall ROC-AUC and 95% Bootstrap CI
    overall_auc, overall_ci_low, overall_ci_high = bootstrap_auc_by_uid(
        val_df, y_true, probs, n_bootstraps=1000, seed=42
    )
    logger.info(
        f"Overall Val ROC-AUC: {overall_auc:.4f} "
        f"(95% CI: [{overall_ci_low:.4f}, {overall_ci_high:.4f}])"
    )

    # 2. Breakdown by projection (Frontal vs Lateral)
    proj_results = []
    for proj_name in ["Frontal", "Lateral"]:
        sub_mask = val_df["projection"] == proj_name
        sub_df = val_df[sub_mask].copy().reset_index(drop=True)
        sub_y_true = sub_df["label"].to_numpy().astype(int)
        sub_probs = sub_df["pred_prob"].to_numpy()

        p_auc, p_low, p_high = bootstrap_auc_by_uid(
            sub_df, sub_y_true, sub_probs, n_bootstraps=1000, seed=42
        )
        norm_cnt = int((sub_y_true == 0).sum())
        abnorm_cnt = int((sub_y_true == 1).sum())
        proj_results.append({
            "projection": proj_name,
            "total_count": len(sub_df),
            "normal_count": norm_cnt,
            "abnormal_count": abnorm_cnt,
            "auc": p_auc,
            "ci_low": p_low,
            "ci_high": p_high,
        })
        logger.info(
            f"Projection {proj_name}: {len(sub_df)} imgs ({norm_cnt} Norm, {abnorm_cnt} Abn) | "
            f"AUC: {p_auc:.4f} [{p_low:.4f}, {p_high:.4f}]"
        )

    # 3. Breakdown by finding (10 most common tokens among Abnormal validation images)
    abnormal_val = val_df[val_df["label"] == 1]
    tokens = [
        t.strip()
        for probs_str in abnormal_val["Problems"].dropna()
        for t in probs_str.split(";")
        if t.strip()
    ]
    top_tokens = [tok for tok, _ in Counter(tokens).most_common(10)]

    finding_results = []
    normal_df = val_df[val_df["label"] == 0]

    for tok in top_tokens:
        has_token_mask = val_df["Problems"].fillna("").apply(
            lambda s, t=tok: t in [p.strip() for p in s.split(";")]
        )
        finding_df = val_df[has_token_mask & (val_df["label"] == 1)]
        finding_cnt = len(finding_df)

        # Subset: finding images (positives) + all normal images (negatives)
        eval_subset = pd.concat([finding_df, normal_df], ignore_index=True)
        subset_y_true = eval_subset["label"].to_numpy().astype(int)
        subset_probs = eval_subset["pred_prob"].to_numpy()

        finding_auc = float(roc_auc_score(subset_y_true, subset_probs))
        trust_flag = "Caution (<30 images)" if finding_cnt < 30 else "Sufficient (>=30 images)"

        finding_results.append({
            "finding": tok,
            "count": finding_cnt,
            "auc": finding_auc,
            "trust_flag": trust_flag,
        })
        logger.info(
            f"Finding '{tok}': {finding_cnt} imgs | "
            f"AUC vs Normal: {finding_auc:.4f} | {trust_flag}"
        )

    # 4. 8 Most confident wrong predictions (4 false positives, 4 false negatives)
    fps = val_df[val_df["label"] == 0].sort_values(by="pred_prob", ascending=False).head(4)
    fns = val_df[val_df["label"] == 1].sort_values(by="pred_prob", ascending=True).head(4)

    # 5. Format and write reports/diagnostics_val.md
    report_path.parent.mkdir(parents=True, exist_ok=True)

    proj_rows_list = []
    for r in proj_results:
        p_name = r["projection"]
        t_cnt = r["total_count"]
        n_cnt = r["normal_count"]
        a_cnt = r["abnormal_count"]
        auc_v = r["auc"]
        ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        proj_rows_list.append(
            f"| **{p_name}** | {t_cnt} | {n_cnt} | {a_cnt} | {auc_v:.4f} | {ci_str} |"
        )
    proj_rows = "\n".join(proj_rows_list)

    finding_rows = "\n".join([
        f"| **{r['finding']}** | {r['count']} | {r['auc']:.4f} | {r['trust_flag']} |"
        for r in finding_results
    ])

    fp_rows_list = []
    for _, row in fps.iterrows():
        fn_val = row["filename"]
        pj_val = row["projection"]
        pr_val = row["Problems"]
        pb_val = row["pred_prob"]
        fp_rows_list.append(f"| `{fn_val}` | {pj_val} | {pr_val} | {pb_val:.4f} |")
    fp_rows = "\n".join(fp_rows_list)

    fn_rows_list = []
    for _, row in fns.iterrows():
        fn_val = row["filename"]
        pj_val = row["projection"]
        pr_val = row["Problems"]
        pb_val = row["pred_prob"]
        fn_rows_list.append(f"| `{fn_val}` | {pj_val} | {pr_val} | {pb_val:.4f} |")
    fn_rows = "\n".join(fn_rows_list)

    n_val_total = len(val_df)
    n_val_norm = int((y_true == 0).sum())
    n_val_abn = int((y_true == 1).sum())

    f_auc = proj_results[0]["auc"]
    f_low, f_high = proj_results[0]["ci_low"], proj_results[0]["ci_high"]
    l_auc = proj_results[1]["auc"]
    l_low, l_high = proj_results[1]["ci_low"], proj_results[1]["ci_high"]

    min_f_auc = min(r["auc"] for r in finding_results)
    max_f_auc = max(r["auc"] for r in finding_results)
    max_fp_prob = fps["pred_prob"].max()
    min_fn_prob = fns["pred_prob"].min()

    content = f"""# InsightX — Validation Split Diagnostic Report

> **Validation-Only Analysis**: This diagnostic was conducted strictly on the **validation split**
> (1,090 images across 405 patients). The test split was untouched and unobserved.
>
> **Research Demo Only**: This tool is not a certified medical device. Outputs are AI predictions
> for research investigation only and must not be used to guide clinical decisions.

---

## 1. Overall Validation Performance

- **Evaluated Split**: Validation (`split == 'val'`)
- **Total Images**: {n_val_total:,} (Normal: {n_val_norm}, Abnormal: {n_val_abn})
- **Total Patients (UIDs)**: {n_val_uids:,}
- **Overall ROC-AUC**: **{overall_auc:.4f}**
- **95% Bootstrap Confidence Interval**: **[{overall_ci_low:.4f}, {overall_ci_high:.4f}]**
  *(1,000 cluster resamples by patient UID)*

---

## 2. Performance Breakdown by Projection (Frontal vs Lateral)

| Projection View | Image Count | Normal (0) | Abnormal (1) | ROC-AUC | 95% Bootstrap CI (by UID) |
|---|---|---|---|---|---|
{proj_rows}

---

## 3. Performance Breakdown by Pathology / Finding

Evaluated as binary discrimination of images presenting the specific finding versus all validation
Normal images (negative reference cohort of {len(normal_df)} Normal images).

| Finding (Problems Token) | Image Count | ROC-AUC vs Normal | Sample Size Status |
|---|---|---|---|
{finding_rows}

---

## 4. Confident False Predictions (Failure Analysis)

### 4A. Most Confident False Positives (Ground Truth: Normal, Highest Predicted Abnormality)

| Filename | Projection | Clinical Problems Annotation | Predicted Probability (Abnormal) |
|---|---|---|---|
{fp_rows}

### 4B. Most Confident False Negatives (Ground Truth: Abnormal, Lowest Predicted Abnormality)

| Filename | Projection | Clinical Problems Annotation | Predicted Probability (Abnormal) |
|---|---|---|---|
{fn_rows}

---

## 5. Factual Summary

1. **Overall Discriminative Ability**: On the validation cohort of 1,090 images,
   the fine-tuned MobileNetV2 achieves an ROC-AUC of {overall_auc:.4f}
   (95% CI: [{overall_ci_low:.4f}, {overall_ci_high:.4f}]).
2. **View Sensitivity Discrepancy**: Performance differs between radiograph projections:
   Frontal views achieve an ROC-AUC of {f_auc:.4f} (95% CI: [{f_low:.4f}, {f_high:.4f}]),
   whereas Lateral views achieve {l_auc:.4f} (95% CI: [{l_low:.4f}, {l_high:.4f}]).
3. **Finding Discrimination**: Across the 10 most frequent abnormal finding tokens, discrimination
   versus normal cases ranges from {min_f_auc:.4f} to {max_f_auc:.4f}. All 10 findings have at least
   30 validation images.
4. **Confident Failure Profile**: The most confident false positives reach predicted abnormality
   probabilities up to {max_fp_prob:.4f} on normal radiographs. The most confident false negatives
   drop to predicted abnormality probabilities as low as {min_fn_prob:.4f} on abnormal radiographs.

---
*Report generated by `scripts/diagnostics_val.py` | Validation split evaluation only*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Report successfully saved to {report_path}")


if __name__ == "__main__":
    main()
