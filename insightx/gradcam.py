"""Explainable AI: Grad-CAM visual attention heatmaps for chest radiographs.

Computes gradient-weighted class activation mapping (Grad-CAM) targeting the
fine-tuned MobileNetV2 sigmoid prediction output and the last convolutional feature map.

Adheres to Project Rules:
- Rule 1: Research demo disclaimer; never use the word 'diagnosis'.
- Rule 4: All image processing in-memory only.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from insightx import RESEARCH_DISCLAIMER
from insightx.features import preprocess_image_in_memory

_CACHED_MODEL: tf.keras.Model | None = None


def load_finetuned_model(
    model_path: Path | str = "models/mobilenet_finetuned.keras",
) -> tf.keras.Model:
    """Load and cache fine-tuned MobileNetV2 Keras model in memory."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from insightx.model_io import load_trained_model

        _CACHED_MODEL = load_trained_model()
    return _CACHED_MODEL


def get_model_components(
    model: tf.keras.Model,
) -> tuple[
    tf.keras.layers.Layer,
    tf.keras.layers.Layer,
    tf.keras.layers.Layer,
    tf.keras.layers.Layer,
]:
    """Extract base feature extractor and classification head layers."""
    base_layer = model.get_layer("mobilenetv2_1.00_224")
    gap_layer = model.get_layer("gap")
    drop_layer = model.get_layer("head_dropout")
    try:
        pred_layer = model.get_layer("prediction")
    except ValueError:
        pred_layer = model.get_layer("prediction_output")
    return base_layer, gap_layer, drop_layer, pred_layer


def compute_gradcam_heatmap(
    image: Image.Image | np.ndarray,
    model: tf.keras.Model | None = None,
    model_path: Path | str = "models/mobilenet_finetuned.keras",
) -> tuple[np.ndarray, float]:
    """Compute 2D Grad-CAM attention heatmap targeting final sigmoid output.

    Backpropagates the gradient of the predicted abnormality probability
    with respect to the last convolutional feature map of MobileNetV2.

    Args:
        image: In-memory PIL Image or numpy array (H, W) or (H, W, C).
        model: Pre-loaded fine-tuned Keras model (optional).
        model_path: Path to model if not already loaded.

    Returns:
        Tuple of:
            - heatmap_2d: Float32 array normalized to [0, 1] of shape (224, 224).
            - prediction_prob: Model sigmoid prediction probability in [0, 1].
    """
    if model is None:
        model = load_finetuned_model(model_path)

    base, gap, drop, pred = get_model_components(model)
    img_tensor = preprocess_image_in_memory(image)

    with tf.GradientTape() as tape:
        conv_outputs = base(img_tensor)
        tape.watch(conv_outputs)
        gap_out = gap(conv_outputs)
        drop_out = drop(gap_out, training=False)
        prediction_tensor = pred(drop_out)
        # Target the sigmoid probability for class 1 (Abnormal)
        target_score = prediction_tensor[:, 0]

    # Compute gradients of score w.r.t. last conv feature map
    grads = tape.gradient(target_score, conv_outputs)
    # Global average pooling of gradients gives channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))[0]  # shape (1280,)

    # Weight feature maps by gradient importance
    conv_features = conv_outputs[0]  # shape (7, 7, 1280)
    cam = tf.reduce_sum(conv_features * pooled_grads[tf.newaxis, tf.newaxis, :], axis=-1)

    # Apply ReLU (keep only features having positive impact on target class)
    cam = tf.maximum(cam, 0.0)
    max_val = tf.math.reduce_max(cam)
    if max_val > 0:
        cam = cam / (max_val + 1e-10)

    # Resize 7x7 CAM to standard 224x224
    cam_resized = tf.image.resize(cam[..., tf.newaxis], (224, 224), method="bilinear")
    heatmap_2d = tf.squeeze(cam_resized).numpy().astype(np.float32)
    pred_prob = float(prediction_tensor[0, 0].numpy())

    return heatmap_2d, pred_prob


def overlay_heatmap_on_image(
    image: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    colormap_name: str = "jet",
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend normalized 2D heatmap onto the original image, returning a PNG array.

    Args:
        image: Original in-memory image (PIL Image or numpy array).
        heatmap: 2D numpy array [0, 1] of shape (H, W).
        colormap_name: Matplotlib colormap (default 'jet').
        alpha: Transparency factor for heatmap overlay (0 = only image, 1 = only heatmap).

    Returns:
        uint8 numpy array of shape (H, W, 3) representing the blended PNG image array.
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = np.uint8(np.clip(image * 255.0, 0, 255))
            else:
                img_uint8 = np.uint8(np.clip(image, 0, 255))
        else:
            img_uint8 = image
        pil_img = Image.fromarray(img_uint8).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    # Resize image to match heatmap dimensions or vice versa (standardize to 224x224)
    target_size = (heatmap.shape[1], heatmap.shape[0])
    pil_resized = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_rgb = np.asarray(pil_resized, dtype=np.float32) / 255.0

    # Colorize heatmap
    cmap = plt.get_cmap(colormap_name)
    colored_heatmap = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]  # drop alpha channel

    # Blend
    blended = (1.0 - alpha) * img_rgb + alpha * colored_heatmap
    overlay_png_array = np.uint8(np.clip(blended * 255.0, 0, 255))
    return overlay_png_array


def generate_gradcam_overlay(
    image: Image.Image | np.ndarray,
    model: tf.keras.Model | None = None,
    model_path: Path | str = "models/mobilenet_finetuned.keras",
    colormap_name: str = "jet",
    alpha: float = 0.45,
) -> tuple[np.ndarray, float]:
    """Generate Grad-CAM heatmap overlay as a PNG array directly in memory.

    Adheres to Rule 4 (in-memory processing).

    Args:
        image: Input image (PIL or numpy array).
        model: Fine-tuned model (optional).
        model_path: Path to model if not loaded.
        colormap_name: Color map name.
        alpha: Overlay blending weight.

    Returns:
        Tuple of (png_overlay_array, prediction_probability).
    """
    heatmap_2d, pred_prob = compute_gradcam_heatmap(
        image=image,
        model=model,
        model_path=model_path,
    )
    overlay_array = overlay_heatmap_on_image(
        image=image,
        heatmap=heatmap_2d,
        colormap_name=colormap_name,
        alpha=alpha,
    )
    return overlay_array, pred_prob


def generate_8_test_gradcam_examples(
    splits_path: Path | str = "data/processed/splits.csv",
    images_dir: Path | str = "data/processed/images_224",
    calib_json_path: Path | str = "models/calibration.json",
    output_dir: Path | str = "reports/gradcam_examples",
) -> list[dict[str, Any]]:
    """Generate 8 Grad-CAM example overlays (4 correct, 4 incorrect) from test split.

    Selection:
        - 4 Correct: 2 True Normal (TN), 2 True Abnormal (TP)
        - 4 Incorrect: 2 False Positives (FP), 2 False Negatives (FN)

    Saves overlay PNGs and README.md into output_dir.
    Notice: Heatmaps are for research exploration only and are not clinically meaningful.

    Returns:
        List of generated example metadata dictionaries.
    """
    import json

    import pandas as pd

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_dir = Path(images_dir)
    df = pd.read_csv(splits_path)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    # Load operating threshold and calibrator
    thresh = 0.6522
    calib_file = Path(calib_json_path)
    calibrator = None
    if calib_file.is_file():
        with open(calib_file, encoding="utf-8") as f:
            calib_meta = json.load(f)
            thresh = calib_meta.get("threshold_youden", 0.6522)
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression()
            calibrator.coef_ = np.array([[calib_meta["coef"]]])
            calibrator.intercept_ = np.array([calib_meta["intercept"]])
            calibrator.classes_ = np.array([0, 1])

    model = load_finetuned_model()

    # Compute predictions for all test images
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    from insightx.evaluate import apply_platt_calibration

    def _load_img(p: tf.Tensor) -> tf.Tensor:
        raw = tf.io.read_file(p)
        im = tf.image.decode_png(raw, channels=3)
        im = tf.cast(im, tf.float32)
        return preprocess_input(im)

    test_paths = [str(img_dir / str(f).strip()) for f in test_df["filename"]]
    ds = tf.data.Dataset.from_tensor_slices(test_paths).map(_load_img).batch(32)
    uncal_probs = model.predict(ds, verbose=0).flatten()

    if calibrator is not None:
        probs = apply_platt_calibration(calibrator, uncal_probs)
    else:
        probs = uncal_probs

    test_df["pred_prob"] = probs
    test_df["pred_label"] = (probs >= thresh).astype(int)

    # 4 Correct: 2 TN (most confident), 2 TP (most confident)
    tn_cases = test_df[(test_df["label"] == 0) & (test_df["pred_label"] == 0)].sort_values(
        by="pred_prob", ascending=True
    ).head(2)
    tp_cases = test_df[(test_df["label"] == 1) & (test_df["pred_label"] == 1)].sort_values(
        by="pred_prob", ascending=False
    ).head(2)

    # 4 Incorrect: 2 FP (highest prob), 2 FN (lowest prob)
    fp_cases = test_df[(test_df["label"] == 0) & (test_df["pred_label"] == 1)].sort_values(
        by="pred_prob", ascending=False
    ).head(2)
    fn_cases = test_df[(test_df["label"] == 1) & (test_df["pred_label"] == 0)].sort_values(
        by="pred_prob", ascending=True
    ).head(2)

    selected_groups = [
        ("correct_tn", tn_cases, "Correct (True Normal)"),
        ("correct_tp", tp_cases, "Correct (True Abnormal)"),
        ("incorrect_fp", fp_cases, "Incorrect (False Positive)"),
        ("incorrect_fn", fn_cases, "Incorrect (False Negative)"),
    ]

    results = []
    idx = 1
    md_rows = []

    for prefix, group_df, desc in selected_groups:
        for _, row in group_df.iterrows():
            filename = str(row["filename"]).strip()
            uid = row["uid"]
            true_label = "Abnormal" if row["label"] == 1 else "Normal"
            pred_prob = float(row["pred_prob"])
            pred_class = "Abnormal" if row["pred_label"] == 1 else "Normal"
            problem = str(row.get("Problems", "None"))
            proj = str(row.get("projection", "Unknown"))

            img_file = img_dir / filename
            image = Image.open(img_file).convert("RGB")

            overlay_array, _ = generate_gradcam_overlay(
                image=image,
                model=model,
            )

            out_filename = f"{idx:02d}_{prefix}_{filename}"
            save_path = out_path / out_filename
            Image.fromarray(overlay_array).save(save_path, format="PNG")

            meta = {
                "example_index": idx,
                "category": desc,
                "filename": filename,
                "uid": uid,
                "projection": proj,
                "ground_truth": true_label,
                "predicted_label": pred_class,
                "predicted_probability": round(pred_prob, 4),
                "problems": problem,
                "output_image": str(save_path),
                "disclaimer": RESEARCH_DISCLAIMER,
            }
            results.append(meta)
            md_rows.append(
                f"| `{filename}` | {desc} | {proj} | {true_label} | "
                f"{pred_prob:.4f} | {problem} | `{out_filename}` |"
            )
            print(
                f"[Grad-CAM] Saved {out_filename} | {desc} | Pred Prob: {pred_prob:.4f}"
            )
            idx += 1

    # Write summary README.md
    readme_path = out_path / "README.md"
    readme_lines = [
        "# InsightX: Grad-CAM Explainability Examples (Test Split)",
        "",
        "> **Research Demo Only - This tool is not a certified medical device.**",
        "> Outputs are AI predictions and saliency maps for research investigation only.",
        "> **Heatmaps are gradient-weighted activations of network feature maps and must",
        "> NOT be interpreted as clinically verified or diagnostic.**",
        "",
        "The following 8 test cases illustrate model visual attention across 4 correct and",
        f"4 incorrect predictions under the validation-chosen threshold (`{thresh:.4f}`).",
        "",
        "| Filename | Category | Projection | Ground Truth | Pred Prob | Problems | Overlay File |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ] + md_rows + [""]

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))

    return results


if __name__ == "__main__":
    generate_8_test_gradcam_examples()

