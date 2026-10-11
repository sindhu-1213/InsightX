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
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuned model not found at {path}")
    if _CACHED_MODEL is None:
        _CACHED_MODEL = tf.keras.models.load_model(str(path))
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
    model_path: Path | str = "models/mobilenet_finetuned.keras",
    output_dir: Path | str = "reports/gradcam_examples",
) -> list[dict[str, Any]]:
    """Generate 8 Grad-CAM example overlays (4 normal, 4 abnormal from test split).

    Saves overlay PNGs into output_dir and includes research disclaimer.

    Returns:
        List of generated example metadata dictionaries.
    """
    import pandas as pd

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(splits_path)
    test_df = df[df["split"] == "test"]

    normal_cases = test_df[test_df["label"] == 0].head(4)
    abnormal_cases = test_df[test_df["label"] == 1].head(4)

    cases = []
    for _, row in normal_cases.iterrows():
        cases.append((row, False))
    for _, row in abnormal_cases.iterrows():
        cases.append((row, True))

    model = load_finetuned_model(model_path)
    img_dir = Path("data/raw/images_normalized")
    results = []

    for idx, (row, is_abnormal) in enumerate(cases, start=1):
        filename = str(row["filename"])
        uid = row["uid"]
        problem = str(row.get("Problems", "normal" if not is_abnormal else "abnormal"))
        label_str = "normal" if not is_abnormal else "abnormal"

        # Check if actual image file exists on disk
        img_file = img_dir / filename
        if not img_file.is_file():
            raise FileNotFoundError(
                f"Expected image '{img_file}' on disk in '{img_dir}' but file not found. "
                f"Refusing to generate Grad-CAM examples without real radiograph files."
            )
        image = Image.open(img_file).convert("RGB")

        # Generate overlay PNG array
        overlay_array, pred_prob = generate_gradcam_overlay(
            image=image,
            model=model,
        )

        # Save overlay PNG
        clean_problem = problem.lower().replace(" ", "_").replace("/", "_")
        out_filename = f"example_{idx:02d}_{label_str}_uid{uid}_{clean_problem}.png"
        save_path = out_path / out_filename

        overlay_pil = Image.fromarray(overlay_array)
        overlay_pil.save(save_path, format="PNG")

        meta = {
            "example_index": idx,
            "filename": filename,
            "uid": uid,
            "finding_class": label_str,
            "annotated_problem": problem,
            "predicted_abnormality_probability": round(pred_prob, 4),
            "output_image": str(save_path),
            "disclaimer": RESEARCH_DISCLAIMER,
        }
        results.append(meta)
        print(
            f"[Grad-CAM] Saved {save_path.name} | "
            f"Class: {label_str} | Pred Prob: {pred_prob:.4f}"
        )

    return results


if __name__ == "__main__":
    generate_8_test_gradcam_examples()
