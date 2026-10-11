"""Model export utilities (ONNX / Joblib) and lightweight serving helpers.

Exports fine-tuned MobileNetV2 models to ONNX format using tf2onnx:
1. models/insightx.onnx: Single-output classifier for high-throughput prediction.
2. models/insightx_gradcam.onnx: Dual-output model providing both the last conv
   feature map (7, 7, 1280) and classification logit for zero-TensorFlow serving-time Grad-CAM.

Adheres to Project Rules:
- Rule 1: Research demo disclaimer; never use the word 'diagnosis'.
- Rule 2: ONNX weights are excluded from git via .gitignore.
- Rule 4: In-memory inference support without temporary file leakage.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
import tf2onnx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from insightx import RESEARCH_DISCLAIMER


def export_model_metadata(
    output_path: Path | str = "models/model_config.json",
    model_name: str = "InsightX-MobileNetV2",
    version: str = "1.0.0",
    metrics: dict[str, Any] | None = None,
) -> None:
    """Save lightweight model configuration and medical disclaimer metadata.

    This file is small (<10 KB) and can safely be version controlled.

    Args:
        output_path: Destination JSON file path.
        model_name: Model identifier.
        version: Model version tag.
        metrics: Dictionary of test metrics.
    """
    config = {
        "model_name": model_name,
        "version": version,
        "disclaimer": RESEARCH_DISCLAIMER,
        "modalities": {
            "image": "MobileNetV2 fine-tuned (224x224x3)",
            "classes": ["Normal (0)", "Abnormal (1)"],
        },
        "metrics": metrics or {},
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_inference_artifacts(
    model_path: Path | str = "models/insightx_rf_model.pkl",
    scaler_path: Path | str = "models/insightx_scaler.pkl",
) -> tuple[RandomForestClassifier, StandardScaler]:
    """Load trained legacy baseline classifier and feature scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def export_to_onnx(
    model_path: Path | str = "models/mobilenet_finetuned.keras",
    onnx_path: Path | str = "models/insightx.onnx",
    opset: int = 13,
) -> Path:
    """Export fine-tuned Keras model to ONNX format using tf2onnx.

    Args:
        model_path: Path to saved .keras fine-tuned model.
        onnx_path: Destination .onnx file path.
        opset: ONNX operator set version (default 13).

    Returns:
        Path to generated ONNX model file.
    """
    m_path = Path(model_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Model file not found at {m_path}")

    out_file = Path(onnx_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Export] Loading Keras model from {m_path}...")
    keras_model = tf.keras.models.load_model(str(m_path))

    input_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="image_input"),
    )

    print(f"[Export] Converting to ONNX (opset {opset}) via tf2onnx...")
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_signature,
        opset=opset,
    )

    with open(out_file, "wb") as f:
        f.write(model_proto.SerializeToString())

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[Export] Successfully saved ONNX model to {out_file} ({size_mb:.2f} MB)")
    return out_file


def export_gradcam_onnx(
    model_path: Path | str = "models/mobilenet_finetuned.keras",
    onnx_path: Path | str = "models/insightx_gradcam.onnx",
    weights_json_path: Path | str = "models/gradcam_head_weights.json",
    opset: int = 13,
) -> tuple[Path, Path]:
    """Export dual-output ONNX model returning feature map and logit for serving-time CAM.

    The exported ONNX model outputs:
        1. conv_feature_map: (None, 7, 7, 1280) from base MobileNetV2 output.
        2. logit_output: (None, 1) pre-sigmoid logit.

    Theoretical & Practical Note on TensorFlow-Free Serving-Time Grad-CAM:
    ---------------------------------------------------------------------
    For an architecture where the conv feature map A is followed by Global
    Average Pooling (GAP) and a single Dense unit with weights W:
        Logit z = sum_k (W_k * GAP(A_k)) + b
        d(z) / d(A_{i,j}^k) = W_k / (H * W)
    Thus the channel importance alpha_k = GAP(d(z)/d(A^k)) = W_k / (H * W).
    Since Grad-CAM heatmaps are min-max normalized, the constant 1/(H * W)
    cancels out completely:
        CAM = ReLU( sum_k W_k * A^k )
    Therefore, serving-time Grad-CAM can be computed in pure NumPy directly
    from the conv_feature_map and dense head weights W without needing
    TensorFlow or automatic differentiation tapes!

    Args:
        model_path: Source .keras model.
        onnx_path: Destination dual-output .onnx file path.
        weights_json_path: Path to save dense layer weights for NumPy CAM.
        opset: ONNX operator set version.

    Returns:
        Tuple of (onnx_path, weights_json_path).
    """
    m_path = Path(model_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Model file not found at {m_path}")

    out_onnx = Path(onnx_path)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    out_weights = Path(weights_json_path)

    print(f"[Export] Building dual-output Grad-CAM model from {m_path}...")
    keras_model = tf.keras.models.load_model(str(m_path))

    base_layer = keras_model.get_layer("mobilenetv2_1.00_224")
    gap_layer = keras_model.get_layer("gap")
    dense_layer = keras_model.get_layer("prediction_output")

    dense_w, dense_b = dense_layer.get_weights()  # shape (1280, 1) and (1,)

    # Construct explicit dual-output Keras graph
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")
    conv_out = base_layer(inputs)
    gap_out = gap_layer(conv_out)

    logit_layer = tf.keras.layers.Dense(1, activation=None, name="logit_output")
    logit_out = logit_layer(gap_out)

    dual_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, logit_out])
    logit_layer.set_weights([dense_w, dense_b])

    input_signature = (
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name="image_input"),
    )

    print(f"[Export] Converting dual-output model to ONNX (opset {opset})...")
    model_proto, _ = tf2onnx.convert.from_keras(
        dual_model,
        input_signature=input_signature,
        opset=opset,
    )

    with open(out_onnx, "wb") as f:
        f.write(model_proto.SerializeToString())

    # Save head weights for pure NumPy serving
    weights_payload = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "weights": dense_w.squeeze().tolist(),
        "bias": float(dense_b[0]),
        "feature_dim": 1280,
        "spatial_dim": [7, 7],
    }
    with open(out_weights, "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, indent=2)

    size_mb = out_onnx.stat().st_size / (1024 * 1024)
    print(f"[Export] Successfully saved dual ONNX model to {out_onnx} ({size_mb:.2f} MB)")
    print(f"[Export] Saved dense CAM weights to {out_weights}")
    return out_onnx, out_weights


def compute_serving_cam_numpy(
    conv_features: np.ndarray,
    dense_weights: np.ndarray,
) -> np.ndarray:
    """Compute pure NumPy Class Activation Mapping without TensorFlow.

    Mathematically identical to Grad-CAM on GlobalAveragePooling + Dense architectures.

    Args:
        conv_features: Feature map tensor of shape (7, 7, 1280) or (1, 7, 7, 1280).
        dense_weights: 1D array of length 1280 containing Dense layer weights.

    Returns:
        2D float32 numpy array normalized to [0, 1] of shape (7, 7).
    """
    if conv_features.ndim == 4:
        features = conv_features[0]
    else:
        features = conv_features

    weights = np.asarray(dense_weights).reshape(1, 1, -1)
    cam = np.sum(features * weights, axis=-1)

    # ReLU
    cam = np.maximum(cam, 0.0)
    max_val = np.max(cam)
    if max_val > 0:
        cam = cam / (max_val + 1e-10)

    return cam.astype(np.float32)


def run_full_export() -> None:
    """Run export pipeline for single and dual ONNX models."""
    export_to_onnx()
    export_gradcam_onnx()
    export_model_metadata()


if __name__ == "__main__":
    run_full_export()
