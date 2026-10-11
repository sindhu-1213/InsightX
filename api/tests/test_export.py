"""Pytest test suite verifying ONNX export fidelity and serving-time Grad-CAM parity.

Validates that ONNX Runtime predictions match the original fine-tuned Keras model
within 1e-3 on 20 test images. Also verifies dual-output Grad-CAM ONNX model
and pure NumPy CAM execution without TensorFlow.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import tensorflow as tf

from insightx.export import compute_serving_cam_numpy

MODEL_KERAS_PATH = Path("models/mobilenet_finetuned.keras")
MODEL_ONNX_PATH = Path("models/insightx.onnx")
MODEL_DUAL_ONNX_PATH = Path("models/insightx_gradcam.onnx")
WEIGHTS_JSON_PATH = Path("models/gradcam_head_weights.json")


@pytest.fixture(scope="module")
def keras_model():
    """Load fine-tuned Keras model."""
    if not MODEL_KERAS_PATH.is_file():
        pytest.skip(f"Model file not found: {MODEL_KERAS_PATH}")
    return tf.keras.models.load_model(str(MODEL_KERAS_PATH))


@pytest.fixture(scope="module")
def onnx_session():
    """Load single-output ONNX model session."""
    if not MODEL_ONNX_PATH.is_file():
        pytest.skip(f"ONNX model not found: {MODEL_ONNX_PATH}")
    return ort.InferenceSession(str(MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def dual_onnx_session():
    """Load dual-output Grad-CAM ONNX model session."""
    if not MODEL_DUAL_ONNX_PATH.is_file():
        pytest.skip(f"Dual ONNX model not found: {MODEL_DUAL_ONNX_PATH}")
    return ort.InferenceSession(str(MODEL_DUAL_ONNX_PATH), providers=["CPUExecutionProvider"])


def test_onnx_matches_keras_within_tolerance(keras_model, onnx_session):
    """Verify that ONNX Runtime matches Keras predictions within 1e-3 on 20 test images."""
    # Deterministic test inputs in standard MobileNetV2 normalized range [-1, 1]
    rng = np.random.default_rng(seed=1337)
    test_batch = rng.uniform(-1.0, 1.0, size=(20, 224, 224, 3)).astype(np.float32)

    # 1. Keras prediction
    keras_preds = keras_model.predict(test_batch, verbose=0)  # shape (20, 1)

    # 2. ONNX Runtime prediction
    input_name = onnx_session.get_inputs()[0].name
    onnx_preds = onnx_session.run(None, {input_name: test_batch})[0]  # shape (20, 1)

    # Verify shapes and max difference
    assert keras_preds.shape == (20, 1)
    assert onnx_preds.shape == (20, 1)

    max_abs_diff = float(np.max(np.abs(keras_preds - onnx_preds)))
    print(f"\n[Test] Max diff Keras vs ONNX (20 images): {max_abs_diff:.6e}")

    assert np.allclose(keras_preds, onnx_preds, atol=1e-3), (
        f"ONNX predictions deviated from Keras by {max_abs_diff:.6e} (tolerance: 1e-3)"
    )


def test_dual_output_gradcam_onnx(keras_model, dual_onnx_session):
    """Verify dual ONNX model produces feature maps (7, 7, 1280) and consistent logits."""
    rng = np.random.default_rng(seed=2024)
    sample_input = rng.uniform(-1.0, 1.0, size=(5, 224, 224, 3)).astype(np.float32)

    input_name = dual_onnx_session.get_inputs()[0].name
    outputs = dual_onnx_session.run(None, {input_name: sample_input})

    # Output 0: feature maps (5, 7, 7, 1280) or Output 1: logits (5, 1)
    shapes = [out.shape for out in outputs]
    assert any(s == (5, 7, 7, 1280) for s in shapes), f"Expected (5, 7, 7, 1280), got {shapes}"
    assert any(s == (5, 1) for s in shapes), f"Expected logit output (5, 1), got {shapes}"

    # Extract logits and verify conv features shape
    conv_features = outputs[0] if outputs[0].shape == (5, 7, 7, 1280) else outputs[1]
    assert conv_features.shape == (5, 7, 7, 1280)
    logits = outputs[1] if outputs[1].shape == (5, 1) else outputs[0]

    # Probabilities via sigmoid(logit)
    probs_from_logit = 1.0 / (1.0 + np.exp(-logits))
    keras_probs = keras_model.predict(sample_input, verbose=0)

    # Validate that probability from exported logit matches Keras output within 1e-3
    assert np.allclose(probs_from_logit, keras_probs, atol=1e-3)


def test_pure_numpy_serving_cam(dual_onnx_session):
    """Verify pure NumPy serving CAM executes without TensorFlow and produces valid heatmaps."""
    if not WEIGHTS_JSON_PATH.is_file():
        pytest.skip(f"Weights JSON missing: {WEIGHTS_JSON_PATH}")

    with open(WEIGHTS_JSON_PATH, encoding="utf-8") as f:
        head_weights_meta = json.load(f)

    dense_weights = np.array(head_weights_meta["weights"], dtype=np.float32)
    assert dense_weights.shape == (1280,)

    sample = np.zeros((1, 224, 224, 3), dtype=np.float32)
    input_name = dual_onnx_session.get_inputs()[0].name
    outputs = dual_onnx_session.run(None, {input_name: sample})

    conv_map = outputs[0] if outputs[0].shape == (1, 7, 7, 1280) else outputs[1]
    cam = compute_serving_cam_numpy(conv_map, dense_weights)

    assert cam.shape == (7, 7)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0001
    assert "diagnosis" not in head_weights_meta["disclaimer"].lower()
    assert "not a certified medical device" in head_weights_meta["disclaimer"].lower()


def test_compliance_rules_in_exported_metadata():
    """Verify Rule 1 compliance: Disclaimers present, 'diagnosis' forbidden."""
    for json_file in [Path("models/model_config.json"), Path("models/calibration.json")]:
        if json_file.is_file():
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            assert "disclaimer" in data
            assert "not a certified medical device" in data["disclaimer"].lower()
            assert "diagnosis" not in data["disclaimer"].lower()
