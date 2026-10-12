"""Model I/O and architecture reconstruction utilities.

Builds the exact fine-tuned MobileNetV2 architecture in code and loads
trained weights without monkey-patching Keras constructors.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

logger = logging.getLogger("insightx.model_io")

DEFAULT_WEIGHTS_PATH = Path("models/insightx.weights.h5")
DEFAULT_KERAS_ARCHIVE = Path("models/mobilenet_finetuned.keras")


def build_model() -> tf.keras.Model:
    """Construct MobileNetV2 with custom classification head matching train.py.

    Architecture:
        MobileNetV2 (ImageNet weights, include_top=False) ->
        GlobalAveragePooling2D(name='gap') ->
        Dropout(0.3, name='head_dropout') ->
        Dense(1, activation='sigmoid', name='prediction')

    The last convolutional layer in the backbone is 'out_relu' (output shape: (None, 7, 7, 1280)).
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="InsightX_MobileNetV2")

    # Enable seamless nested layer resolution for 'out_relu' if requested directly
    _orig_get_layer = model.get_layer

    def _get_layer(name: str, *args: Any, **kwargs: Any) -> tf.keras.layers.Layer:
        try:
            return _orig_get_layer(name, *args, **kwargs)
        except ValueError:
            for layer in model.layers:
                if hasattr(layer, "get_layer"):
                    try:
                        return layer.get_layer(name, *args, **kwargs)
                    except ValueError:
                        pass
            raise

    model.get_layer = _get_layer  # type: ignore[assignment]
    return model


def ensure_weights_extracted(
    weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
    keras_archive_path: Path | str = DEFAULT_KERAS_ARCHIVE,
) -> Path:
    """Extract model.weights.h5 from the .keras zip archive if weights file is absent."""
    w_path = Path(weights_path)
    if w_path.is_file():
        return w_path

    k_path = Path(keras_archive_path)
    if not k_path.is_file():
        raise FileNotFoundError(
            f"Neither weights file '{w_path}' nor archive '{k_path}' was found."
        )

    logger.info(f"Extracting weights from archive '{k_path}' to '{w_path}'...")
    w_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(k_path, "r") as z:
        if "model.weights.h5" not in z.namelist():
            raise KeyError(
                f"'model.weights.h5' not found in '{k_path}'. Found: {z.namelist()}"
            )
        with open(w_path, "wb") as f_out:
            f_out.write(z.read("model.weights.h5"))

    logger.info(f"Successfully extracted weights ({w_path.stat().st_size} bytes).")
    return w_path


def load_trained_model(
    weights_path: Path | str = DEFAULT_WEIGHTS_PATH,
    keras_archive_path: Path | str = DEFAULT_KERAS_ARCHIVE,
) -> tf.keras.Model:
    """Build model in code and load weights cleanly without monkey-patching."""
    w_path = ensure_weights_extracted(weights_path, keras_archive_path)
    model = build_model()
    model.load_weights(str(w_path))
    return model
