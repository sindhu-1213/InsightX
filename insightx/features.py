"""Image feature extraction via MobileNetV2 and multimodal feature fusion."""

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_image_feature_extractor() -> tf.keras.Model:
    """Instantiate MobileNetV2 with Global Average Pooling for 1280-dim feature vectors.

    Returns:
        Keras Model producing 1280-dimensional visual embeddings.
    """
    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    model.trainable = False
    return model


def preprocess_image_in_memory(image: Image.Image | np.ndarray) -> np.ndarray:
    """Preprocess a single in-memory PIL image or numpy array for MobileNetV2.

    Adheres to Rule 4: uploaded images are processed in-memory only without disk persistence.

    Args:
        image: PIL Image or numpy array.

    Returns:
        Batch-ready numpy array shaped (1, 224, 224, 3).
    """
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        image = image.resize((224, 224))
        arr = np.array(image, dtype=np.float32)
    else:
        arr = tf.image.resize(image, [224, 224]).numpy()

    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    return preprocess_input(arr)


def extract_single_image_features(
    model: tf.keras.Model,
    image: Image.Image | np.ndarray,
) -> np.ndarray:
    """Extract a 1280-dim feature vector from an in-memory image.

    Args:
        model: MobileNetV2 feature extractor.
        image: In-memory image.

    Returns:
        1D feature vector of shape (1280,).
    """
    processed = preprocess_image_in_memory(image)
    embedding = model(processed, training=False).numpy()
    return embedding.squeeze()


def extract_batch_image_features(
    image_paths: list[str] | np.ndarray,
    batch_size: int = 64,
    model: tf.keras.Model | None = None,
) -> np.ndarray:
    """Extract features from a list of disk image paths using a tf.data prefetch pipeline.

    Args:
        image_paths: Sequence of image file paths.
        batch_size: Batch size for inference.
        model: MobileNetV2 model instance.

    Returns:
        Numpy array of shape (N, 1280).
    """
    if model is None:
        model = build_image_feature_extractor()

    def _load_and_preprocess(path: tf.Tensor) -> tf.Tensor:
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        return preprocess_input(img)

    path_ds = tf.data.Dataset.from_tensor_slices(list(image_paths))
    ds = path_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    features = model.predict(ds, verbose=1)
    return np.asarray(features)


def fuse_modalities(
    image_features: np.ndarray,
    text_features: np.ndarray,
) -> np.ndarray:
    """Horizontally concatenate visual (1280-dim) and clinical NLP (1000-dim) vectors.

    Args:
        image_features: Array of shape (N, 1280) or (1280,).
        text_features: Array of shape (N, 1000) or (1000,).

    Returns:
        Fused array of shape (N, 2280) or (2280,).
    """
    img_arr = np.asarray(image_features)
    txt_arr = np.asarray(text_features)

    if img_arr.ndim == 1:
        img_arr = img_arr.reshape(1, -1)
    if txt_arr.ndim == 1:
        txt_arr = txt_arr.reshape(1, -1)

    return np.hstack((img_arr, txt_arr))
