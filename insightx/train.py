"""Training and fine-tuning routines for InsightX: MobileNetV2 on Chest Radiographs.

Notice:
    Research Demo Only - This tool is not a certified medical device.
    Outputs are AI predictions/findings for research purposes only and
    must not be used to guide clinical decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("insightx.train")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import tensorflow as tf

    tf.random.set_seed(seed)


def build_data_augmentation() -> Any:
    """Construct light augmentation layers safe for chest radiographs.

    Strictly avoids horizontal flips to preserve cardiac and visceral laterality.
    Applies gentle rotation (~5 degrees), zoom (0.05), brightness (0.1), and contrast (0.1).
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(5.0 / 360.0, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(height_factor=(-0.05, 0.05), fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(factor=0.1, value_range=(0.0, 255.0)),
            tf.keras.layers.RandomContrast(factor=0.1),
        ],
        name="light_augmentation",
    )


def load_train_val_data(
    splits_path: Path | str = "data/processed/splits.csv",
    smoke: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and val splits from splits.csv, strictly ignoring the test split.

    Args:
        splits_path: Path to splits.csv.
        smoke: If True, extract 256 train and 128 val samples (stratified).
        seed: Random seed for stratified sampling.

    Returns:
        Tuple of (train_df, val_df).
    """
    df = pd.read_csv(splits_path)

    # Strictly isolate train and val splits; never touch the test split
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    if smoke:
        logger.info(
            f"Smoke mode: subsampling {len(train_df)} train -> 256, {len(val_df)} val -> 128"
        )
        train_df, _ = train_test_split(
            train_df,
            train_size=256,
            stratify=train_df["label"],
            random_state=seed,
        )
        train_df = train_df.reset_index(drop=True)

        val_df, _ = train_test_split(
            val_df,
            train_size=128,
            stratify=val_df["label"],
            random_state=seed,
        )
        val_df = val_df.reset_index(drop=True)

    return train_df, val_df


def build_dataset_pipeline(
    df: pd.DataFrame,
    images_dir: Path | str = "data/processed/images_224",
    batch_size: int = 32,
    is_training: bool = False,
    augmenter: Any = None,
    seed: int = 42,
) -> Any:
    """Build tf.data input pipeline for chest radiographs.

    Reads 224x224 grayscale PNGs with 3 channels, casts to float32,
    applies MobileNetV2 preprocess_input, caches, and prefetches.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img_folder = Path(images_dir)
    file_paths = [str(img_folder / str(f).strip()) for f in df["filename"]]
    labels = df["label"].to_numpy().astype(np.float32)

    def _load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_png(img_bytes, channels=3)
        img = tf.cast(img, tf.float32)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    if is_training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 2048),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        if augmenter is not None:
            ds = ds.map(
                lambda x, y: (preprocess_input(augmenter(x, training=True)), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda x, y: (preprocess_input(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    else:
        ds = ds.map(
            lambda x, y: (preprocess_input(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_mobilenet_model() -> tuple[Any, Any]:
    """Construct MobileNetV2 with custom classification head.

    Architecture:
        MobileNetV2 (ImageNet weights, include_top=False) ->
        GlobalAveragePooling2D -> Dropout(0.3) -> Dense(1, sigmoid, name='prediction')
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2

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
    return model, base_model


def find_last_conv_layer(model: Any) -> str:
    """Discover the name of the last convolutional layer for Grad-CAM explainability."""
    import tensorflow as tf

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if (
                    isinstance(sub_layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
                    or "relu" in sub_layer.name.lower()
                    or "conv" in sub_layer.name.lower()
                ):
                    return sub_layer.name
    for layer in reversed(model.layers):
        if (
            isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D))
            or "conv" in layer.name.lower()
        ):
            return layer.name
    return "out_relu"


class EpochTimerCallback:
    """Track execution duration in seconds for each epoch."""

    def __init__(self) -> None:
        import tensorflow as tf

        class _Callback(tf.keras.callbacks.Callback):
            def __init__(self) -> None:
                super().__init__()
                self.epoch_times: list[float] = []
                self._start_time: float = 0.0

            def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                self._start_time = time.time()

            def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
                duration = time.time() - self._start_time
                self.epoch_times.append(duration)

        self._cb = _Callback()

    @property
    def callback(self) -> Any:
        return self._cb

    @property
    def epoch_times(self) -> list[float]:
        return self._cb.epoch_times


def plot_training_curves(
    history_a: dict[str, list[float]],
    history_b: dict[str, list[float]] | None,
    output_path: Path | str = "reports/training_curves.png",
) -> None:
    """Plot and save training and validation loss and AUC curves across both phases."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _combine(metric: str) -> list[float]:
        vals = list(history_a.get(metric, []))
        if history_b and metric in history_b:
            vals.extend(history_b[metric])
        return vals

    train_loss = _combine("loss")
    val_loss = _combine("val_loss")
    train_auc = _combine("auc")
    val_auc = _combine("val_auc")
    phase_a_len = len(history_a.get("loss", []))

    epochs_range = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs_range, train_loss, "o-", label="Train Loss", color="#1976D2")
    axes[0].plot(epochs_range, val_loss, "s--", label="Val Loss", color="#D32F2F")
    if phase_a_len < len(train_loss):
        axes[0].axvline(
            phase_a_len + 0.5,
            color="gray",
            linestyle=":",
            label="Phase B (Fine-tuning)",
        )
    axes[0].set_title("Binary Cross-Entropy Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # AUC plot
    axes[1].plot(epochs_range, train_auc, "o-", label="Train AUC", color="#388E3C")
    axes[1].plot(epochs_range, val_auc, "s--", label="Val AUC", color="#F57C00")
    if phase_a_len < len(train_loss):
        axes[1].axvline(
            phase_a_len + 0.5,
            color="gray",
            linestyle=":",
            label="Phase B (Fine-tuning)",
        )
    axes[1].set_title("Area Under ROC Curve (AUC)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("InsightX MobileNetV2 Fine-Tuning Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Training curves saved to: {out}")


def train_mobilenet_finetuned(
    splits_path: Path | str = "data/processed/splits.csv",
    images_dir: Path | str = "data/processed/images_224",
    epochs_head: int = 5,
    epochs_ft: int = 15,
    batch_size: int = 32,
    seed: int = 42,
    smoke: bool = False,
    model_save_path: Path | str = "models/mobilenet_finetuned.keras",
    curves_save_path: Path | str = "reports/training_curves.png",
    log_save_path: Path | str = "reports/train_log.json",
) -> dict[str, Any]:
    """Execute two-phase MobileNetV2 fine-tuning on chest radiographs.

    Phase A: Train classification head with frozen base (Adam lr=1e-3).
    Phase B: Unfreeze last 30 layers (BatchNorm frozen) up to epochs_ft (Adam lr=1e-5).
    """
    import tensorflow as tf

    set_random_seeds(seed)

    # Adjust outputs and epochs for smoke run
    if smoke:
        epochs_head = 1
        epochs_ft = 1
        model_save_path = Path("models/smoke/mobilenet_finetuned.keras")
        curves_save_path = Path("reports/smoke/training_curves.png")
        log_save_path = Path("reports/smoke/train_log.json")
        logger.info(
            f"[SMOKE] Redirecting outputs to: {model_save_path.parent}, {log_save_path.parent}"
        )

    t_total_start = time.time()

    # 1. Load train and val data (test split is strictly excluded)
    train_df, val_df = load_train_val_data(splits_path=splits_path, smoke=smoke, seed=seed)
    logger.info(f"Dataset split sizes: Train={len(train_df):,}, Val={len(val_df):,}")

    # 2. Compute class weights from train split
    y_train = train_df["label"].to_numpy().astype(int)
    classes = np.array([0, 1])
    weights_arr = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {0: float(weights_arr[0]), 1: float(weights_arr[1])}
    logger.info(f"Train Class Weights: {class_weights}")

    # 3. Build data pipelines
    augmenter = build_data_augmentation()
    train_ds = build_dataset_pipeline(
        df=train_df,
        images_dir=images_dir,
        batch_size=batch_size,
        is_training=True,
        augmenter=augmenter,
        seed=seed,
    )
    val_ds = build_dataset_pipeline(
        df=val_df,
        images_dir=images_dir,
        batch_size=batch_size,
        is_training=False,
        seed=seed,
    )

    # 4. Instantiate Model
    model, base_model = build_mobilenet_model()
    last_conv_name = find_last_conv_layer(model)
    logger.info(f"Model instantiated. Discovered last conv layer: '{last_conv_name}'")

    # Metrics
    metrics = [
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    # -------------------------------------------------------------------------
    # Phase A: Train Head (Base Model Frozen)
    # -------------------------------------------------------------------------
    logger.info(f"[PHASE A] Training classification head for {epochs_head} epoch(s) (lr=1e-3)...")
    base_model.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    timer_a = EpochTimerCallback()
    hist_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        class_weight=class_weights,
        callbacks=[timer_a.callback],
        verbose=1,
    )

    # -------------------------------------------------------------------------
    # Phase B: Fine-tune last 30 layers (BatchNorm frozen)
    # -------------------------------------------------------------------------
    logger.info(
        f"[PHASE B] Unfreezing last 30 layers (BatchNorm frozen) up to {epochs_ft} epoch(s)..."
    )
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        if (
            isinstance(layer, tf.keras.layers.BatchNormalization)
            or "batchnormalization" in layer.__class__.__name__.lower()
        ):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    timer_b = EpochTimerCallback()
    callbacks_b = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        timer_b.callback,
    ]

    hist_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_ft,
        class_weight=class_weights,
        callbacks=callbacks_b,
        verbose=1,
    )

    total_runtime = time.time() - t_total_start
    logger.info(f"Training completed in {total_runtime:.2f} seconds.")

    # 5. Plot and save training curves
    plot_training_curves(
        hist_a.history,
        hist_b.history,
        output_path=curves_save_path,
    )

    # 6. Format and save train_log.json
    log_data: dict[str, Any] = {
        "tensorflow_version": tf.__version__,
        "seed": seed,
        "batch_size": batch_size,
        "smoke": smoke,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
        },
        "class_weights": class_weights,
        "epochs_head_configured": epochs_head,
        "epochs_ft_configured": epochs_ft,
        "seconds_per_epoch": {
            "phase_a": [float(round(s, 2)) for s in timer_a.epoch_times],
            "phase_b": [float(round(s, 2)) for s in timer_b.epoch_times],
        },
        "total_runtime_seconds": float(round(total_runtime, 2)),
        "metrics_per_epoch": {
            "phase_a": {
                k: [float(round(v, 4)) for v in vals] for k, vals in hist_a.history.items()
            },
            "phase_b": {
                k: [float(round(v, 4)) for v in vals] for k, vals in hist_b.history.items()
            },
        },
        "model_save_path": str(model_save_path),
        "last_conv_layer_name": last_conv_name,
    }

    log_file = Path(log_save_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Training log written to: {log_file}")

    return log_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune MobileNetV2 on Chest Radiographs (Normal vs Abnormal)"
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default="data/processed/splits.csv",
        help="Path to splits CSV file",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/processed/images_224",
        help="Path to resized 224x224 grayscale images directory",
    )
    parser.add_argument(
        "--epochs-head",
        type=int,
        default=5,
        help="Number of epochs for Phase A (head training)",
    )
    parser.add_argument(
        "--epochs-ft",
        type=int,
        default=15,
        help="Maximum epochs for Phase B (fine-tuning)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across libraries",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast sanity run with 256 train & 128 val samples, 1 epoch each phase",
    )
    args = parser.parse_args()

    train_mobilenet_finetuned(
        splits_path=args.splits_path,
        images_dir=args.images_dir,
        epochs_head=args.epochs_head,
        epochs_ft=args.epochs_ft,
        batch_size=args.batch_size,
        seed=args.seed,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
