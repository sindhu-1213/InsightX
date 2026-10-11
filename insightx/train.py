"""Training and fine-tuning routines for InsightX: MobileNetV2 End-to-End & Baseline Models.

Google Colab Quickstart:
-------------------------
To train on Google Colab with GPU acceleration and data mounted from Google Drive:

1. Mount Google Drive:
   >>> from google.colab import drive
   >>> drive.mount('/content/drive')

2. Locate or uncompress the Indiana University dataset images:
   >>> # Recommended for fast I/O (extract images to local Colab SSD):
   >>> !unzip -q "/content/drive/MyDrive/InsightX/images_normalized.zip" \
   ...     -d "/content/data/raw/images_normalized"

3. Clone repo or sync code:
   >>> !git clone https://github.com/<your-username>/InsightX.git /content/InsightX
   >>> %cd /content/InsightX
   >>> !pip install -q -r requirements.txt

4. Run fine-tuning end-to-end:
   >>> !python -m insightx.train \
   ...     --data-path /content/data/raw/images_normalized \
   ...     --splits-path data/processed/splits.csv \
   ...     --epochs-a 5 \
   ...     --epochs-b 15 \
   ...     --batch-size 32

Notice:
    Research Demo Only - This tool is not a certified medical device.
    Outputs are AI predictions/findings for research purposes only and
    must not be used to guide clinical decisions.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from insightx import RESEARCH_DISCLAIMER

CLUSTER_PROFILE_NAMES: dict[int, str] = {
    0: "Unremarkable / Baseline Findings",
    1: "Cardiac & Mediastinal Conditions",
    2: "Pulmonary Opacities / Effusions",
    3: "Degenerative / Osseous Findings",
    4: "Post-Surgical / Hardware",
}


def build_data_augmentation() -> tf.keras.Sequential:
    """Construct light augmentation layers safe for chest radiographs.

    Strictly avoids horizontal flips (flips cardiac and visceral laterality).
    Applies gentle rotation and brightness adjustments only.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.04, fill_mode="nearest"),
            tf.keras.layers.RandomBrightness(factor=0.08),
        ],
        name="light_augmentation",
    )


def build_dataset_pipeline(
    df: pd.DataFrame,
    images_dir: Path | str,
    batch_size: int = 32,
    is_training: bool = False,
    augmenter: tf.keras.Sequential | None = None,
) -> tf.data.Dataset:
    """Build tf.data input pipeline for chest radiographs.

    Reads 224x224 grayscale images, replicates to 3 channels, applies MobileNetV2
    preprocessing, and optionally injects light augmentation.
    """
    img_folder = Path(images_dir)
    file_paths = [str(img_folder / str(f).strip()) for f in df["filename"]]
    labels = df["label"].to_numpy().astype(np.float32)

    def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        img_bytes = tf.io.read_file(path)
        # Decode grayscale (1 channel) and replicate to 3 channels for MobileNetV2
        img = tf.image.decode_png(img_bytes, channels=1)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, [224, 224])
        img = preprocess_input(tf.cast(img, tf.float32))
        return img, label

    def _augment(img: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if augmenter is not None:
            img = augmenter(img, training=True)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), 2000), seed=42)

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training and augmenter is not None:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_mobilenet_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Construct MobileNetV2 with custom binary classification head.

    Architecture:
        Input(224, 224, 3) -> MobileNetV2 Base -> GlobalAveragePooling2D
        -> Dropout(0.3) -> Dense(1, activation='sigmoid')

    Returns:
        Tuple of (full_model, base_model).
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
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="InsightX_MobileNetV2")
    return model, base_model


def plot_training_curves(
    history_a: dict[str, list[float]],
    history_b: dict[str, list[float]] | None,
    output_path: Path | str = "reports/training_curves.png",
) -> None:
    """Plot and save loss and AUC training curves across both phases."""
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
    print(f"[OK] Training curves saved to: {out}")


def train_mobilenet_finetuned(
    splits_path: Path | str = "data/processed/splits.csv",
    images_dir: Path | str = "data/raw/images_normalized",
    epochs_a: int = 5,
    epochs_b: int = 15,
    batch_size: int = 32,
    model_save_path: Path | str = "models/mobilenet_finetuned.keras",
    curves_save_path: Path | str = "reports/training_curves.png",
    log_save_path: Path | str = "reports/train_log.json",
) -> dict[str, Any]:
    """Execute two-phase MobileNetV2 fine-tuning with patient splits and class weighting."""
    df = pd.read_csv(splits_path)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Verify existing image files
    img_folder = Path(images_dir)
    all_files = [img_folder / str(fn).strip() for fn in df["filename"]]
    existing_count = sum(1 for f in all_files if f.is_file())

    if existing_count < len(all_files):
        raise FileNotFoundError(
            f"Expected image directory '{img_folder}' containing {len(all_files)} images from "
            f"'{splits_path}', but found {existing_count} valid image files on disk. "
            f"Please download the Indiana University Chest X-ray images to '{img_folder}'."
        )

    # Compute class weights from train split only
    y_train = train_df["label"].to_numpy()
    classes = np.array([0, 1])
    weights_arr = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = {0: float(weights_arr[0]), 1: float(weights_arr[1])}
    print(f"[WEIGHTS] Computed Train Class Weights: {class_weights}")

    augmenter = build_data_augmentation()
    train_ds = build_dataset_pipeline(
        train_df, img_folder, batch_size=batch_size, is_training=True, augmenter=augmenter
    )
    val_ds = build_dataset_pipeline(val_df, img_folder, batch_size=batch_size, is_training=False)
    test_ds = build_dataset_pipeline(test_df, img_folder, batch_size=batch_size, is_training=False)

    model, base_model = build_mobilenet_model()

    # Phase A: Train new classification head with frozen base
    print(f"\n[PHASE A] Training classification head for {epochs_a} epochs (base frozen)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    hist_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_a,
        class_weight=class_weights,
        verbose=1,
    )

    # Phase B: Fine-tune last 30 layers with lower learning rate
    print(f"\n[PHASE B] Unfreezing last 30 layers for up to {epochs_b} epochs (lr=1e-5)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
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
    ]

    hist_b = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_b,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on held-out test split
    print("\n[TEST] Evaluating best model on test split...")
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=1)

    # Plot and save curves
    plot_training_curves(
        hist_a.history, hist_b.history, output_path=curves_save_path
    )

    # Build and write train log
    log_data: dict[str, Any] = {
        "disclaimer": RESEARCH_DISCLAIMER,
        "config": {
            "epochs_phase_a": epochs_a,
            "epochs_phase_b": epochs_b,
            "batch_size": batch_size,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "class_weights": class_weights,
        },
        "phase_a_history": {k: [float(v) for v in vals] for k, vals in hist_a.history.items()},
        "phase_b_history": {k: [float(v) for v in vals] for k, vals in hist_b.history.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }

    log_path = Path(log_save_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print(f"[OK] Training log written to: {log_path}")
    return log_data




# -----------------------------------------------------------------------------
# Baseline Classical ML & Clustering Routines (Backward Compatibility)
# -----------------------------------------------------------------------------

def fit_patient_clusters(
    fused_features: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
) -> tuple[KMeans, StandardScaler, np.ndarray]:
    """Cluster patient profiles using scaled multimodal fused features."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(fused_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    return kmeans, scaler, clusters


def train_random_forest_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray | pd.Series,
    tune_hyperparameters: bool = False,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, StandardScaler]:
    """Train Random Forest classifier for binary Normal (0) vs Abnormal (1) finding prediction."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    if tune_hyperparameters:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        }
        base_rf = RandomForestClassifier(
            random_state=random_state, class_weight="balanced", n_jobs=-1
        )
        grid = GridSearchCV(base_rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
        grid.fit(X_scaled, y_train)
        model = grid.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            max_features="sqrt",
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_scaled, y_train)

    return model, scaler


def evaluate_classifier(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Evaluate classifier performance on held-out test data."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "report": classification_report(
            y_test, y_pred, target_names=["Normal", "Abnormal"], output_dict=True
        ),
    }


def save_artifacts(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    model_path: Path | str = "models/insightx_rf_model.pkl",
    scaler_path: Path | str = "models/insightx_scaler.pkl",
) -> None:
    """Save trained model and feature scaler to disk."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="InsightX: Fine-tune MobileNetV2 End-to-End on Chest Radiographs"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/images_normalized",
        help="Path to folder containing normalized chest X-ray images",
    )
    parser.add_argument(
        "--splits-path",
        type=str,
        default="data/processed/splits.csv",
        help="Path to splits CSV file",
    )
    parser.add_argument(
        "--epochs-a",
        type=int,
        default=5,
        help="Number of epochs for Phase A (head training)",
    )
    parser.add_argument(
        "--epochs-b",
        type=int,
        default=15,
        help="Number of epochs for Phase B (fine-tuning)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/mobilenet_finetuned.keras",
        help="Output destination path for saved fine-tuned Keras model",
    )
    parser.add_argument(
        "--curves-path",
        type=str,
        default="reports/training_curves.png",
        help="Output destination path for training curves figure",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="reports/train_log.json",
        help="Output destination path for JSON training metrics log",
    )
    args = parser.parse_args()

    train_mobilenet_finetuned(
        splits_path=args.splits_path,
        images_dir=args.data_path,
        epochs_a=args.epochs_a,
        epochs_b=args.epochs_b,
        batch_size=args.batch_size,
        model_save_path=args.model_path,
        curves_save_path=args.curves_path,
        log_save_path=args.log_path,
    )


if __name__ == "__main__":
    main()
