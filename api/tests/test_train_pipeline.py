"""Unit tests for training data pipeline, split isolation, and model architecture."""

from pathlib import Path

import pandas as pd
import tensorflow as tf

from insightx.train import (
    build_data_augmentation,
    build_dataset_pipeline,
    build_mobilenet_model,
    find_last_conv_layer,
    load_train_val_data,
)


def test_no_test_split_rows_in_train_or_val():
    """Assert that load_train_val_data never touches or includes rows from the test split."""
    splits_path = Path("data/processed/splits.csv")
    assert splits_path.is_file(), "data/processed/splits.csv is missing"

    full_df = pd.read_csv(splits_path)
    test_uids = set(full_df[full_df["split"] == "test"]["uid"].unique())

    # 1. Full train/val data
    train_df, val_df = load_train_val_data(splits_path=splits_path, smoke=False)
    assert set(train_df["split"].unique()) == {"train"}
    assert set(val_df["split"].unique()) == {"val"}
    assert set(train_df["uid"].unique()).isdisjoint(test_uids), "Test patient leaked into train_df!"
    assert set(val_df["uid"].unique()).isdisjoint(test_uids), "Test patient leaked into val_df!"

    # 2. Smoke train/val data
    smoke_train, smoke_val = load_train_val_data(splits_path=splits_path, smoke=True)
    assert len(smoke_train) == 256
    assert len(smoke_val) == 128
    assert set(smoke_train["split"].unique()) == {"train"}
    assert set(smoke_val["split"].unique()) == {"val"}
    assert set(smoke_train["uid"].unique()).isdisjoint(test_uids)
    assert set(smoke_val["uid"].unique()).isdisjoint(test_uids)


def test_dataset_pipeline_shapes_and_dtypes():
    """Assert that build_dataset_pipeline outputs expected tensor shapes and float32 dtypes."""
    splits_path = Path("data/processed/splits.csv")
    images_dir = Path("data/processed/images_224")

    df = pd.read_csv(splits_path)
    sample_df = df[df["split"] == "train"].head(4).copy().reset_index(drop=True)

    ds = build_dataset_pipeline(
        df=sample_df,
        images_dir=images_dir,
        batch_size=2,
        is_training=False,
    )

    batch_x, batch_y = next(iter(ds))

    assert batch_x.shape == (2, 224, 224, 3), f"Unexpected image shape: {batch_x.shape}"
    assert batch_x.dtype == tf.float32, f"Unexpected image dtype: {batch_x.dtype}"
    assert batch_y.shape == (2,), f"Unexpected label shape: {batch_y.shape}"
    assert batch_y.dtype == tf.float32, f"Unexpected label dtype: {batch_y.dtype}"


def test_data_augmentation_layers_no_horizontal_flip():
    """Assert augmentation has no horizontal flips and preserves dimensions."""
    augmenter = build_data_augmentation()

    # Must strictly avoid horizontal flips for clinical laterality preservation
    for layer in augmenter.layers:
        assert "flip" not in layer.name.lower(), f"Prohibited flip layer detected: {layer.name}"

    dummy_input = tf.random.uniform((2, 224, 224, 3), minval=0.0, maxval=255.0)
    augmented = augmenter(dummy_input, training=True)
    assert augmented.shape == (2, 224, 224, 3)


def test_model_architecture_and_last_conv_discoverable():
    """Assert model has Dense(1, sigmoid) named 'prediction' and discoverable last conv layer."""
    model, base_model = build_mobilenet_model()

    # Prediction head verification
    pred_layer = model.get_layer("prediction")
    assert pred_layer is not None
    assert isinstance(pred_layer, tf.keras.layers.Dense)
    assert pred_layer.units == 1
    assert pred_layer.activation.__name__ == "sigmoid"

    # Base model should be frozen initially
    assert not base_model.trainable

    # Discover last conv layer for Grad-CAM
    last_conv = find_last_conv_layer(model)
    assert isinstance(last_conv, str)
    assert len(last_conv) > 0
    assert "relu" in last_conv.lower() or "conv" in last_conv.lower()
