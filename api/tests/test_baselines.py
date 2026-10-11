"""Sanity tests for baseline feature alignment and split integrity."""

from pathlib import Path

import numpy as np
import pandas as pd


def test_no_uid_appears_in_more_than_one_split():
    """Assert that no patient uid appears in more than one partition of splits.csv."""
    splits_path = Path("data/processed/splits.csv")
    assert splits_path.is_file(), "data/processed/splits.csv missing"

    df = pd.read_csv(splits_path)
    train_uids = set(df[df["split"] == "train"]["uid"].unique())
    val_uids = set(df[df["split"] == "val"]["uid"].unique())
    test_uids = set(df[df["split"] == "test"]["uid"].unique())

    assert train_uids.isdisjoint(val_uids), f"Train/Val overlap: {train_uids & val_uids}"
    assert val_uids.isdisjoint(test_uids), f"Val/Test overlap: {val_uids & test_uids}"
    assert train_uids.isdisjoint(test_uids), f"Train/Test overlap: {train_uids & test_uids}"

    total_uids = train_uids | val_uids | test_uids
    assert total_uids == set(df["uid"].unique())


def test_feature_rows_align_with_splits_csv_order():
    """Assert that feature rows in image_features.npy align with splits.csv order."""
    splits_path = Path("data/processed/splits.csv")
    features_path = Path("data/processed/image_features.npy")
    index_path = Path("data/processed/feature_index.csv")

    assert splits_path.is_file(), "data/processed/splits.csv missing"
    assert features_path.is_file(), "data/processed/image_features.npy missing"
    assert index_path.is_file(), "data/processed/feature_index.csv missing"

    splits_df = pd.read_csv(splits_path)
    feature_index_df = pd.read_csv(index_path)
    features = np.load(features_path)

    assert len(features) == len(splits_df), (
        f"Length mismatch: features {len(features)} vs splits {len(splits_df)}"
    )
    assert features.shape == (len(splits_df), 1280), (
        f"Unexpected features shape: {features.shape}"
    )
    assert len(feature_index_df) == len(splits_df), (
        f"Index length mismatch: index {len(feature_index_df)} vs splits {len(splits_df)}"
    )

    # Filename order in feature_index.csv must exactly match splits.csv row for row
    assert feature_index_df["filename"].tolist() == splits_df["filename"].tolist(), (
        "Filenames in feature_index.csv do not match splits.csv row order"
    )
