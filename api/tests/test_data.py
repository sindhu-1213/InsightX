"""Tests for dataset loading, Problems label mapping, and patient-level splitting."""

from pathlib import Path

import numpy as np
import pandas as pd

from insightx.data import (
    create_patient_splits,
    filter_by_existing_images,
    map_problem_to_label,
)


def test_label_mapping_normal_and_cardiomegaly():
    """Verify Step 2 & 5: map_problem_to_label maps 'normal' -> 0 and 'Cardiomegaly' -> 1."""
    # Step 5 explicit requirements
    assert map_problem_to_label("normal") == 0
    assert map_problem_to_label("Cardiomegaly") == 1

    # Additional pathology checks
    assert map_problem_to_label("Lung") == 1
    assert map_problem_to_label("Calcified Granuloma") == 1
    assert map_problem_to_label("Opacity") == 1
    assert map_problem_to_label("Scoliosis") == 1

    # Exclusion checks: 'No Indexing', empty string, whitespace, and NaN
    assert map_problem_to_label("No Indexing") is None
    assert map_problem_to_label("no indexing") is None
    assert map_problem_to_label("") is None
    assert map_problem_to_label("   ") is None
    assert map_problem_to_label(None) is None
    assert map_problem_to_label(np.nan) is None


def test_no_uid_overlaps_between_splits():
    """Verify Step 4 & 5: no patient uid appears in more than one split partition."""
    # Synthetic dataset with multiple projections per patient
    sample_df = pd.DataFrame({
        "uid": [101, 101, 102, 102, 103, 103, 104, 104, 105, 105,
                106, 106, 107, 107, 108, 108, 109, 109, 110, 110],
        "filename": [f"img_{i}.png" for i in range(20)],
        "projection": ["Frontal", "Lateral"] * 10,
        "findings": ["Clear lungs"] * 20,
        "Problems": ["normal", "normal", "Cardiomegaly", "Cardiomegaly"] * 5,
        "label": [0, 0, 1, 1] * 5,
    })

    splits_df = create_patient_splits(
        sample_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        output_path="data/processed/test_splits_temp.csv",
    )

    train_uids = set(splits_df[splits_df["split"] == "train"]["uid"].unique())
    val_uids = set(splits_df[splits_df["split"] == "val"]["uid"].unique())
    test_uids = set(splits_df[splits_df["split"] == "test"]["uid"].unique())

    # Proving zero overlap across all pairs
    assert train_uids.isdisjoint(val_uids), f"Train/Val overlap: {train_uids & val_uids}"
    assert val_uids.isdisjoint(test_uids), f"Val/Test overlap: {val_uids & test_uids}"
    assert train_uids.isdisjoint(test_uids), f"Train/Test overlap: {train_uids & test_uids}"

    # Verify total unique patients preserved
    total_split_uids = train_uids | val_uids | test_uids
    assert total_split_uids == set(sample_df["uid"].unique())

    # Clean up temp file
    temp_file = Path("data/processed/test_splits_temp.csv")
    if temp_file.exists():
        temp_file.unlink()


def test_saved_splits_file_integrity():
    """Verify the saved data/processed/splits.csv file exists and satisfies split invariants."""
    splits_path = Path("data/processed/splits.csv")
    assert splits_path.is_file(), "data/processed/splits.csv was not generated."

    df = pd.read_csv(splits_path)
    assert "uid" in df.columns
    assert "filename" in df.columns
    assert "projection" in df.columns
    assert "findings" in df.columns
    assert "label" in df.columns
    assert "split" in df.columns

    train_uids = set(df[df["split"] == "train"]["uid"].unique())
    val_uids = set(df[df["split"] == "val"]["uid"].unique())
    test_uids = set(df[df["split"] == "test"]["uid"].unique())

    assert train_uids.isdisjoint(val_uids)
    assert val_uids.isdisjoint(test_uids)
    assert train_uids.isdisjoint(test_uids)

    # All labels must be strictly 0 or 1
    assert set(df["label"].unique()).issubset({0, 1})


def test_filter_by_existing_images_with_temp_dir(tmp_path: Path):
    """Verify image filter keeps only rows whose files exist in the images directory."""
    img_dir = tmp_path / "images_normalized"
    img_dir.mkdir()

    # Create dummy image files
    (img_dir / "exists_1.png").touch()
    (img_dir / "exists_2.png").touch()

    df = pd.DataFrame({
        "uid": [1, 2, 3],
        "filename": ["exists_1.png", "missing_image.png", "exists_2.png"],
    })

    filtered = filter_by_existing_images(df, images_dir=img_dir)
    assert len(filtered) == 2
    assert list(filtered["filename"]) == ["exists_1.png", "exists_2.png"]
