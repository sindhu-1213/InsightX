"""Dataset loading, label definition, and patient-level splitting for InsightX.

Notice:
    Research Demo Only - This tool is not a certified medical device.
    Outputs are AI predictions/findings for research purposes only and
    must not be used to guide clinical decisions.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def map_problem_to_label(problem: Any) -> int | None:
    """Map the 'Problems' clinical annotation to a binary prediction label.

    Label mapping definition:
        - 0 (Normal): When Problems is exactly 'normal'.
        - 1 (Abnormal): For any other clinical finding or pathology value
                        (e.g., 'Cardiomegaly', 'Lung', 'Opacity', 'Scoliosis').
        - None (Drop): When Problems is 'No Indexing' (case-insensitive),
                       empty string, or NaN.

    Args:
        problem: Value from the 'Problems' column.

    Returns:
        0 for Normal, 1 for Abnormal, or None if the record should be excluded.
    """
    if problem is None or pd.isna(problem):
        return None

    prob_str = str(problem).strip()
    if not prob_str or prob_str.lower() == "no indexing":
        return None

    if prob_str == "normal":
        return 0

    return 1


def filter_by_existing_images(
    df: pd.DataFrame,
    images_dir: Path | str = "data/raw/images_normalized",
) -> pd.DataFrame:
    """Keep rows whose image file exists in the specified images folder.

    Args:
        df: DataFrame containing a 'filename' column.
        images_dir: Directory containing normalized chest X-rays.

    Returns:
        DataFrame filtered to rows where the image file exists on disk.
    """
    img_dir = Path(images_dir)
    exists_mask = df["filename"].apply(
        lambda fname: (img_dir / str(fname).strip()).is_file()
    )
    return df[exists_mask].reset_index(drop=True)


def load_dataset(
    projections_path: Path | str = "data/raw/indiana_projections.csv",
    reports_path: Path | str = "data/raw/indiana_reports.csv",
    images_dir: Path | str = "data/raw/images_normalized",
    require_images_exist: bool = False,
) -> pd.DataFrame:
    """Load projections and reports, merge on uid, filter images, and assign labels.

    Steps:
        1. Inner-joins projections and reports on 'uid'.
        2. Filters rows whose image file exists in images_dir if the folder exists
           and contains images (or if require_images_exist is True).
        3. Defines labels: 0 for exact 'normal', 1 for any other finding,
           dropping 'No Indexing' and empty rows.
        4. Retains the clinical 'findings' text column for text baselines.

    Args:
        projections_path: Path to projections CSV.
        reports_path: Path to reports CSV.
        images_dir: Directory of normalized images.
        require_images_exist: If True, strictly require image file existence
                              even if images_dir is missing or empty.

    Returns:
        Cleaned, labeled DataFrame.
    """
    df_proj = pd.read_csv(projections_path)
    df_repo = pd.read_csv(reports_path)

    # 1. Merge on uid (inner join)
    merged = pd.merge(df_proj, df_repo, on="uid", how="inner")

    # Filter to existing images if images directory is available
    img_path = Path(images_dir)
    if img_path.exists() and any(img_path.iterdir()):
        merged = filter_by_existing_images(merged, images_dir=img_path)
    elif require_images_exist:
        merged = filter_by_existing_images(merged, images_dir=img_path)

    # 2. Define label and drop 'No Indexing' or empty rows
    labels = merged["Problems"].apply(map_problem_to_label)
    merged["label"] = labels
    clean_df = merged[merged["label"].notna()].copy()
    clean_df["label"] = clean_df["label"].astype(int)
    clean_df = clean_df.reset_index(drop=True)

    # 3. Keep text column (findings) for baseline
    # Retain core columns: uid, filename, projection, findings, Problems, label
    columns_to_keep = [
        "uid",
        "filename",
        "projection",
        "findings",
        "Problems",
        "label",
    ]
    # Retain optional additional columns if present
    for col in ["impression", "indication", "comparison"]:
        if col in clean_df.columns and col not in columns_to_keep:
            columns_to_keep.append(col)

    clean_df = clean_df[columns_to_keep]

    # Print class counts
    normal_count = int((clean_df["label"] == 0).sum())
    abnormal_count = int((clean_df["label"] == 1).sum())
    total_count = len(clean_df)

    print("=" * 60)
    print("  InsightX Dataset Loaded")
    print(f"  Total Valid Samples: {total_count:,} across {clean_df['uid'].nunique():,} patients")
    print("  Class Counts (Problems column mapping):")
    print(f"    - Normal   (0): {normal_count:,} ({normal_count / total_count * 100:.1f}%)")
    print(f"    - Abnormal (1): {abnormal_count:,} ({abnormal_count / total_count * 100:.1f}%)")
    print("=" * 60)

    return clean_df


def create_patient_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_path: Path | str = "data/processed/splits.csv",
) -> pd.DataFrame:
    """Create 70/15/15 train/val/test splits using GroupShuffleSplit on patient uid.

    Ensures zero patient leakage: all projections for a single uid remain
    in exactly one split partition.

    Args:
        df: Cleaned and labeled DataFrame with 'uid' column.
        train_ratio: Proportion of patients for training (default 0.70).
        val_ratio: Proportion of patients for validation (default 0.15).
        test_ratio: Proportion of patients for test (default 0.15).
        seed: Random seed for reproducibility.
        output_path: Path to save splits CSV.

    Returns:
        DataFrame with an added 'split' column ('train', 'val', or 'test').
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Ratios must sum to 1.0.")

    # Split 1: Train (70%) vs Temp (30%)
    temp_ratio = val_ratio + test_ratio
    gss_1 = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_idx, temp_idx = next(gss_1.split(df, groups=df["uid"]))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # Split 2: Val (15%) vs Test (15%) -> 50/50 of temp
    relative_test_ratio = test_ratio / temp_ratio
    gss_2 = GroupShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=seed)
    val_idx, test_idx = next(gss_2.split(temp_df, groups=temp_df["uid"]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    splits_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Save to disk
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(out, index=False)
    print(f"\n[OK] Saved splits to: {out}")

    # Print summary table
    print_split_summary(splits_df)

    return splits_df


def print_split_summary(splits_df: pd.DataFrame) -> None:
    """Print formatted summary table showing sizes and class balance per split."""
    rows = []
    for split_name in ["train", "val", "test"]:
        sub = splits_df[splits_df["split"] == split_name]
        n_samples = len(sub)
        n_patients = sub["uid"].nunique()
        n_normal = int((sub["label"] == 0).sum())
        n_abnormal = int((sub["label"] == 1).sum())
        pct_abnormal = (n_abnormal / n_samples * 100) if n_samples > 0 else 0.0

        rows.append({
            "Split": split_name,
            "Samples": n_samples,
            "Patients": n_patients,
            "Normal (0)": n_normal,
            "Abnormal (1)": n_abnormal,
            "% Abnormal": f"{pct_abnormal:.1f}%",
        })

    # Total row
    total_samples = len(splits_df)
    total_patients = splits_df["uid"].nunique()
    total_norm = int((splits_df["label"] == 0).sum())
    total_abnorm = int((splits_df["label"] == 1).sum())
    pct_total = (total_abnorm / total_samples * 100) if total_samples > 0 else 0.0

    rows.append({
        "Split": "TOTAL",
        "Samples": total_samples,
        "Patients": total_patients,
        "Normal (0)": total_norm,
        "Abnormal (1)": total_abnorm,
        "% Abnormal": f"{pct_total:.1f}%",
    })

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("  SPLIT SUMMARY & CLASS BALANCE (70 / 15 / 15)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    df = load_dataset()
    create_patient_splits(df)
def patient_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset at the patient uid level into train and test sets."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["uid"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
