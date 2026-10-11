"""Dataset preparation and validation utility for InsightX.

Notice:
    Research Demo Only - This tool is not a certified medical device.
"""

import argparse
from pathlib import Path

import pandas as pd


def verify_dataset(data_dir: Path) -> None:
    """Check integrity of CSV files and normalized images in data/raw."""
    projections = data_dir / "indiana_projections.csv"
    reports = data_dir / "indiana_reports.csv"
    images_dir = data_dir / "images_normalized"

    print("Checking dataset integrity in:", data_dir)
    if projections.exists():
        df_p = pd.read_csv(projections)
        print(f"  [OK] indiana_projections.csv ({len(df_p):,} rows)")
    else:
        print(f"  [MISSING] {projections}")

    if reports.exists():
        df_r = pd.read_csv(reports)
        print(f"  [OK] indiana_reports.csv ({len(df_r):,} rows)")
    else:
        print(f"  [MISSING] {reports}")

    if images_dir.exists() and images_dir.is_dir():
        imgs = list(images_dir.glob("*.png"))
        print(f"  [OK] images_normalized/ ({len(imgs):,} PNG images found)")
    else:
        print("  [INFO] images_normalized/ directory not found or empty.")
        print("         To download Indiana University Chest X-rays, visit:")
        print("         https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university")


def main() -> None:
    parser = argparse.ArgumentParser(description="InsightX Dataset Utility")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to raw dataset folder",
    )
    args = parser.parse_args()
    verify_dataset(args.data_dir)


if __name__ == "__main__":
    main()
