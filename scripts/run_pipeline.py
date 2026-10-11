"""End-to-end training and evaluation pipeline runner."""

from pathlib import Path

from insightx import RESEARCH_DISCLAIMER
from insightx.data import load_and_merge_data, patient_level_split
from insightx.export import export_model_metadata


def main() -> None:
    print("=" * 60)
    print("  InsightX - Multimodal Pipeline Runner")
    print(f"  Notice: {RESEARCH_DISCLAIMER}")
    print("=" * 60)

    raw_dir = Path("data/raw")
    if not (raw_dir / "indiana_reports.csv").exists():
        print(f"Dataset not found at {raw_dir}. Please run scripts/download_data.py first.")
        return

    print("Loading and cleaning reports and projections...")
    df = load_and_merge_data()
    print(f"Loaded {len(df)} clinical entries across {df['uid'].nunique()} patients.")

    print("Executing patient-level split (Rule 3)...")
    train_df, test_df = patient_level_split(df, test_size=0.2, random_state=42)
    print(f"Train samples: {len(train_df)} | Test samples: {len(test_df)}")

    # Export initial metadata configuration
    export_model_metadata(output_path="models/model_config.json")
    print("Pipeline initialized successfully.")


if __name__ == "__main__":
    main()
