"""Fast parallel verification script for InsightX raw image dataset and projections CSV."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def check_image(path: Path) -> tuple[str, bool, str]:
    fname = path.name
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return fname, True, ""
    except Exception as e:
        return fname, False, str(e)


def verify_dataset(
    images_dir_str: str = "data/raw/images_normalized",
    projections_path_str: str = "data/raw/indiana_projections.csv",
    reports_path_str: str = "data/raw/indiana_reports.csv",
):
    images_dir = Path(images_dir_str)
    projections_path = Path(projections_path_str)
    reports_path = Path(reports_path_str)

    print("=" * 60)
    print("VERIFYING DATASET INTEGRITY")
    print("=" * 60)

    # 1. Count PNGs on disk
    disk_files = {p.name: p for p in images_dir.glob("*.png")}
    total_pngs_on_disk = len(disk_files)
    print(f"PNG files on disk ({images_dir}): {total_pngs_on_disk}")

    # Check non-PNG files in images_dir
    all_files = list(images_dir.iterdir())
    non_pngs = [p.name for p in all_files if not p.name.lower().endswith(".png")]
    print(f"Non-PNG files on disk: {len(non_pngs)}")
    if non_pngs:
        print(f"  Non-PNG files: {non_pngs}")

    # 2. Check indiana_projections.csv
    df_proj = pd.read_csv(projections_path)
    total_proj_rows = len(df_proj)
    print(f"\nProjections CSV rows ({projections_path}): {total_proj_rows}")
    csv_filenames = df_proj["filename"].astype(str).str.strip().tolist()
    unique_csv_filenames = set(csv_filenames)
    print(f"Unique filenames in projections CSV: {len(unique_csv_filenames)}")

    # Rows whose image is missing on disk
    missing_on_disk = [fn for fn in csv_filenames if fn not in disk_files]
    missing_on_disk_unique = set(missing_on_disk)
    print(f"\nCSV rows whose image is missing on disk: {len(missing_on_disk)}")
    print(f"Unique filenames missing on disk: {len(missing_on_disk_unique)}")
    if missing_on_disk_unique:
        for fn in sorted(missing_on_disk_unique):
            print(f"  - Missing: {fn}")

    # Images on disk not referenced in CSV
    disk_filenames = set(disk_files.keys())
    unreferenced_on_disk = disk_filenames - unique_csv_filenames
    print(f"\nImages on disk NOT referenced in CSV: {len(unreferenced_on_disk)}")
    if unreferenced_on_disk:
        for fn in sorted(unreferenced_on_disk):
            print(f"  - Unreferenced: {fn}")

    # 3. Check for unreadable / corrupt images using multi-threading
    print("\nVerifying PIL readability for all disk images (multi-threaded)...")
    corrupt_images = []
    unreadable_errors = {}

    max_workers = min(32, (os.cpu_count() or 4) * 2)
    file_list = list(disk_files.values())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_image, p): p for p in file_list}
        for future in tqdm(as_completed(futures), total=len(file_list), desc="Checking"):
            fname, is_valid, err = future.result()
            if not is_valid:
                corrupt_images.append(fname)
                unreadable_errors[fname] = err

    print(f"\nUnreadable / Corrupt images found: {len(corrupt_images)}")
    if corrupt_images:
        for fname in sorted(corrupt_images):
            print(f"  - Corrupt: {fname} -> {unreadable_errors[fname]}")
    else:
        print("  None. All images on disk are 100% readable and valid PIL images.")

    # 4. Reports CSV summary
    df_repo = pd.read_csv(reports_path)
    print(f"\nReports CSV rows: {len(df_repo)}")
    print(f"Unique uids in projections: {df_proj['uid'].nunique()}")
    print(f"Unique uids in reports: {df_repo['uid'].nunique()}")

    return {
        "total_pngs_on_disk": total_pngs_on_disk,
        "total_proj_rows": total_proj_rows,
        "missing_on_disk_rows": len(missing_on_disk),
        "unreferenced_on_disk": len(unreferenced_on_disk),
        "corrupt_images": len(corrupt_images),
    }


if __name__ == "__main__":
    verify_dataset()
