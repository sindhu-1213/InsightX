"""Image preprocessing pipeline for InsightX.

Resizes chest X-ray images to 224x224 (aspect-ratio preserved with padding, grayscale).
Reads images listed in splits.csv and writes to data/processed/images_224/.
Resumable, shows progress bar, and safely logs corrupt/unreadable images.

Notice:
    Research Demo Only - This tool is not a certified medical device.
    Outputs are AI predictions/findings for research purposes only and
    must not be used to guide clinical decisions.
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("insightx.preprocess")


def resize_with_padding(
    image: Image.Image,
    target_size: tuple[int, int] = (224, 224),
    pad_color: int = 0,
) -> Image.Image:
    """Resize image preserving aspect ratio and pad to target_size in grayscale.

    Args:
        image: Source PIL image.
        target_size: (width, height) desired output dimensions.
        pad_color: Pixel fill value for padding canvas (0 = black).

    Returns:
        Padded grayscale PIL Image with dimensions target_size.
    """
    img = image.convert("L")
    w, h = img.size
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("L", target_size, color=pad_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def preprocess_image_for_onnx(
    image: Image.Image,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[Image.Image, np.ndarray]:
    """Preprocess image for MobileNetV2 ONNX inference.

    Reuses resize_with_padding:
    1. Converts to grayscale.
    2. Resizes with padding preserving aspect ratio (LANCZOS).
    3. Replicates grayscale to 3 channels.
    4. Applies MobileNetV2 preprocess_input: (x / 127.5) - 1.0.

    Returns:
        (canvas_pil, tensor_batch) where tensor_batch has shape (1, 224, 224, 3) float32.
    """
    canvas = resize_with_padding(image, target_size=target_size)
    arr_gray = np.asarray(canvas, dtype=np.float32)
    arr_3ch = np.repeat(arr_gray[:, :, np.newaxis], 3, axis=-1)
    tensor = (arr_3ch / 127.5) - 1.0
    return canvas, tensor[np.newaxis, ...]


def process_single_image(
    filename: str,
    raw_dir: Path,
    output_dir: Path,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[str, str]:
    """Process a single image file.

    Returns:
        (filename, status) where status is 'SKIPPED', 'PROCESSED', or 'ERROR: <msg>'
    """
    out_file = output_dir / filename
    if out_file.is_file() and out_file.stat().st_size > 0:
        return filename, "SKIPPED"

    in_file = raw_dir / filename
    if not in_file.is_file():
        return filename, f"ERROR: File not found on disk at {in_file}"

    try:
        with Image.open(in_file) as img:
            processed = resize_with_padding(img, target_size=target_size)
            processed.save(out_file, format="PNG", optimize=True)
        return filename, "PROCESSED"
    except Exception as exc:
        logger.warning("Failed processing image %s: %s", filename, exc)
        return filename, f"ERROR: {exc}"


def run_preprocessing(
    splits_csv: Path | str = "data/processed/splits.csv",
    raw_dir: Path | str = "data/raw/images_normalized",
    output_dir: Path | str = "data/processed/images_224",
    target_size: tuple[int, int] = (224, 224),
    max_workers: int | None = None,
) -> dict[str, int]:
    """Run full preprocessing pipeline across all files listed in splits.csv.

    Args:
        splits_csv: Path to splits.csv containing the 'filename' column.
        raw_dir: Directory with source images.
        output_dir: Output directory for 224x224 images.
        target_size: Desired image dimensions.
        max_workers: Thread worker pool size.

    Returns:
        Dictionary with counts of processed, skipped, and failed files.
    """
    splits_path = Path(splits_csv)
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    if not splits_path.is_file():
        raise FileNotFoundError(
            f"Splits file not found: {splits_path}. Run insightx.data first to generate splits."
        )

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_path}")

    out_path.mkdir(parents=True, exist_ok=True)

    df_splits = pd.read_csv(splits_path)
    if "filename" not in df_splits.columns:
        raise ValueError(f"'filename' column missing in {splits_path}")

    filenames = df_splits["filename"].astype(str).str.strip().unique().tolist()
    total_images = len(filenames)
    logger.info("Found %d unique images listed in %s", total_images, splits_path)

    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) * 2)

    counts = {"PROCESSED": 0, "SKIPPED": 0, "FAILED": 0}
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image,
                fname,
                raw_path,
                out_path,
                target_size,
            ): fname
            for fname in filenames
        }

        for fut in tqdm(
            as_completed(futures),
            total=total_images,
            desc="Preprocessing images (224x224)",
        ):
            fname, status = fut.result()
            if status == "PROCESSED":
                counts["PROCESSED"] += 1
            elif status == "SKIPPED":
                counts["SKIPPED"] += 1
            else:
                counts["FAILED"] += 1
                errors.append((fname, status))

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Target Directory: {out_path}")
    print(f"  Total Images:     {total_images:,}")
    print(f"  Newly Processed:  {counts['PROCESSED']:,}")
    print(f"  Skipped (Cached): {counts['SKIPPED']:,}")
    print(f"  Failed / Skipped: {counts['FAILED']:,}")
    if errors:
        print("\n  Errors encountered:")
        for fname, err in errors[:10]:
            print(f"    - {fname}: {err}")
    print("=" * 60)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="InsightX Image Preprocessing to 224x224 grayscale PNGs"
    )
    parser.add_argument(
        "--splits", default="data/processed/splits.csv", help="Path to splits CSV"
    )
    parser.add_argument(
        "--raw-dir", default="data/raw/images_normalized", help="Path to raw images dir"
    )
    parser.add_argument(
        "--output-dir", default="data/processed/images_224", help="Path to output dir"
    )
    parser.add_argument("--workers", type=int, default=None, help="Worker threads count")
    args = parser.parse_args()

    run_preprocessing(
        splits_csv=args.splits,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
