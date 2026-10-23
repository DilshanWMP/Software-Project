"""Data preparation, cleaning, EXIF stripping, and leakage-safe patient/group splitting for F2."""

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = PROJECT_ROOT / "data/raw/m2caiseg/trainval"
CLEAN_DEFAULT = PROJECT_ROOT / "data/clean"
PROCESSED_DEFAULT = PROJECT_ROOT / "data/processed"
REPORTS_DEFAULT = PROJECT_ROOT / "reports"
MANIFEST_DEFAULT = PROJECT_ROOT / "data/split_manifest.csv"


def crop_black_borders(img: np.ndarray, threshold: int = 15) -> np.ndarray | None:
    """Crop black endoscopic/laparoscopic mask borders.
    
    Returns cropped image, or None if the image is entirely dark.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > threshold)
    if len(ys) == 0:
        return None
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def compute_blur_score(img: np.ndarray) -> float:
    """Compute the variance of the Laplacian as a measure of image focus/sharpness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_tool_in_mask(mask_path: Path) -> bool:
    """Detect tool presence in M2CAI-seg ground-truth mask (G==255, R==0 in RGB / BGR G==255, R==0)."""
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return False
    # OpenCV loads BGR: channel 1 is G, channel 2 is R
    tool_pixels = (mask[:, :, 1] == 255) & (mask[:, :, 2] == 0)
    return bool(tool_pixels.any())


def derive_group_id(stem: str) -> str:
    """Derive patient/procedure temporal sequence group to prevent leakage."""
    if stem.startswith("0") and len(stem) > 1 and stem != "0":
        return "video_alt_00"
    match = re.search(r"(\d+)", stem)
    frame_num = int(match.group(1)) if match else 0
    return f"video_01_seq_{frame_num // 1500:02d}"


def compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_sample_grids(
    df: pd.DataFrame,
    clean_dir: Path,
    reports_dir: Path,
    n_samples: int = 8,
) -> None:
    """Save visual sample grids for inspection of class label sanity."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(df["label"].unique())

    # Per-class sample grids
    for label in classes:
        subset = df[df["label"] == label].sample(
            min(n_samples, len(df[df["label"] == label])), random_state=42
        )
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        fig.suptitle(f"Sample Grid: {label} (F2 Verification)", fontsize=16, fontweight="bold")
        axes = axes.flatten()

        for idx, (_, row) in enumerate(subset.iterrows()):
            img_path = clean_dir / row["filename"]
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img_rgb)
            axes[idx].set_title(f"{row['image_id']} ({row['split']})\nBlur: {row['blur']:.1f}", fontsize=9)
            axes[idx].axis("off")

        # Hide any unused axes
        for j in range(len(subset), len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        out_path = reports_dir / f"sample_grid_{label}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved class sample grid to {out_path}")

    # Combined overview grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Dataset Sample Overview: tool_present (Top) vs anatomy_only (Bottom)", fontsize=14, fontweight="bold")
    for row_idx, label in enumerate(["tool_present", "anatomy_only"]):
        subset = df[df["label"] == label].sample(4, random_state=42)
        for col_idx, (_, row) in enumerate(subset.iterrows()):
            ax = axes[row_idx, col_idx]
            img_path = clean_dir / row["filename"]
            img = cv2.imread(str(img_path))
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{label}\n{row['image_id']} [{row['split']}]", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    overview_path = reports_dir / "sample_grids_overview.png"
    plt.savefig(overview_path, dpi=150)
    plt.close()
    print(f"Saved combined overview grid to {overview_path}")


def prepare_dataset(
    raw_dir: Path,
    clean_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    manifest_path: Path,
    crop_thr: int = 15,
    blur_cutoff: float = 10.0,
    n_splits: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Execute complete F2 pipeline: clean, EXIF strip, blur filter, group-split, and freeze."""
    images_dir = raw_dir / "images"
    masks_dir = raw_dir / "groundtruth"

    if not images_dir.exists():
        raise FileNotFoundError(f"Raw images directory not found: {images_dir}")

    # Prepare directories
    clean_dir.mkdir(parents=True, exist_ok=True)
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"Discovered {len(image_paths)} raw images in {images_dir}")

    records = []
    dropped_blur = 0
    dropped_dark = 0

    for path in image_paths:
        stem = path.stem
        mask_path = masks_dir / f"{stem}_gt.png"
        if not mask_path.exists():
            continue

        raw_img = cv2.imread(str(path))
        if raw_img is None:
            continue

        # 1. Crop black border
        cropped = crop_black_borders(raw_img, threshold=crop_thr)
        if cropped is None or min(cropped.shape[:2]) < 100:
            dropped_dark += 1
            continue

        # 2. Focus / blur calculation
        blur_val = compute_blur_score(cropped)
        if blur_val < blur_cutoff:
            dropped_blur += 1
            print(f"Dropped blurry image {path.name} (blur={blur_val:.2f} < {blur_cutoff})")
            continue

        # 3. Save cleaned frame (re-saving with OpenCV strips EXIF metadata for privacy)
        clean_filename = f"{stem}.jpg"
        clean_file_path = clean_dir / clean_filename
        cv2.imwrite(str(clean_file_path), cropped)

        # 4. Determine label and patient/group
        has_tool = is_tool_in_mask(mask_path)
        label = "tool_present" if has_tool else "anatomy_only"
        group_id = derive_group_id(stem)
        file_hash = compute_file_sha256(clean_file_path)

        records.append(
            {
                "image_id": stem,
                "filename": clean_filename,
                "clean_path": clean_file_path.relative_to(PROJECT_ROOT).as_posix(),
                "group_id": group_id,
                "label": label,
                "blur": blur_val,
                "image_sha256": file_hash,
                "width": cropped.shape[1],
                "height": cropped.shape[0],
            }
        )

    df = pd.DataFrame(records)
    print(f"\nCleaned and valid frames: {len(df)} (Dropped: {dropped_dark} dark/tiny, {dropped_blur} blurry)")
    print(f"Unique groups / sequences: {df['group_id'].nunique()}")
    print("Class counts before split:\n", df["label"].value_counts())

    # 5. Stratified Group Split (70/15/15 target with fixed seed)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [idx for _, idx in sgkf.split(df, df["label"], groups=df["group_id"])]

    df["split"] = "train"
    df.loc[folds[0], "split"] = "test"
    df.loc[folds[1], "split"] = "val"

    # Automated Leakage Assertion
    leakage = df.groupby("group_id")["split"].nunique()
    assert (leakage == 1).all(), "Leakage detected! Video sequences cross splits."
    print("\n[LEAKAGE CHECK] PASSED: 0 group overlap across train, val, and test.")

    # 6. Organize into data/processed/{split}/{class}/
    for row in df.itertuples():
        dest_dir = processed_dir / row.split / row.label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clean_dir / row.filename, dest_dir / row.filename)

    # 7. Write split manifest and compute test-set freeze checksum
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    manifest_checksum = compute_file_sha256(manifest_path)
    print(f"\nSaved split manifest to {manifest_path}")
    print(f"FROZEN TEST SPLIT CHECKSUM (SHA-256): {manifest_checksum}")

    # 8. Report distributions
    split_dist = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print("\n--- Final Split Distribution (Images) ---")
    print(split_dist)
    print("\n--- Final Split Distribution (Groups) ---")
    print(df.groupby(["split", "label"])["group_id"].nunique().unstack(fill_value=0))

    # 9. Generate visual sample grids
    generate_sample_grids(df, clean_dir, reports_dir)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--clean-dir", type=Path, default=CLEAN_DEFAULT)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DEFAULT)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DEFAULT)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--crop-thr", type=int, default=15)
    parser.add_argument("--blur-cutoff", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        raw_dir=args.raw_dir.resolve(),
        clean_dir=args.clean_dir.resolve(),
        processed_dir=args.processed_dir.resolve(),
        reports_dir=args.reports_dir.resolve(),
        manifest_path=args.manifest_path.resolve(),
        crop_thr=args.crop_thr,
        blur_cutoff=args.blur_cutoff,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
