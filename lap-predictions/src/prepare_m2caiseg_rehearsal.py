# lap-predictions/src/prepare_m2caiseg_rehearsal.py
"""Prepare M2CAI-seg rehearsal data with EXIF stripping, blur filtering, and group-split."""

from pathlib import Path
from src.prepare_dataset import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Preparing M2CAI-seg rehearsal dataset...")
    prepare_dataset(
        raw_dir=PROJECT_ROOT / "data/raw/m2caiseg/trainval",
        clean_dir=PROJECT_ROOT / "data/clean",
        processed_dir=PROJECT_ROOT / "data/processed",
        reports_dir=PROJECT_ROOT / "reports",
        manifest_path=PROJECT_ROOT / "data/split_manifest.csv",
        crop_thr=15,
        blur_cutoff=10.0,
        n_splits=7,
        seed=42,
    )

if __name__ == "__main__":
    main()

