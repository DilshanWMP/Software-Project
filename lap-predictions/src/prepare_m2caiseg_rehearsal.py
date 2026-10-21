# lap-predictions/src/prepare_m2caiseg_rehearsal.py
import re
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAW_DIR = Path("data/raw/m2caiseg/trainval")
IMAGES_DIR = RAW_DIR / "images"
MASKS_DIR = RAW_DIR / "groundtruth"
PROCESSED_DIR = Path("data/processed")

def has_tool_mask(mask_path: Path) -> bool:
    """Detect tool presence: m2caiSeg tool masks use bright cyan/green/magenta hues."""
    mask = cv2.imread(str(mask_path))
    if mask is None:
        return False
    # Check for instrument marker colors (channels where G==255 and R==0, or bright saturated components)
    tool_pixels = np.any((mask[:, :, 1] == 255) & (mask[:, :, 2] == 0))
    return bool(tool_pixels)

def main():
    rows = []
    image_files = sorted(list(IMAGES_DIR.glob("*.jpg")))
    print(f"Found {len(image_files)} images in {IMAGES_DIR}")

    for img_path in image_files:
        stem = img_path.stem
        mask_path = MASKS_DIR / f"{stem}_gt.png"
        if not mask_path.exists():
            continue

        # Extract frame index for temporal grouping (group frames into 4000-frame chunks)
        frame_match = re.search(r"(\d+)", stem)
        frame_num = int(frame_match.group(1)) if frame_match else 0
        group_id = f"seq_{frame_num // 4000}"

        is_tool = has_tool_mask(mask_path)
        label = "tool_present" if is_tool else "anatomy_only"

        rows.append({
            "filepath": str(img_path),
            "filename": img_path.name,
            "group_id": group_id,
            "label": label,
            "frame_num": frame_num
        })

    df = pd.DataFrame(rows)
    print(f"Total matched pairs: {len(df)}")
    print("Class distribution:\n", df["label"].value_counts())

    # Stratified Group Split (approx 70% train / 15% val / 15% test)
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    folds = [test_idx for _, test_idx in sgkf.split(df, df["label"], groups=df["group_id"])]

    df["split"] = "train"
    df.loc[folds[0], "split"] = "test"
    df.loc[folds[1], "split"] = "val"

    # Automated Leakage Assertion
    leakage = df.groupby("group_id")["split"].nunique()
    assert (leakage == 1).all(), "Leakage detected! Video sequences cross splits."
    print("✅ Leakage check PASSED: 0 group overlap between train, val, and test.")

    print("\nSplit counts:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))

    # Clean existing processed directory and copy files
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for r in df.itertuples():
        dest_dir = PROCESSED_DIR / r.split / r.label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.filepath, dest_dir / r.filename)

    manifest_path = Path("data/split_manifest.csv")
    df.to_csv(manifest_path, index=False)
    print(f"\nSaved split manifest to {manifest_path}")

if __name__ == "__main__":
    main()
