from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path("data/raw/m2caiseg")
SPLITS = {"train", "trainval", "test"}
MASK_WORDS = ("mask", "groundtruth", "ground_truth", "label", "annot")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

rows = []
for p in ROOT.rglob("*"):
    if p.suffix.lower() not in EXTS:
        continue
    rel = [x.lower() for x in p.relative_to(ROOT).parts]
    split = next((x for x in rel if x in SPLITS), None)
    kind = "mask" if any(w in part for part in rel[:-1] for w in MASK_WORDS) else "image"
    rows.append({"filepath": str(p), "split_given": split, "kind": kind,
                 "name": p.name, "folder": str(p.parent.relative_to(ROOT))})

df = pd.DataFrame(rows)
df.to_csv("data/m2caiseg_manifest.csv", index=False)

print("== Images/masks per split ==")
print(df.groupby(["split_given", "kind"], dropna=False).size())
print("\n== Files per folder ==")
print(df.groupby("folder").size())
print("\n== Sample file names ==")
print(df["name"].head(10).tolist())

print("\n== Mask pixel values (first 3 masks) ==")
for p in df[df.kind == "mask"].filepath.head(3):
    a = np.array(Image.open(p))
    flat = a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1, 1)
    vals, cnt = np.unique(flat, axis=0, return_counts=True)
    print(p, a.shape, f"{len(vals)} unique values")
    print(vals[:15].tolist())