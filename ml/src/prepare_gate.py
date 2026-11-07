"""Build the endoscopy-gate splits.

    ml\\.venv\\Scripts\\python.exe -I ml\\src\\prepare_gate.py

Positives (label "endoscopy"): every kept image in ml/splits/all_images.csv (including the
"exclude" classes such as BBPS / impacted stool, which are still endoscopy), keeping the
findings split, so no findings-test image is ever seen by the gate in training.
Negatives (label "not_endoscopy"):
- COCO val2017 (5000 everyday photos), random 70/15/15 (seed 42);
- synthetic "screens": solid / gradient / noise backgrounds with text and UI boxes
  (dark menus, black screens with overlay text, white documents).
Hard negatives with black octagon/circle borders + overlay text are made on the fly by
the capture-look augmentation in train.py (p_capture, applied to both classes, so a black
border with text is not a shortcut for "endoscopy"; the border is cropped afterwards).
Writes ml/splits/gate_{train,val,test}.csv (path, source, label, kind).
"""
from __future__ import annotations

import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SEED, SPLITS_DIR  # noqa: E402

COCO_ZIP = DATA_DIR / "raw" / "coco-val2017.zip"
COCO_DIR = DATA_DIR / "extracted" / "coco"
SYN_DIR = DATA_DIR / "extracted" / "synthetic_neg"
N_SYNTH = 400


def extract_coco() -> list[Path]:
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    root = COCO_DIR.resolve()
    with zipfile.ZipFile(COCO_ZIP) as z:
        for m in z.infolist():
            if m.is_dir() or not m.filename.lower().endswith(".jpg"):
                continue
            t = (COCO_DIR / Path(m.filename).name).resolve()
            if not str(t).startswith(str(root)):
                raise RuntimeError("unsafe path")
            if t.exists() and t.stat().st_size == m.file_size:
                continue
            t.write_bytes(z.read(m))
    return sorted(COCO_DIR.glob("*.jpg"))


def make_synthetic(n: int) -> list[Path]:
    SYN_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    out = []
    for i in range(n):
        p = SYN_DIR / f"syn_{i:04d}.jpg"
        out.append(p)
        if p.exists():
            continue
        w, h = rng.choice([(1920, 1080), (1280, 720), (1024, 768), (800, 600)])
        kind = rng.choice(["solid", "gradient", "noise", "doc", "menu"])
        if kind == "solid":
            img = np.full((h, w, 3), [rng.randint(0, 255) for _ in range(3)], np.uint8)
        elif kind == "gradient":
            a = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
            b = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
            t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
            img = np.broadcast_to(a + (b - a) * t, (h, w, 3)).astype(np.uint8).copy()
        elif kind == "noise":
            small = np.random.default_rng(i).integers(0, 255, (h // 16, w // 16, 3), dtype=np.uint8)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        elif kind == "doc":
            img = np.full((h, w, 3), 245, np.uint8)
        else:
            img = np.full((h, w, 3), rng.randint(0, 40), np.uint8)
        for _ in range(rng.randint(0, 6)):     # UI boxes
            x0, y0 = rng.randint(0, w - 50), rng.randint(0, h - 30)
            cv2.rectangle(img, (x0, y0), (x0 + rng.randint(40, w // 2), y0 + rng.randint(20, h // 3)),
                          [rng.randint(0, 255) for _ in range(3)], rng.choice([-1, 2]))
        col = (20, 20, 20) if kind == "doc" else (235, 235, 235)
        for _ in range(rng.randint(1, 25)):    # text lines
            cv2.putText(img, "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghij0123456789:/")
                                     for _ in range(rng.randint(4, 40))),
                        (rng.randint(0, w // 2), rng.randint(20, h - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        rng.uniform(0.4, 1.4), col, rng.choice([1, 2]), cv2.LINE_AA)
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def split_list(paths: list[Path], rng: random.Random) -> dict[str, list[Path]]:
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    a, b = int(0.70 * n), int(0.85 * n)
    return {"train": paths[:a], "val": paths[a:b], "test": paths[b:]}


def main():
    allimg = pd.read_csv(SPLITS_DIR / "all_images.csv")
    coco = extract_coco()
    syn = make_synthetic(N_SYNTH)
    print(f"[gate] endoscopy {len(allimg)}, coco {len(coco)}, synthetic {len(syn)}")
    rng = random.Random(SEED)
    cs, ss = split_list(coco, rng), split_list(syn, rng)
    for sp in ["train", "val", "test"]:
        pos = allimg[allimg.split == sp]
        rows = [{"path": p, "source": s, "label": "endoscopy", "kind": "gi"}
                for p, s in zip(pos.path, pos.source)]
        rows += [{"path": p.relative_to(DATA_DIR).as_posix(), "source": "coco", "label": "not_endoscopy",
                  "kind": "photo"} for p in cs[sp]]
        rows += [{"path": p.relative_to(DATA_DIR).as_posix(), "source": "synthetic", "label": "not_endoscopy",
                  "kind": "screen"} for p in ss[sp]]
        df = pd.DataFrame(rows).sort_values("path")
        df.to_csv(SPLITS_DIR / f"gate_{sp}.csv", index=False, lineterminator="\n")
        print(f"[gate] {sp}: {df.label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
