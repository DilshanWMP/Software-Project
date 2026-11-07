"""Golden files for the C# parity test (internal/AI_CONTRACT.md section 6).

    ml\\.venv\\Scripts\\python.exe ml\\src\\make_golden.py            # uses EndoscopyApp/AiModels

Builds ml/golden/images/*.jpg (small) and ml/golden/cases.json from the exported ONNX
files (onnxruntime CPU), so the C# side compares against exactly what ships.

Image set (8-12):
- up to --n-test frozen-test images, one per class, HyperKvasir only (CC BY 4.0, OK to commit),
  otherwise the GI snapshots in the app's Media folder (IMG_20260502_*);
- 3 non-endoscopy photos (2 Media webcam snapshots IMG_20261106_*, topped up with COCO val2017);
- 1 GI image placed on a black octagon canvas with overlay text (border-crop case);
- 1 almost-black frame (no-crop fallback case).
Images are re-encoded as JPEG (max side 512); the crop box is computed from the saved file.
Re-run after every model export: the probabilities change, the images stay.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import capture_look  # noqa: E402
from common import AIMODELS_DIR, DATA_DIR, GOLDEN_DIR, REPO_DIR, SPLITS_DIR, load_json, save_json  # noqa: E402
from preprocess import imread_bgr, load_and_crop, to_tensor  # noqa: E402

MEDIA = REPO_DIR / "EndoscopyApp" / "bin" / "Debug" / "net9.0-windows" / "Media"


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def shrink(bgr, max_side=512):
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def save_jpg(bgr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])


def coco_images(n):
    zp = DATA_DIR / "raw" / "coco-val2017.zip"
    out = []
    if not zp.exists():
        return out
    with zipfile.ZipFile(zp) as z:
        names = sorted(m for m in z.namelist() if m.lower().endswith(".jpg"))
        for m in random.Random(42).sample(names, min(n, len(names))):
            buf = np.frombuffer(z.read(m), np.uint8)
            out.append((Path(m).stem, cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)))
    return out


def collect(n_test: int):
    """Returns list of (name, bgr)."""
    items = []
    gi_sources = []
    test_csv = SPLITS_DIR / "test.csv"
    if test_csv.exists():
        import pandas as pd
        df = pd.read_csv(test_csv)
        # Only HyperKvasir images (CC BY 4.0) are committed; GastroVision is research-only.
        df = df[df["source"] == "hyperkvasir"].sort_values("path")
        picked = []
        for lab in df["label"].unique():   # one per class
            sub = df[df.label == lab]
            picked.append(sub.iloc[len(sub) // 2])
        for r in picked[:n_test]:
            p = DATA_DIR / r["path"]
            gi_sources.append((f"test_{r['label']}", imread_bgr(p)))
    if not gi_sources:
        for p in sorted(MEDIA.rglob("IMG_20260502_*.jpg"))[:n_test]:
            gi_sources.append((f"gi_{p.stem}", imread_bgr(p)))
    items += [(n, shrink(b)) for n, b in gi_sources]

    nonendo = [(f"nonendo_{p.stem}", imread_bgr(p)) for p in sorted(MEDIA.rglob("IMG_20261106_*.jpg"))[:2]]
    nonendo += [(f"nonendo_coco_{n}", b) for n, b in coco_images(3 - len(nonendo))]
    items += [(n, shrink(b)) for n, b in nonendo]

    if gi_sources:   # black-bordered capture-look case
        base_rgb, _ = load_and_crop_arr(gi_sources[0][1])
        rng = random.Random(7)
        frame = None
        for _ in range(20):
            f = capture_look(base_rgb, rng)
            if f.shape[1] > base_rgb.shape[1] * 1.15:
                frame = f
                break
        frame = frame if frame is not None else capture_look(base_rgb, rng)
        items.append(("bordered_capture", shrink(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 640)))

    dark = np.full((240, 320, 3), 8, np.uint8)       # almost black; tiny bright blob -> no crop
    cv2.circle(dark, (160, 120), 12, (90, 90, 90), -1)
    items.append(("dark_frame", dark))
    return items


def load_and_crop_arr(bgr):
    from preprocess import crop_array
    return crop_array(bgr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=AIMODELS_DIR)
    ap.add_argument("--out-dir", type=Path, default=GOLDEN_DIR)
    ap.add_argument("--n-test", type=int, default=6)
    args = ap.parse_args()
    import onnxruntime as ort

    fj = load_json(args.models_dir / "gi_findings.json")
    gj = load_json(args.models_dir / "endo_gate.json")
    fs = ort.InferenceSession(str(args.models_dir / "gi_findings.onnx"), providers=["CPUExecutionProvider"])
    gs = ort.InferenceSession(str(args.models_dir / "endo_gate.onnx"), providers=["CPUExecutionProvider"])
    keys = [c["key"] for c in sorted(fj["classes"], key=lambda c: c["index"])]

    img_dir = args.out_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)
    cases = []
    for name, bgr in collect(args.n_test):
        fn = f"{name}.jpg"
        save_jpg(bgr, img_dir / fn)
        crop, box = load_and_crop(img_dir / fn)          # from the saved file, like C#
        lf = fs.run(["logits"], {"input": to_tensor(crop, fj["input_size"])})[0][0]
        lg = gs.run(["logits"], {"input": to_tensor(crop, gj["input_size"])})[0][0]
        probs = softmax(lf.astype(np.float64) / fj["temperature"])
        p_endo = float(softmax(lg.astype(np.float64) / gj["temperature"])[1])
        cases.append({"image": fn, "crop": list(box), "findings_probs": [round(float(p), 6) for p in probs],
                      "gate_p_endo": round(p_endo, 6), "top_key": keys[int(np.argmax(probs))]})
        h, w = bgr.shape[:2]
        print(f"{fn:40s} {w}x{h} crop {box} top {cases[-1]['top_key']} {probs.max():.3f} p_endo {p_endo:.3f}")
    with open(args.out_dir / "cases.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=1)
        f.write("\n")
    save_json({"findings_model_version": fj["model_version"], "gate_model_version": gj["model_version"],
               "note": "cases.json probabilities come from these ONNX files via onnxruntime CPU; "
                       "regenerate with ml/src/make_golden.py after every export."},
              args.out_dir / "models.json")
    print(f"wrote {len(cases)} cases to {args.out_dir}")


if __name__ == "__main__":
    main()
