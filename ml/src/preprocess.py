"""Reference preprocessing for the GI findings / endoscopy-gate models.

This file is the Python half of internal/AI_CONTRACT.md section 2. The C# engine
must produce the same crop box and (within 1e-3) the same probabilities, so keep
this simple and do not "improve" it without changing the contract.

    crop_rgb, (x0, y0, x1, y1) = load_and_crop(path)
    x = to_tensor(crop_rgb, 384)          # float32 [1, 3, 384, 384], RGB 0..1

The box is inclusive: crop = img[y0:y1 + 1, x0:x1 + 1].
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Defaults = contract json "preprocess" block.
BORDER_GRAY_THRESHOLD = 20
BORDER_MIN_FRACTION = 0.02
BORDER_MIN_KEEP = 0.25
# Contract section 2 step 1: decode as BGR, ignore EXIF orientation.
IMREAD_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def crop_box(bgr: np.ndarray,
             gray_threshold: int = BORDER_GRAY_THRESHOLD,
             min_fraction: float = BORDER_MIN_FRACTION,
             min_keep: float = BORDER_MIN_KEEP) -> tuple[int, int, int, int]:
    """Return the inclusive (x0, y0, x1, y1) border-crop box for a BGR uint8 image."""
    h, w = bgr.shape[:2]
    full = (0, 0, w - 1, h - 1)
    if bgr.ndim == 2:
        gray = bgr
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = gray > gray_threshold
    row_frac = mask.mean(axis=1)
    col_frac = mask.mean(axis=0)
    rows = np.flatnonzero(row_frac > min_fraction)
    cols = np.flatnonzero(col_frac > min_fraction)
    if rows.size == 0 or cols.size == 0:
        return full
    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    if (x1 - x0 + 1) < min_keep * w or (y1 - y0 + 1) < min_keep * h:
        return full
    return (x0, y0, x1, y1)


def crop_array(bgr: np.ndarray, **kw) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Border-crop a BGR uint8 array. Returns (rgb_uint8_crop, (x0, y0, x1, y1))."""
    x0, y0, x1, y1 = crop_box(bgr, **kw)
    crop = bgr[y0:y1 + 1, x0:x1 + 1]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb), (x0, y0, x1, y1)


def imread_bgr(path: str | Path) -> np.ndarray:
    """cv2.imread(IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION), as in the contract.

    EXIF orientation is deliberately ignored (WPF shows images unrotated, so the
    heatmap must align with the raw pixel grid). Falls back to imdecode (same flags)
    only for non-ASCII Windows paths that imread cannot open.
    """
    img = cv2.imread(str(path), IMREAD_FLAGS)
    if img is None:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, IMREAD_FLAGS) if data.size else None
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def load_and_crop(path: str | Path, **kw) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Decode an image file as BGR and border-crop it (contract section 2, steps 1-3)."""
    return crop_array(imread_bgr(path), **kw)


def resize(rgb: np.ndarray, size: int) -> np.ndarray:
    """Stretch-resize to size x size with INTER_LINEAR (contract step 4)."""
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)


def to_tensor(crop_rgb: np.ndarray, size: int) -> np.ndarray:
    """Contract steps 4-5: resize, /255, HWC->CHW, batch dim. Returns float32 [1,3,S,S]."""
    x = resize(crop_rgb, size).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def preprocess_file(path: str | Path, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    crop, box = load_and_crop(path)
    return to_tensor(crop, size), box


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        c, b = load_and_crop(p)
        print(p, "box", b, "crop", c.shape)
