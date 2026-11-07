"""Training augmentation (numpy / OpenCV only, no extra deps).

Order for a training sample (findings and gate):
    raw image --contract crop--> crop
      [p_capture] capture-look: put the crop on a black canvas with an octagon /
                  circle / rounded-rect mask, add overlay text / timestamps
                  --contract crop again--> crop      (train == inference path)
      mild random-resized-crop + stretch resize to S
      flips / 90 degree rotations
      colour: brightness, contrast, saturation, hue, white balance
      blur, JPEG re-compression
    -> float tensor RGB 0..1 (normalisation is inside the model)
"""
from __future__ import annotations

import random

import cv2
import numpy as np

from preprocess import crop_array

_FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
          cv2.FONT_HERSHEY_COMPLEX_SMALL]
_WORDS = ["OLYMPUS", "EVIS", "CV-190", "FUJINON", "PENTAX", "WLI", "NBI", "BLI", "LCI",
          "Ex", "Eh", "A1", "Zoom", "x1.5", "CVP", "Name:", "ID:", "Sex: M", "Age: 54",
          "Comment:", "Freeze", "Rec", "HD", "1080i", "Enh A5", "Iris: Auto", "DR"]


def _rand_text(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.35:
        return (f"{rng.randint(2010, 2027)}/{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d} "
                f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}")
    if kind < 0.55:
        return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
    if kind < 0.7:
        return f"ID {rng.randint(100000, 99999999)}"
    return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 3)))


def _draw_text(img: np.ndarray, rng: random.Random, n: int, regions: list[tuple[int, int, int, int]]):
    h, w = img.shape[:2]
    for _ in range(n):
        x0, y0, x1, y1 = rng.choice(regions)
        if x1 - x0 < 20 or y1 - y0 < 12:
            continue
        scale = rng.uniform(0.35, 1.0) * max(h, w) / 900
        thick = 1 if scale < 0.8 else 2
        txt = _rand_text(rng)
        col = rng.choice([(255, 255, 255), (230, 230, 230), (0, 255, 0), (0, 255, 255),
                          (255, 255, 0), (200, 200, 255)])
        (tw, th), _ = cv2.getTextSize(txt, _FONTS[0], scale, thick)
        x = rng.randint(x0, max(x0, x1 - tw))
        y = rng.randint(min(y0 + th, y1), y1)
        cv2.putText(img, txt, (x, y), rng.choice(_FONTS), scale, col, thick, cv2.LINE_AA)


def capture_look(rgb: np.ndarray, rng: random.Random, text_prob: float = 0.8) -> np.ndarray:
    """Simulate an HDMI capture frame: crop placed on black canvas with a scope-shaped
    mask and on-screen text. Returns an RGB uint8 frame (NOT cropped)."""
    h, w = rgb.shape[:2]
    # Canvas: 16:9-ish or 4:3-ish, image placed off-centre (usually right of the text panel).
    ar = rng.choice([16 / 9, 4 / 3, 5 / 4])
    ch = int(h * rng.uniform(1.0, 1.25))
    cw = max(int(ch * ar), w + 4)
    canvas = np.zeros((ch, cw, 3), np.uint8)
    canvas[:] = rng.randint(0, 12)                   # not always pure black
    ox = rng.randint(0, cw - w)
    oy = rng.randint(0, ch - h)

    # Mask the image itself (scope shape).
    mask = np.zeros((h, w), np.uint8)
    shape = rng.choice(["octagon", "circle", "rounded", "rect"])
    if shape == "circle":
        r = int(min(h, w) / 2 * rng.uniform(1.0, 1.3))
        cv2.circle(mask, (w // 2, h // 2), r, 255, -1)
    elif shape == "octagon":
        c = rng.uniform(0.08, 0.25)
        cx, cy = int(w * c), int(h * c)
        pts = np.array([[cx, 0], [w - 1 - cx, 0], [w - 1, cy], [w - 1, h - 1 - cy],
                        [w - 1 - cx, h - 1], [cx, h - 1], [0, h - 1 - cy], [0, cy]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    elif shape == "rounded":
        r = int(min(h, w) * rng.uniform(0.05, 0.2))
        cv2.rectangle(mask, (r, 0), (w - 1 - r, h - 1), 255, -1)
        cv2.rectangle(mask, (0, r), (w - 1, h - 1 - r), 255, -1)
        for cx, cy in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
            cv2.circle(mask, (cx, cy), r, 255, -1)
    else:
        mask[:] = 255
    region = canvas[oy:oy + h, ox:ox + w]
    region[mask > 0] = rgb[mask > 0]

    if rng.random() < text_prob:
        regions = [(0, 0, ox, ch), (ox + w, 0, cw, ch), (0, 0, cw, oy), (0, oy + h, cw, ch)]
        regions = [r for r in regions if r[2] - r[0] > 30 and r[3] - r[1] > 15]
        if regions:
            _draw_text(canvas, rng, rng.randint(1, 6), regions)
        if rng.random() < 0.25:   # occasionally text over the image itself
            _draw_text(canvas, rng, 1, [(ox, oy, ox + w, oy + h)])
    # Capture-card scaling: often to 1920x1080 or 1280x720.
    if rng.random() < 0.5:
        tw = rng.choice([1280, 1920, 1024])
        canvas = cv2.resize(canvas, (tw, int(tw * ch / cw)), interpolation=cv2.INTER_AREA)
    return canvas


def capture_look_then_crop(rgb: np.ndarray, rng: random.Random) -> np.ndarray:
    frame = capture_look(rgb, rng)
    crop, _ = crop_array(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return crop


def random_resized_crop(rgb, rng, size, scale=(0.7, 1.0), ratio=(0.8, 1.25)):
    h, w = rgb.shape[:2]
    area = h * w
    for _ in range(10):
        a = area * rng.uniform(*scale)
        r = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
        cw = int(round(np.sqrt(a * r)))
        chh = int(round(np.sqrt(a / r)))
        if 0 < cw <= w and 0 < chh <= h:
            x = rng.randint(0, w - cw)
            y = rng.randint(0, h - chh)
            rgb = rgb[y:y + chh, x:x + cw]
            break
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)


def color_jitter(rgb, rng, b=0.2, c=0.2, s=0.2, hue_deg=6, wb=0.1):
    x = rgb.astype(np.float32)
    # white balance: per-channel gain
    x *= np.array([rng.uniform(1 - wb, 1 + wb) for _ in range(3)], np.float32)
    x *= rng.uniform(1 - b, 1 + b)
    m = x.mean()
    x = (x - m) * rng.uniform(1 - c, 1 + c) + m
    x = np.clip(x, 0, 255).astype(np.uint8)
    if s > 0 or hue_deg > 0:
        hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue_deg, hue_deg) / 2) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - s, 1 + s), 0, 255)
        x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return x


def jpeg(rgb, rng, qmin=35, qmax=95):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, rng.randint(qmin, qmax)])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else rgb


class TrainAugment:
    def __init__(self, size: int, p_capture: float = 0.4, p_jpeg: float = 0.3,
                 p_blur: float = 0.15, p_lowres: float = 0.15, rot90: bool = True,
                 scale=(0.65, 1.0)):
        self.size = size
        self.p_capture, self.p_jpeg, self.p_blur, self.p_lowres = p_capture, p_jpeg, p_blur, p_lowres
        self.rot90 = rot90
        self.scale = scale

    def __call__(self, crop_rgb: np.ndarray, rng: random.Random) -> np.ndarray:
        x = crop_rgb
        if rng.random() < self.p_capture:
            x = capture_look_then_crop(x, rng)
        x = random_resized_crop(x, rng, self.size, scale=self.scale)
        if rng.random() < 0.5:
            x = x[:, ::-1]
        if rng.random() < 0.5:
            x = x[::-1]
        if self.rot90 and rng.random() < 0.5:
            x = np.rot90(x, rng.randint(1, 3))
        x = np.ascontiguousarray(x)
        x = color_jitter(x, rng)
        if rng.random() < self.p_lowres:   # resolution loss (small source upscaled)
            f = rng.uniform(0.3, 0.7)
            s = max(16, int(self.size * f))
            x = cv2.resize(cv2.resize(x, (s, s), interpolation=cv2.INTER_AREA),
                           (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if rng.random() < self.p_blur:
            x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.4, 1.4))
        if rng.random() < self.p_jpeg:
            x = jpeg(x, rng)
        return x


def stress_variant(crop_rgb: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic capture-look stress version of a test image, already contract-cropped
    (returns the crop that inference would see)."""
    rng = random.Random(seed)
    x = capture_look_then_crop(crop_rgb, rng)
    x = color_jitter(x, rng, b=0.15, c=0.15, s=0.15, hue_deg=4, wb=0.08)
    x = jpeg(x, rng, 50, 90)
    return x
