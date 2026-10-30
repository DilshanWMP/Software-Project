# lap-predictions/predictions/src/dataset.py
"""Albumentations-backed ImageFolder dataset for laparoscopy classification."""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder

IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=25, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.1),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 24), hole_width_range=(8, 24), p=0.2),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

eval_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


class AlbImageFolder(ImageFolder):
    """ImageFolder that applies Albumentations transforms instead of torchvision."""

    def __init__(self, root, transform):
        super().__init__(root)
        self.tf = transform

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as img:
            img_rgb = np.array(img.convert("RGB"))
        return self.tf(image=img_rgb)["image"], target
