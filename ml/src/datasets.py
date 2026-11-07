"""Datasets for findings and gate training/eval.

CSV columns (ml/splits/*.csv): path (relative to ml/data), source, source_class, label, group
Gate CSVs (ml/splits/gate_*.csv): path, source, label (endoscopy / not_endoscopy), kind
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from augment import TrainAugment
from common import resolve_data_path
from preprocess import load_and_crop, resize


def to_chw_float(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)


class ImageCsvDataset(Dataset):
    """Generic: rows with `path` and `label`; label -> index via `classes`.

    train=True  -> contract crop + TrainAugment (random, seeded per epoch/index)
    train=False -> exactly the contract preprocessing (crop + bilinear stretch)
    """

    def __init__(self, df: pd.DataFrame, classes: list[str], size: int, train: bool,
                 augment: TrainAugment | None = None, seed: int = 42, transform=None):
        self.paths = df["path"].tolist()
        self.labels = [classes.index(l) for l in df["label"]]
        self.size = size
        self.train = train
        self.aug = augment or (TrainAugment(size) if train else None)
        self.seed = seed
        self.transform = transform     # optional fn(crop_rgb, idx) -> crop_rgb (eval stress)

    def __len__(self):
        return len(self.paths)

    def load_crop(self, i):
        crop, _ = load_and_crop(resolve_data_path(self.paths[i]))
        return crop

    def __getitem__(self, k):
        # Training indices may be epoch-encoded (EpochSampler): k = epoch * N + i.
        n = len(self.paths)
        i, epoch = k % n, k // n
        crop = self.load_crop(i)
        if self.train:
            rng = random.Random(self.seed * 1_000_003 + epoch * 100_003 + i)
            img = self.aug(crop, rng)
        else:
            if self.transform is not None:
                crop = self.transform(crop, i)
            img = resize(crop, self.size)
        return to_chw_float(img), self.labels[i]


def class_counts(labels: list[int], n: int) -> np.ndarray:
    return np.bincount(np.asarray(labels), minlength=n).astype(np.int64)


def sampler_weights(labels: list[int], n: int, power: float) -> torch.Tensor:
    """Per-sample weights ~ n_c^-power (power 0 = natural, 1 = fully balanced)."""
    cnt = class_counts(labels, n).astype(np.float64)
    w_c = np.where(cnt > 0, cnt ** -power, 0.0)
    return torch.as_tensor(w_c[np.asarray(labels)], dtype=torch.double)


class EpochSampler(torch.utils.data.Sampler):
    """Wraps a sampler and encodes the epoch into each index (k = epoch * N + i), so
    augmentation randomness changes per epoch even with persistent workers."""

    def __init__(self, base, n: int):
        self.base, self.n, self.epoch = base, n, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        off = self.epoch * self.n
        for i in self.base:
            yield off + int(i)
