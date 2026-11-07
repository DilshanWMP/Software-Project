"""Shared constants and helpers for the ml/ pipeline."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1]          # <repo>/ml
REPO_DIR = ML_DIR.parent
DATA_DIR = ML_DIR / "data"
SPLITS_DIR = ML_DIR / "splits"
CONFIG_DIR = ML_DIR / "config"
RUNS_DIR = ML_DIR / "runs"
CKPT_DIR = ML_DIR / "checkpoints"
REPORTS_DIR = ML_DIR / "reports"
GOLDEN_DIR = ML_DIR / "golden"
AIMODELS_DIR = REPO_DIR / "EndoscopyApp" / "AiModels"
INTERNAL_DIR = REPO_DIR / "internal"

SEED = 42

# Super-class keys from the contract, in the canonical order. A trained model
# uses the subset that survives the data review (see ml/config/class_map.json).
SUPER_CLASSES = {
    "no_abnormal_finding": {"display": "No abnormal finding", "abnormal": False},
    "inflammation": {"display": "Inflammation", "abnormal": True},
    "ulcer": {"display": "Ulcer", "abnormal": True},
    "bleeding": {"display": "Bleeding / blood", "abnormal": True},
    "polyp": {"display": "Polyp", "abnormal": True},
    "barretts": {"display": "Barrett's oesophagus", "abnormal": True},
    "varices": {"display": "Varices", "abnormal": True},
    "possible_tumour": {"display": "Possible tumour", "abnormal": True},
    "other_procedure": {"display": "Procedure-related (dye, resection, tools)", "abnormal": False},
}

PREPROCESS_BLOCK = {
    "border_crop": True,
    "border_gray_threshold": 20,
    "border_min_fraction": 0.02,
    "border_min_keep": 0.25,
    "resize": "bilinear_stretch",
}

DEFAULT_QUALITY = {
    "min_side_px": 128,
    "dark_mean_below": 25,
    "bright_mean_above": 235,
    "blur_laplacian_var_below": 15.0,
    "analysis_long_side_px": 512,
}

GATE_CLASSES = ["not_endoscopy", "endoscopy"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_data_path(rel: str) -> Path:
    """Split CSVs store paths relative to ml/data/."""
    p = Path(rel)
    return p if p.is_absolute() else DATA_DIR / p
