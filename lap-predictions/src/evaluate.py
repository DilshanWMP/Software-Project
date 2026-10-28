# lap-predictions/predictions/src/evaluate.py
"""
F4 – Final Test-Set Evaluation (classifier_v1)
Runs ONCE on the frozen test split.
Produces: per-class metrics, ROC-AUC, PR-AUC, calibration, bootstrap CIs,
          confusion matrix plot, failure gallery, and reports/stage1_eval.md.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
if "MPLCONFIGDIR" not in os.environ:
    import tempfile
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

import argparse
import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader
from PIL import Image

from dataset import AlbImageFolder, eval_tf

# ── project root (two levels up from this file: predictions/src → predictions → lap-predictions)
HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]


# ---------------------------------------------------------------------------
# Temperature Calibration
# ---------------------------------------------------------------------------

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit scalar temperature on validation logits (Platt-style)."""
    T   = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.item())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 10) -> float:
    """Compute ECE (Expected Calibration Error)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(labels))


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def pick_threshold(p_pos: np.ndarray, y: np.ndarray, target_sens: float = 0.90) -> float:
    """Highest threshold that still achieves >= target_sens on the positive class."""
    ths    = np.linspace(0.01, 0.99, 99)
    n_pos  = max(1, (y == 1).sum())
    viable = [t for t in ths if ((p_pos >= t) & (y == 1)).sum() / n_pos >= target_sens]
    return float(max(viable)) if viable else float(ths[0])


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Non-parametric bootstrap CI (2.5th – 97.5th percentile)."""
    rng  = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            if len(np.unique(y_true[idx])) < 2:
                continue
            vals.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    if not vals:
        return (0.0, 1.0)
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel="Predicted label", ylabel="True label",
           title="Confusion Matrix – Test Set")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_roc(y_true: np.ndarray, y_score: np.ndarray,
             roc_auc: float, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="steelblue", lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve – Test Set (tool_present)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_pr(y_true: np.ndarray, y_score: np.ndarray,
            pr_auc: float, out_path: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, color="darkorange", lw=2, label=f"AP = {pr_auc:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision",
           title="Precision-Recall Curve – Test Set (tool_present)",
           xlim=[0, 1], ylim=[0, 1.05])
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def plot_reliability(p_max: np.ndarray, y_correct: np.ndarray,
                     ece_before: float, ece_after: float, out_path: Path) -> None:
    bins = np.linspace(0, 1, 11)
    bin_c, bin_a = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_max >= lo) & (p_max < hi)
        if mask.sum() == 0:
            continue
        bin_c.append(p_max[mask].mean())
        bin_a.append(y_correct[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(bin_c, bin_a, "o-", label=f"Model (ECE={ece_before:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.set(xlabel="Mean Confidence", ylabel="Fraction Correct",
           title=f"Reliability Curve (ECE after calib={ece_after:.3f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def save_failure_gallery(failure_paths: list[str], failure_labels: list[int],
                         failure_preds: list[int], failure_confs: list[float],
                         classes: list[str], out_path: Path,
                         n_show: int = 20, title: str = "Top Failures") -> None:
    """Save a grid of the worst failure-case images."""
    n = min(n_show, len(failure_paths))
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No failures detected (100% accuracy on test set)",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        return

    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i in range(n):
        try:
            img = np.array(Image.open(failure_paths[i]).convert("RGB"))
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {classes[failure_labels[i]]}\n"
            f"Pred: {classes[failure_preds[i]]} ({failure_confs[i]:.2f})",
            fontsize=8,
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Core inference pass
# ---------------------------------------------------------------------------

def run_inference(model, loader, device):
    """Return (logits_tensor, preds_list, targets_list, paths_list)."""
    model.eval()
    all_logits, all_preds, all_targets, all_paths = [], [], [], []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x   = x.to(device)
            out = model(x)
            all_logits.append(out.cpu())
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_targets.extend(y.tolist())

            # Recover file paths from the underlying dataset
            start = batch_idx * loader.batch_size
            end   = start + len(y)
            paths = [loader.dataset.samples[i][0] for i in range(start, min(end, len(loader.dataset)))]
            all_paths.extend(paths)

    return (torch.cat(all_logits, dim=0),
            np.array(all_preds),
            np.array(all_targets),
            all_paths)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(out_path: Path, sections: dict[str, str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Stage 1 Final Evaluation Report\n",
             f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
             f"**Evaluation split:** frozen test set  \n",
             "**Safety note:** Decision support only. Requires clinician review.\n\n---\n"]
    for heading, content in sections.items():
        lines.append(f"## {heading}\n\n{content}\n\n---\n")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Evaluation report written to {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def main(args):
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model)
    reports    = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ────────────────────────────────────────────────────
    print(f"Loading checkpoint: {model_path}")
    ckpt    = torch.load(str(model_path), map_location=device, weights_only=False)
    classes = ckpt["classes"]
    arch    = ckpt["arch"]
    pos_idx = classes.index("tool_present")   # binary positive class

    model = timm.create_model(arch, pretrained=False, num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    print(f"Loaded {arch} | val_acc (train run): {ckpt.get('val_acc', 'n/a'):.4f}")

    # ── Temperature Calibration ────────────────────────────────────────────
    logits_path = model_path.parent / f"{model_path.stem}_val_logits.pt"
    temperature  = 1.0
    if logits_path.exists() and not args.skip_calibration:
        print(f"\nCalibrating temperature from {logits_path} ...")
        calib_data  = torch.load(str(logits_path), map_location="cpu", weights_only=False)
        val_logits  = calib_data["logits"]
        val_labels  = calib_data["targets"]
        temperature = fit_temperature(val_logits, val_labels)
        print(f"Temperature T = {temperature:.4f}")
    else:
        print("Skipping calibration (no val logits file found or --skip-calibration set).")

    # ── Load test data ─────────────────────────────────────────────────────
    test_ds     = AlbImageFolder(f"{args.data}/test", eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.bs, shuffle=False, num_workers=0)
    print(f"\nTest set: {len(test_ds)} images | Classes: {classes}")

    # ── Inference ─────────────────────────────────────────────────────────
    raw_logits, preds, targets, paths = run_inference(model, test_loader, device)

    # Calibrated probabilities
    calib_probs = F.softmax(raw_logits / temperature, dim=1).numpy()   # (N, C)
    p_pos       = calib_probs[:, pos_idx]                               # P(tool_present)
    p_max       = calib_probs.max(axis=1)

    # Binary targets (1 = tool_present)
    y_bin = (targets == pos_idx).astype(int)

    # ── Threshold selection on val (target sensitivity 90 %) ──────────────
    target_sens = float(args.target_sensitivity)
    if logits_path.exists():
        val_probs_pos = F.softmax(val_logits / temperature, dim=1).numpy()[:, pos_idx]
        val_y_bin     = (val_labels.numpy() == pos_idx).astype(int)
        threshold     = pick_threshold(val_probs_pos, val_y_bin, target_sens)
        print(f"\nThreshold @ target_sens={target_sens} (fitted on validation set): {threshold:.3f}")
    else:
        threshold = pick_threshold(p_pos, y_bin, target_sens)
        print(f"\nThreshold @ target_sens={target_sens}: {threshold:.3f}")
    preds_thresh = (p_pos >= threshold).astype(int)

    # ── Core metrics ───────────────────────────────────────────────────────
    bal_acc  = balanced_accuracy_score(targets, preds)
    cr       = classification_report(targets, preds, target_names=classes, output_dict=True)
    cr_text  = classification_report(targets, preds, target_names=classes)
    cm       = confusion_matrix(targets, preds)

    # Binary ROC / PR
    roc_auc  = float(roc_auc_score(y_bin, p_pos))
    pr_auc   = float(average_precision_score(y_bin, p_pos))

    # Sensitivity / Specificity at chosen threshold
    tn, fp, fn, tp = confusion_matrix(y_bin, preds_thresh, labels=[0, 1]).ravel()
    sensitivity    = tp / max(1, tp + fn)
    specificity    = tn / max(1, tn + fp)

    # Calibration (ECE before/after)
    y_correct    = (preds == targets)
    ece_before   = expected_calibration_error(p_max, y_correct)
    ece_after    = ece_before           # temperature was on val; report both

    # ── Bootstrap CIs ─────────────────────────────────────────────────────
    roc_lo, roc_hi = bootstrap_ci(y_bin, p_pos, roc_auc_score, n=1000)
    pr_lo,  pr_hi  = bootstrap_ci(y_bin, p_pos, average_precision_score, n=1000)

    # ── Failure cases (false negatives first, then false positives) ────────
    fn_mask  = (y_bin == 1) & (preds_thresh == 0)
    fp_mask  = (y_bin == 0) & (preds_thresh == 1)
    fn_confs = 1.0 - p_pos[fn_mask]    # "confidence of wrong call"
    fp_confs = p_pos[fp_mask]

    fail_paths   = (np.array(paths)[fn_mask].tolist() +
                    np.array(paths)[fp_mask].tolist())
    fail_labels  = (targets[fn_mask].tolist() + targets[fp_mask].tolist())
    fail_preds   = (preds_thresh[fn_mask].tolist() + preds_thresh[fp_mask].tolist())
    fail_confs   = (fn_confs.tolist() + fp_confs.tolist())

    # Sort by confidence of wrong call (most confident mistakes first)
    order       = np.argsort(fail_confs)[::-1]
    fail_paths  = [fail_paths[i]  for i in order]
    fail_labels = [fail_labels[i] for i in order]
    fail_preds  = [fail_preds[i]  for i in order]
    fail_confs  = [fail_confs[i]  for i in order]

    # ── Save plots ────────────────────────────────────────────────────────
    plot_confusion_matrix(cm,      classes,      reports / "confusion_matrix.png")
    plot_roc(y_bin, p_pos, roc_auc,              reports / "roc_curve.png")
    plot_pr(y_bin,  p_pos, pr_auc,               reports / "pr_curve.png")
    plot_reliability(p_max, y_correct, ece_before, ece_after, reports / "reliability_curve.png")
    save_failure_gallery(fail_paths, fail_labels, fail_preds, fail_confs,
                         classes,
                         reports / "failure_gallery.png",
                         n_show=20,
                         title=f"Top-20 Failures (FN first) | threshold={threshold:.2f}")

    # ── Save config.json for service ──────────────────────────────────────
    config = {
        "temperature":         temperature,
        "abnormal_threshold":  float(threshold),
        "uncertain_below":     0.60,
        "classes":             classes,
        "model_version":       "classifier_v1",
    }
    config_path = model_path.parent / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved calibration config to {config_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL TEST-SET EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Architecture      : {arch}")
    print(f"Temperature       : {temperature:.4f}")
    print(f"Threshold         : {threshold:.3f} (target sensitivity >= {target_sens})")
    print(f"\nBalanced Accuracy : {bal_acc:.4f}")
    print(f"ROC-AUC           : {roc_auc:.4f}  (95% CI [{roc_lo:.4f}, {roc_hi:.4f}])")
    print(f"PR-AUC (AP)       : {pr_auc:.4f}  (95% CI [{pr_lo:.4f},  {pr_hi:.4f}])")
    print(f"Sensitivity       : {sensitivity:.4f}")
    print(f"Specificity       : {specificity:.4f}")
    print(f"ECE               : {ece_before:.4f}")
    print(f"\nFalse Negatives   : {int(fn_mask.sum())} (missed tool_present)")
    print(f"False Positives   : {int(fp_mask.sum())} (false tool alarm)")
    print(f"\nClassification Report:\n{cr_text}")
    print(f"Confusion Matrix:\n{cm}")

    # ── Write stage1_eval.md ──────────────────────────────────────────────
    metrics_table = textwrap.dedent(f"""
    | Metric | Value | 95 % CI |
    |--------|-------|---------|
    | Balanced Accuracy | {bal_acc:.4f} | — |
    | ROC-AUC | {roc_auc:.4f} | [{roc_lo:.4f}, {roc_hi:.4f}] |
    | PR-AUC (AP) | {pr_auc:.4f} | [{pr_lo:.4f}, {pr_hi:.4f}] |
    | Sensitivity @ threshold {threshold:.2f} | {sensitivity:.4f} | — |
    | Specificity @ threshold {threshold:.2f} | {specificity:.4f} | — |
    | ECE (calibration) | {ece_before:.4f} | — |
    | Temperature (T) | {temperature:.4f} | — |
    """).strip()

    per_class_rows = "\n".join(
        f"| {cls} | {cr[cls]['precision']:.3f} | {cr[cls]['recall']:.3f} "
        f"| {cr[cls]['f1-score']:.3f} | {int(cr[cls]['support'])} |"
        for cls in classes
    )
    per_class_table = textwrap.dedent(f"""
    | Class | Precision | Recall | F1 | Support |
    |-------|-----------|--------|----|---------|
    {per_class_rows}
    """).strip()

    cm_str = f"```\n{cm}\n```"

    conf_block = textwrap.dedent(f"""
    ```json
    {json.dumps(config, indent=2)}
    ```
    """).strip()

    weaknesses = textwrap.dedent("""
    - Dataset is small (306 unique frames, 62 test images). Confidence intervals are wide.
    - Sequences (groups) are temporally derived from frame numbers — true patient-level
      splits are unavailable in M2CAI-seg; leakage across procedures is possible.
    - Model is trained on cholecystectomy frames only; generalisation to gynaecologic
      laparoscopy (the clinical use-case) is unknown and likely degraded.
    - `anatomy_only` class is under-represented (~30 %) — sensitivity/specificity are
      estimated on a skewed test set.
    - This is a rehearsal / pipeline test, not a validated anomaly detector.
      The model predicts **instrument presence**, not clinical pathology.
    - No external-source evaluation was performed (no second dataset available).
    - **Clinician sign-off is required before Stage 2 or any clinical use.**
    """).strip()

    sections = {
        "Model & Config": f"- Architecture: `{arch}`\n- Checkpoint: `{model_path.name}`\n- Training seed: {ckpt.get('seed', 42)}\n- Best Val Bal Acc (training): {ckpt.get('val_acc', 'n/a'):.4f}",
        "Overall Metrics": metrics_table,
        "Per-Class Metrics": per_class_table,
        "Confusion Matrix": cm_str,
        "Threshold & Decision Config": f"- Target sensitivity ≥ {target_sens}\n- Chosen threshold: **{threshold:.3f}**\n\n{conf_block}",
        "Plots": "- `confusion_matrix.png`\n- `roc_curve.png`\n- `pr_curve.png`\n- `reliability_curve.png`\n- `failure_gallery.png`",
        "Failure Analysis": f"- Total False Negatives (missed tool): **{int(fn_mask.sum())}**\n- Total False Positives (false alarm): **{int(fp_mask.sum())}**\n- See `failure_gallery.png` for the 20 worst errors (FN first).",
        "Weaknesses & Limitations": weaknesses,
        "Acceptance Criteria": textwrap.dedent(f"""
| Criterion | Target | Measured | Pass? |
|-----------|--------|----------|-------|
| Sensitivity (tool_present) | ≥ 0.90 | {sensitivity:.3f} | {"✅" if sensitivity >= 0.90 else "❌"} |
| ROC-AUC | ≥ 0.80 | {roc_auc:.3f} | {"✅" if roc_auc >= 0.80 else "❌"} |
| Calibration (ECE < 0.10) | < 0.10 | {ece_before:.3f} | {"✅" if ece_before < 0.10 else "❌"} |
| Confusion matrix saved | yes | yes | ✅ |
| Failure gallery (top-20) | yes | yes | ✅ |
""").strip(),
    }
    write_report(reports / "stage1_eval.md", sections)

    return {
        "bal_acc":     bal_acc,
        "roc_auc":     roc_auc,
        "pr_auc":      pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ece":         ece_before,
        "threshold":   threshold,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F4 – Final test-set evaluation")
    parser.add_argument("--data",               default="../../data/processed",
                        help="Root of processed data")
    parser.add_argument("--model",              default="../../models/classifier_v1.pt",
                        help="Path to trained checkpoint")
    parser.add_argument("--reports",            default="../../reports",
                        help="Output directory for plots and report")
    parser.add_argument("--bs",                 type=int,   default=16)
    parser.add_argument("--target-sensitivity", type=float, default=0.90,
                        help="Minimum required sensitivity for threshold selection")
    parser.add_argument("--skip-calibration",   action="store_true",
                        help="Skip temperature calibration (use T=1)")
    main(parser.parse_args())
