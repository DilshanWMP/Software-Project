# lap-predictions/predictions/src/train.py
"""
F3 – Baseline Classifier Training (classifier_v1)
Two-step transfer learning: freeze backbone → train head, then unfreeze → full fine-tune.
Logs to TensorBoard; saves best checkpoint by validation balanced accuracy.
Saves val logits for calibration (F4).
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score
import timm

from dataset import AlbImageFolder, train_tf, eval_tf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(data_root: str, bs: int):
    train_ds = AlbImageFolder(f"{data_root}/train", train_tf)
    val_ds   = AlbImageFolder(f"{data_root}/val",   eval_tf)
    classes  = train_ds.classes

    targets = np.array([y for _, y in train_ds.samples])
    counts  = np.bincount(targets)
    weights = (1.0 / counts)[targets]
    sampler = WeightedRandomSampler(weights.tolist(), len(weights))

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=False)
    return train_loader, val_loader, classes


def epoch_step(model, loader, criterion, device, optimizer=None):
    """Run one epoch; if optimizer is None, runs in eval mode."""
    is_train = optimizer is not None
    model.train(is_train)
    losses, preds, targets, all_logits = [], [], [], []

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(y.cpu().tolist())
            all_logits.append(logits.detach().cpu())

    mean_loss = float(np.mean(losses))
    bal_acc   = float(balanced_accuracy_score(targets, preds))
    logits_t  = torch.cat(all_logits, dim=0)
    return mean_loss, bal_acc, logits_t, targets


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(args):
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = make_loaders(args.data, args.bs)
    print(f"Classes: {classes} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # TensorBoard writer
    if args.tb_dir:
        tb_dir = Path(args.tb_dir)
    else:
        models_dir = Path(args.out).resolve().parent
        tb_dir = models_dir.parent / "runs" / Path(args.out).stem
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer  = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard log dir: {tb_dir}")

    model     = timm.create_model(args.arch, pretrained=True,
                                  num_classes=len(classes), drop_rate=0.25).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # -----------------------------------------------------------------------
    # Step A: Train classifier head only (backbone frozen)
    # -----------------------------------------------------------------------
    print("\n--- Step A: Training Classifier Head (backbone frozen) ---")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True

    opt_head = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)

    for ep in range(args.head_epochs):
        trl, tra, _, _ = epoch_step(model, train_loader, criterion, device, opt_head)
        vll, vla, _, _ = epoch_step(model, val_loader,   criterion, device)
        print(f"  [Head {ep+1:02d}/{args.head_epochs}] "
              f"Train loss={trl:.4f} bacc={tra:.3f} | Val loss={vll:.4f} bacc={vla:.3f}")
        writer.add_scalars("head/loss",    {"train": trl, "val": vll}, ep)
        writer.add_scalars("head/bal_acc", {"train": tra, "val": vla}, ep)

    # -----------------------------------------------------------------------
    # Step B: Fine-tune full network at low LR
    # -----------------------------------------------------------------------
    print("\n--- Step B: Full Fine-Tuning ---")
    for p in model.parameters():
        p.requires_grad = True

    opt_full  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=args.epochs)

    best_val_acc   = -1.0
    patience_count = 0
    best_val_logits  = None
    best_val_targets = None

    for ep in range(args.epochs):
        trl, tra, _, _          = epoch_step(model, train_loader, criterion, device, opt_full)
        vll, vla, vlogits, vtgt = epoch_step(model, val_loader,   criterion, device)
        scheduler.step()

        print(f"  [Epoch {ep+1:02d}/{args.epochs}] "
              f"Train loss={trl:.4f} bacc={tra:.3f} | Val loss={vll:.4f} bacc={vla:.3f}")
        writer.add_scalars("full/loss",    {"train": trl, "val": vll}, ep)
        writer.add_scalars("full/bal_acc", {"train": tra, "val": vla}, ep)
        writer.add_scalar("full/lr", scheduler.get_last_lr()[0], ep)

        gap = tra - vla
        writer.add_scalar("full/generalisation_gap", gap, ep)

        if vla > best_val_acc:
            best_val_acc     = vla
            patience_count   = 0
            best_val_logits  = vlogits
            best_val_targets = vtgt
            torch.save({
                "state":   model.state_dict(),
                "classes": classes,
                "arch":    args.arch,
                "val_acc": best_val_acc,
                "epoch":   ep + 1,
                "seed":    args.seed,
            }, str(out_path))
            classes_path = out_path.parent / "classes.json"
            with open(classes_path, "w", encoding="utf-8") as f:
                json.dump(classes, f, indent=2)
            print(f"    --> Best checkpoint saved (Val BalAcc: {best_val_acc:.4f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {ep+1}.")
                break

    writer.close()

    # Save val logits for temperature calibration in F4
    if best_val_logits is not None:
        logits_path = out_path.parent / f"{out_path.stem}_val_logits.pt"
        torch.save({
            "logits":  best_val_logits,
            "targets": torch.tensor(best_val_targets, dtype=torch.long),
            "classes": classes,
        }, str(logits_path))
        print(f"Val logits saved to {logits_path} (for calibration)")

    print(f"\nTraining complete. Best Val BalAcc: {best_val_acc:.4f}")
    return best_val_acc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train laparoscopy classifier (F3)")
    parser.add_argument("--data",        default="../../data/processed",
                        help="Root of processed data (contains train/val/test subfolders)")
    parser.add_argument("--arch",        default="efficientnet_b0",
                        help="timm model architecture")
    parser.add_argument("--out",         default="../../models/classifier_v1.pt",
                        help="Output checkpoint path")
    parser.add_argument("--tb-dir",     default=None,
                        help="TensorBoard log directory (default: runs/<model_stem>)")
    parser.add_argument("--bs",          type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Full fine-tune learning rate")
    parser.add_argument("--head_epochs", type=int,   default=5,
                        help="Head-only warm-up epochs")
    parser.add_argument("--epochs",      type=int,   default=25,
                        help="Full fine-tune epochs")
    parser.add_argument("--patience",    type=int,   default=6,
                        help="Early stopping patience")
    parser.add_argument("--seed",        type=int,   default=42)
    main(parser.parse_args())
