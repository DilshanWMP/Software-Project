"""Train a findings classifier or the endoscopy gate.

Findings (from repo root, PowerShell):
    ml\\.venv\\Scripts\\python.exe ml\\src\\train.py --arch efficientnet_b0 --img-size 384 ^
        --epochs 30 --batch-size 32 --lr 3e-4 --run-name b0_384

Gate:
    ml\\.venv\\Scripts\\python.exe ml\\src\\train.py --task gate --arch mobilenetv3_small_100 ^
        --img-size 224 --epochs 8 --batch-size 64 --lr 1e-3 --run-name gate_mnv3

Smoke test: add --limit 300 --epochs 1.

Choices
- Optimiser AdamW, linear warmup then cosine decay (per step), AMP (bf16/fp16 autocast).
- Class imbalance: square-root class-balanced sampling (--balance-power 0.5). Each
  sample is drawn with weight n_c^-0.5. Full balancing (power 1) would repeat the
  ~150-image classes ~50x per epoch and overfit them; weighted CE with 1/n_c weights
  on top of natural sampling gives very noisy gradients for the rare classes (a few
  per batch carry most of the loss). Sqrt sampling is the usual middle ground; the
  remaining bias toward big classes is corrected later by per-class thresholds and the
  "uncertain" rule chosen on validation.
- Label smoothing 0.05 (temperature scaling in evaluate.py fixes calibration).
- Early stopping on validation macro-F1 (patience --patience epochs).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import TrainAugment  # noqa: E402
from common import CKPT_DIR, GATE_CLASSES, RUNS_DIR, SPLITS_DIR, load_json, save_json, seed_everything  # noqa: E402
from datasets import EpochSampler, ImageCsvDataset, class_counts, sampler_weights  # noqa: E402
from models import Normalized, build_model  # noqa: E402


def load_split(task: str, name: str, splits_dir: Path) -> pd.DataFrame:
    prefix = "gate_" if task == "gate" else ""
    return pd.read_csv(splits_dir / f"{prefix}{name}.csv")


def get_classes(task: str, splits_dir: Path, train_df: pd.DataFrame) -> list[str]:
    if task == "gate":
        return list(GATE_CLASSES)
    meta = splits_dir / "classes.json"
    if meta.exists():
        return load_json(meta)["classes"]
    return sorted(train_df["label"].unique())


def limit_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified-ish subset: up to n rows, every label represented."""
    if not n or len(df) <= n:
        return df
    per = max(1, n // df["label"].nunique())
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby("label")]
    out = pd.concat(parts)
    if len(out) < n:
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    return out.reset_index(drop=True)


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            out = model(x)
        all_logits.append(out.float().cpu())
        all_y.append(y)
    return torch.cat(all_logits), torch.cat(all_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["findings", "gate"], default="findings")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--balance-power", type=float, default=None,
                    help="sampler weight n_c^-p (0 natural, 0.5 sqrt, 1 balanced); "
                         "default 0.5 findings, 1.0 gate")
    ap.add_argument("--p-capture", type=float, default=None,
                    help="probability of capture-look augmentation (default 0.4 findings, 0.5 gate)")
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: limit train/val rows")
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    tr = load_split(args.task, "train", args.splits_dir)
    va = load_split(args.task, "val", args.splits_dir)
    classes = get_classes(args.task, args.splits_dir, tr)
    tr = tr[tr["label"].isin(classes)].reset_index(drop=True)
    va = va[va["label"].isin(classes)].reset_index(drop=True)
    if args.limit:
        tr = limit_df(tr, args.limit, args.seed)
        va = limit_df(va, max(60, args.limit // 3), args.seed)
    n_cls = len(classes)
    print(f"[train] task={args.task} arch={args.arch} S={args.img_size} classes={classes}")
    print(f"[train] train={len(tr)} val={len(va)} device={device} amp={amp_dtype}")

    if args.balance_power is None:
        args.balance_power = 1.0 if args.task == "gate" else 0.5
    p_capture = args.p_capture if args.p_capture is not None else (0.5 if args.task == "gate" else 0.4)
    aug = TrainAugment(args.img_size, p_capture=p_capture,
                       scale=(0.35, 1.0) if args.task == "gate" else (0.65, 1.0))
    ds_tr = ImageCsvDataset(tr, classes, args.img_size, train=True, augment=aug, seed=args.seed)
    ds_va = ImageCsvDataset(va, classes, args.img_size, train=False)
    counts = class_counts(ds_tr.labels, n_cls)
    print("[train] train counts:", dict(zip(classes, counts.tolist())))

    g = torch.Generator().manual_seed(args.seed)
    sampler = EpochSampler(WeightedRandomSampler(sampler_weights(ds_tr.labels, n_cls, args.balance_power),
                                                 num_samples=len(ds_tr), replacement=True, generator=g),
                           len(ds_tr))
    pw = args.workers > 0
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=pw)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=pw)

    net = Normalized(build_model(args.arch, n_cls, pretrained=True, drop_rate=args.drop_rate)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(dl_tr))
    total = args.epochs * steps_per_epoch
    warm = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    run_dir = RUNS_DIR / args.run_name
    ck_dir = CKPT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg.update(splits_dir=str(args.splits_dir), classes=classes, train_counts=counts.tolist(),
               n_train=len(tr), n_val=len(va), p_capture_used=p_capture)
    save_json(cfg, ck_dir / "config.json")
    log_path = run_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                                "val_bal_acc", "lr", "seconds"])

    best, best_ep, bad = -1.0, -1, 0
    for ep in range(args.epochs):
        t0 = time.time()
        sampler.set_epoch(ep)
        net.train()
        tot, nb = 0.0, 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = crit(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        logits, yv = predict(net, dl_va, device, amp_dtype)
        vloss = nn.functional.cross_entropy(logits, yv).item()
        pred = logits.argmax(1).numpy()
        yv = yv.numpy()
        acc = float((pred == yv).mean())
        mf1 = float(f1_score(yv, pred, average="macro", labels=list(range(n_cls)), zero_division=0))
        bal = float(balanced_accuracy_score(yv, pred))
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {ep + 1}/{args.epochs}] loss {tot / max(1, nb):.4f} val_loss {vloss:.4f} "
              f"acc {acc:.4f} macroF1 {mf1:.4f} balAcc {bal:.4f} lr {lr_now:.2e} {dt:.0f}s", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep + 1, tot / max(1, nb), vloss, acc, mf1, bal, lr_now, round(dt, 1)])
        if mf1 > best:
            best, best_ep, bad = mf1, ep + 1, 0
            torch.save({"task": args.task, "arch": args.arch, "img_size": args.img_size,
                        "classes": classes, "state_dict": net.model.state_dict(),
                        "epoch": ep + 1, "val_macro_f1": mf1, "args": cfg},
                       ck_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train] early stop at epoch {ep + 1} (best {best:.4f} @ {best_ep})")
                break
    save_json({"best_val_macro_f1": best, "best_epoch": best_ep}, ck_dir / "summary.json")
    print(f"[train] done. best val macro-F1 {best:.4f} at epoch {best_ep}. ckpt: {ck_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
