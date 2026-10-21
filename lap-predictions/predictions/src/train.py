# lap-predictions/src/train.py
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score
import timm

from dataset import AlbImageFolder, train_tf, eval_tf

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = AlbImageFolder(f"{args.data}/train", train_tf)
    val_ds = AlbImageFolder(f"{args.data}/val", eval_tf)
    classes = train_ds.classes
    print(f"Classes: {classes} | Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Balanced class sampling
    targets = np.array([y for _, y in train_ds.samples])
    class_counts = np.bincount(targets)
    weights = (1.0 / class_counts)[targets]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=args.bs, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    model = timm.create_model(args.arch, pretrained=True, num_classes=len(classes), drop_rate=0.2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def epoch_step(loader, optimizer=None):
        is_train = optimizer is not None
        model.train(is_train)
        losses, preds, targets_all = [], [], []

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.set_grad_enabled(is_train):
                out = model(x)
                loss = criterion(out, y)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            losses.append(loss.item())
            preds.extend(out.argmax(dim=1).cpu().tolist())
            targets_all.extend(y.cpu().tolist())

        mean_loss = float(np.mean(losses))
        bal_acc = float(balanced_accuracy_score(targets_all, preds))
        return mean_loss, bal_acc

    # Step A: Freeze backbone, train classifier head only
    print("\n--- Step A: Training Classifier Head ---")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    opt_head = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for ep in range(args.head_epochs):
        trl, tra = epoch_step(train_loader, opt_head)
        vll, vla = epoch_step(val_loader)
        print(f"[Head {ep+1}/{args.head_epochs}] Train Loss: {trl:.4f}, BalAcc: {tra:.3f} | Val Loss: {vll:.4f}, BalAcc: {vla:.3f}")

    # Step B: Unfreeze full network, fine-tune at low LR with Cosine Annealing
    print("\n--- Step B: Fine-Tuning Full Network ---")
    for param in model.parameters():
        param.requires_grad = True

    opt_full = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=args.epochs)

    best_val_acc = -1.0
    patience_count = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        trl, tra = epoch_step(train_loader, opt_full)
        vll, vla = epoch_step(val_loader)
        scheduler.step()

        print(f"[Epoch {ep+1:02d}/{args.epochs}] Train: loss={trl:.4f}, acc={tra:.3f} | Val: loss={vll:.4f}, acc={vla:.3f}")

        if vla > best_val_acc:
            best_val_acc = vla
            patience_count = 0
            torch.save({
                "state": model.state_dict(),
                "classes": classes,
                "arch": args.arch,
                "val_acc": best_val_acc,
                "epoch": ep + 1
            }, str(out_path))
            print(f"  --> Saved new best checkpoint to {out_path} (Val BalAcc: {best_val_acc:.3f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping triggered at epoch {ep+1}.")
                break

    print(f"\nTraining Complete. Best Validation Balanced Accuracy: {best_val_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--arch", default="efficientnet_b0")
    parser.add_argument("--out", default="models/classifier_v1.pt")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
