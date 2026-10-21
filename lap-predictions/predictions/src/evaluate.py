# lap-predictions/src/evaluate.py
import argparse
import numpy as np
import torch
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

from dataset import AlbImageFolder, eval_tf

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.model, map_location=device)
    classes = checkpoint["classes"]
    arch = checkpoint["arch"]

    model = timm.create_model(arch, pretrained=False, num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["state"])
    model.eval()

    val_ds = AlbImageFolder(f"{args.data}/val", eval_tf)
    loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False)

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(dim=1).cpu().tolist())
            targets.extend(y.tolist())

    print("\n=== Validation Evaluation Summary ===")
    print(f"Model Arch         : {arch}")
    print(f"Balanced Accuracy  : {balanced_accuracy_score(targets, preds):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, preds))
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--model", default="models/classifier_v1.pt")
    parser.add_argument("--bs", type=int, default=16)
    main(parser.parse_args())
