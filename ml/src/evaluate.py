"""Calibrate on val and evaluate on the frozen test split.

Findings:
    ml\\.venv\\Scripts\\python.exe ml\\src\\evaluate.py --checkpoint ml\\checkpoints\\b0_384\\best.pt
Gate:
    ml\\.venv\\Scripts\\python.exe ml\\src\\evaluate.py --checkpoint ml\\checkpoints\\gate_mnv3\\best.pt

Writes ml/reports/<run>/report.json (+ PNGs) and internal/ML_EVAL_<run>.md.
export_onnx.py reads the "calibration" and "quality_suggested" blocks from report.json.

Methods (findings)
- Temperature: one scalar T minimising val NLL (LBFGS on log T).
- Per-class threshold (the "AboveThreshold" flag in the app), chosen on val with the
  calibrated probabilities:
    abnormal classes: the largest t whose one-vs-rest val sensitivity (p_c >= t) is >= 0.90
                      (screening bias: missing a finding is worse than a false flag)
    other classes:    t maximising one-vs-rest val F1
  clipped to [0.05, 0.90].
- uncertain_below: on val, the smallest u in 0.30..0.85 such that top-1 accuracy over
  images with max-prob >= u reaches 0.95 while keeping >= 70 % coverage; if that is not
  reachable, the u that keeps 80 % coverage. Clipped to [0.40, 0.80].
- Quality thresholds: 1st / 99th percentiles of the train-crop statistics (capped by the
  contract defaults) so that ~1 % of good dataset images would be flagged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment import stress_variant  # noqa: E402
from common import (DEFAULT_QUALITY, INTERNAL_DIR, REPO_DIR, REPORTS_DIR, SPLITS_DIR, SUPER_CLASSES,  # noqa: E402
                    resolve_data_path, save_json, seed_everything)
from datasets import ImageCsvDataset  # noqa: E402
from models import Normalized, build_model  # noqa: E402
from preprocess import load_and_crop, to_tensor  # noqa: E402


def stress_transform(crop, i):
    return stress_variant(crop, seed=10_000 + i)


# ------------------------------------------------------------------ helpers
def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == labels[m]).mean() - conf[m].mean())
    return float(e)


def per_class(y, pred, probs, classes):
    out = {}
    for c, k in enumerate(classes):
        pos = y == c
        tp = int(((pred == c) & pos).sum())
        fn = int(((pred != c) & pos).sum())
        fp = int(((pred == c) & ~pos).sum())
        tn = int(((pred != c) & ~pos).sum())
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        prec = tp / max(1, tp + fp)
        f1 = 2 * prec * sens / max(1e-9, prec + sens)
        try:
            auc = float(roc_auc_score(pos, probs[:, c])) if 0 < pos.sum() < len(y) else None
        except ValueError:
            auc = None
        out[k] = {"support": int(pos.sum()), "sensitivity": sens, "specificity": spec,
                  "precision": prec, "f1": f1, "auc": auc}
    return out


def metrics(y, probs, classes):
    pred = probs.argmax(1)
    n = len(classes)
    sens = [((pred == c) & (y == c)).sum() / max(1, (y == c).sum()) for c in range(n) if (y == c).any()]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n)), zero_division=0)),
        "balanced_accuracy": float(np.mean(sens)),
        "ece": ece(probs, y),
        "per_class": per_class(y, pred, probs, classes),
        "confusion": confusion_matrix(y, pred, labels=list(range(n))).tolist(),
    }


def plot_cm(cm, classes, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(1, cm.sum(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 3, 1.0 * len(classes) + 2.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def choose_thresholds(y, probs, classes, target_sens=0.90):
    out = {}
    grid = np.round(np.arange(0.05, 0.901, 0.01), 2)
    for c, k in enumerate(classes):
        pos = y == c
        if not pos.any():
            out[k] = 0.5
            continue
        p = probs[:, c]
        if SUPER_CLASSES[k]["abnormal"]:
            ok = [t for t in grid if (p[pos] >= t).mean() >= target_sens]
            out[k] = float(max(ok)) if ok else 0.05
        else:
            f1s = []
            for t in grid:
                pr = p >= t
                tp = (pr & pos).sum()
                f1s.append(2 * tp / max(1, pr.sum() + pos.sum()))
            out[k] = float(grid[int(np.argmax(f1s))])
    return out


def choose_uncertain(y, probs, target_acc=0.95, min_cov=0.70):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    table = []
    for u in np.round(np.arange(0.30, 0.851, 0.01), 2):
        m = conf >= u
        table.append((float(u), float(m.mean()), float(correct[m].mean()) if m.any() else 1.0))
    ok = [u for u, cov, acc in table if acc >= target_acc and cov >= min_cov]
    if ok:
        u = ok[0]
        rule = f"smallest u with val selective accuracy >= {target_acc} and coverage >= {min_cov}"
    else:
        u = float(np.quantile(conf, 0.20))
        rule = "target accuracy not reachable with >=70% coverage; u keeps 80% val coverage"
    u = float(np.clip(round(u, 2), 0.40, 0.80))
    return u, rule, table


def selective(y, probs, u):
    conf = probs.max(1)
    m = conf >= u
    correct = probs.argmax(1) == y
    return {"uncertain_below": u, "coverage": float(m.mean()),
            "accuracy_confident": float(correct[m].mean()) if m.any() else None,
            "accuracy_uncertain": float(correct[~m].mean()) if (~m).any() else None}


def quality_stats(df: pd.DataFrame, n: int = 3000, seed: int = 42):
    sub = df.sample(min(n, len(df)), random_state=seed)
    means, laps, sides = [], [], []
    L = DEFAULT_QUALITY["analysis_long_side_px"]
    for p in sub["path"]:
        crop, _ = load_and_crop(resolve_data_path(p))
        h, w = crop.shape[:2]
        sides.append(min(h, w))
        g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        s = L / max(h, w)
        g = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        means.append(float(g.mean()))
        laps.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    means, laps, sides = map(np.asarray, (means, laps, sides))
    pct = lambda a: {f"p{q}": float(np.percentile(a, q)) for q in (0.5, 1, 5, 50, 95, 99, 99.5)}
    stats = {"n": int(len(sub)), "mean_brightness": pct(means), "laplacian_var": pct(laps),
             "min_side": pct(sides)}
    q = dict(DEFAULT_QUALITY)
    q["dark_mean_below"] = int(min(DEFAULT_QUALITY["dark_mean_below"], np.floor(np.percentile(means, 1))))
    q["bright_mean_above"] = int(max(DEFAULT_QUALITY["bright_mean_above"], np.ceil(np.percentile(means, 99))))
    q["blur_laplacian_var_below"] = float(round(min(DEFAULT_QUALITY["blur_laplacian_var_below"],
                                                    np.percentile(laps, 1)), 1))
    for k, a, cond in [("dark", means, means < q["dark_mean_below"]),
                       ("bright", means, means > q["bright_mean_above"]),
                       ("blur", laps, laps < q["blur_laplacian_var_below"])]:
        stats[f"flagged_{k}_fraction"] = float(cond.mean())
    return stats, q


@torch.no_grad()
def infer(net, df, classes, size, device, bs, workers, transform=None):
    ds = ImageCsvDataset(df, classes, size, train=False, transform=transform)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
    out, ys = [], []
    for x, y in dl:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(net(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(out), torch.cat(ys)


def md_metrics_table(m, classes):
    L = ["| class | support | sensitivity | specificity | precision | F1 | AUC |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for k in classes:
        r = m["per_class"][k]
        auc = f"{r['auc']:.3f}" if r["auc"] is not None else "-"
        L.append(f"| {k} | {r['support']} | {r['sensitivity']:.3f} | {r['specificity']:.3f} | "
                 f"{r['precision']:.3f} | {r['f1']:.3f} | {auc} |")
    return L


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke: limit rows per split")
    ap.add_argument("--no-doc", action="store_true")
    args = ap.parse_args()
    seed_everything()

    ck = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    run = args.run_name or args.checkpoint.parent.name
    task, classes, size = ck.get("task", "findings"), ck["classes"], ck["img_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ck["arch"], len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    net = Normalized(model).to(device).eval()
    prefix = "gate_" if task == "gate" else ""
    dfs = {}
    for sp in ["train", "val", "test"]:
        d = pd.read_csv(args.splits_dir / f"{prefix}{sp}.csv")
        d = d[d["label"].isin(classes)].reset_index(drop=True)
        if args.limit and len(d) > args.limit:
            per = max(2, args.limit // len(classes))
            d = pd.concat([g.sample(min(len(g), per), random_state=42) for _, g in d.groupby("label")])
            d = d.reset_index(drop=True)
        dfs[sp] = d
    out_dir = REPORTS_DIR / run
    out_dir.mkdir(parents=True, exist_ok=True)
    kw = dict(classes=classes, size=size, device=device, bs=args.batch_size, workers=args.workers)

    print(f"[eval] {run}: task={task} val={len(dfs['val'])} test={len(dfs['test'])}")
    lv, yv = infer(net, dfs["val"], **kw)
    lt, yt = infer(net, dfs["test"], **kw)
    ls, ys = infer(net, dfs["test"], transform=stress_transform, **kw)
    T = fit_temperature(lv, yv)
    print(f"[eval] temperature {T:.4f}")
    yv_np, yt_np, ys_np = yv.numpy(), yt.numpy(), ys.numpy()
    pv = F.softmax(lv / T, 1).numpy()
    pt_raw = F.softmax(lt, 1).numpy()
    pt = F.softmax(lt / T, 1).numpy()
    ps = F.softmax(ls / T, 1).numpy()

    report = {"run": run, "task": task, "arch": ck["arch"], "img_size": size, "classes": classes,
              "checkpoint": str(args.checkpoint), "best_epoch": ck.get("epoch"),
              "smoke_limit": args.limit or None}

    if task == "findings":
        thr = choose_thresholds(yv_np, pv, classes)
        u, rule, table = choose_uncertain(yv_np, pv)
        report["calibration"] = {"temperature": T, "uncertain_below": u, "thresholds": thr,
                                 "uncertain_rule": rule}
        report["val"] = metrics(yv_np, pv, classes)
        report["test"] = metrics(yt_np, pt, classes)
        report["test"]["ece_before_temperature"] = ece(pt_raw, yt_np)
        report["test"]["selective"] = selective(yt_np, pt, u)
        report["test_stress"] = metrics(ys_np, ps, classes)
        report["test_stress"]["selective"] = selective(ys_np, ps, u)
        report["val_uncertain_table"] = table
        print("[eval] quality stats on train crops ...")
        qs, qsug = quality_stats(dfs["train"], n=300 if args.limit else 3000)
        report["quality_stats"] = qs
        report["quality_suggested"] = qsug
        plot_cm(report["test"]["confusion"], classes, out_dir / "confusion_test.png", f"{run} test")
        plot_cm(report["test_stress"]["confusion"], classes, out_dir / "confusion_test_stress.png",
                f"{run} test (capture-look stress)")
        save_json(report, out_dir / "report.json")
        m = report["test"]
        print(f"[eval] TEST macroF1 {m['macro_f1']:.4f} balAcc {m['balanced_accuracy']:.4f} "
              f"acc {m['accuracy']:.4f} ECE {m['ece_before_temperature']:.4f}->{m['ece']:.4f} | "
              f"stress macroF1 {report['test_stress']['macro_f1']:.4f}")
        if not args.no_doc:
            write_findings_doc(report, out_dir)
    else:
        report.update(gate_eval(net, T, yv_np, pv, yt_np, pt, ys_np, ps, dfs, size, device))
        save_json(report, out_dir / "report.json")
        t = report["test"]
        print(f"[eval] GATE test: neg rejected {t['negatives_rejected']:.4f}, "
              f"pos accepted {t['positives_accepted']:.4f}; stress {report['test_stress']}")
        if not args.no_doc:
            write_gate_doc(report)
    print(f"[eval] report: {out_dir / 'report.json'}")


def gate_eval(net, T, yv, pv, yt, pt, ys, ps, dfs, size, device, min_accept=0.99):
    # Threshold on val: the largest t in [0.30, 0.90] that still accepts >= 99 % of val
    # endoscopy images (false rejection of real frames is the costly error), which then
    # rejects as many non-endoscopy images as possible.
    pe_v = pv[:, 1]
    grid = np.round(np.arange(0.30, 0.901, 0.01), 2)
    ok = [t for t in grid if (yv == 1).any() and (pe_v[yv == 1] >= t).mean() >= min_accept]
    thr = float(max(ok)) if ok else 0.30

    def rates(y, p):
        pe = p[:, 1]
        return {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
                "positives_accepted": float((pe[y == 1] >= thr).mean()) if (y == 1).any() else None,
                "negatives_rejected": float((pe[y == 0] < thr).mean()) if (y == 0).any() else None,
                "auc": float(roc_auc_score(y, pe)) if 0 < y.sum() < len(y) else None}

    res = {"calibration": {"temperature": T, "endoscopy_threshold": thr,
                           "threshold_rule": f"largest t accepting >= {min_accept} of val endoscopy"},
           "val": rates(yv, pv), "test": rates(yt, pt), "test_stress": rates(ys, ps)}
    # Sanity check on the user's own snapshots (not training data).
    media = REPO_DIR / "EndoscopyApp" / "bin" / "Debug" / "net9.0-windows" / "Media"
    snaps = []
    for p in sorted(media.rglob("IMG_*.jpg"))[:200]:
        try:
            crop, _ = load_and_crop(p)
        except ValueError:
            continue
        x = torch.from_numpy(to_tensor(crop, size)).to(device)
        with torch.no_grad():
            pe = F.softmax(net(x).float() / T, 1)[0, 1].item()
        snaps.append({"file": p.relative_to(media).as_posix(), "p_endo": round(pe, 4),
                      "accepted": pe >= thr})
    res["media_snapshots"] = snaps
    return res


def write_findings_doc(r, out_dir):
    classes = r["classes"]
    m, s, c = r["test"], r["test_stress"], r["calibration"]
    L = [f"# ML_EVAL_{r['run']}.md: findings model evaluation", "",
         f"Arch `{r['arch']}` @ {r['img_size']}, best epoch {r['best_epoch']}. "
         f"Checkpoint `{r['checkpoint']}`." + (" **SMOKE TEST (limited data) - numbers meaningless.**"
                                              if r.get("smoke_limit") else ""), "",
         "Decision support only. Requires clinician review.", "",
         "## Frozen test set", "",
         f"n = {m['n']}; accuracy {m['accuracy']:.3f}; **macro-F1 {m['macro_f1']:.3f}**; "
         f"balanced accuracy {m['balanced_accuracy']:.3f}; ECE {m['ece_before_temperature']:.3f} -> "
         f"{m['ece']:.3f} after temperature {c['temperature']:.3f}.", ""]
    L += md_metrics_table(m, classes)
    sel = m["selective"]
    L += ["", f"Uncertain rule (u = {c['uncertain_below']}; {c['uncertain_rule']}): test coverage "
          f"{sel['coverage']:.3f}, accuracy when confident {sel['accuracy_confident']}, "
          f"when uncertain {sel['accuracy_uncertain']}.", "",
          "Confusion matrix: `" + str((out_dir / 'confusion_test.png').relative_to(REPO_DIR)) + "`", "",
          "## Capture-look stress test (same test images, synthetic black border + overlay text + "
          "colour shift + JPEG, then the contract crop)", "",
          f"macro-F1 {s['macro_f1']:.3f}; balanced accuracy {s['balanced_accuracy']:.3f}; "
          f"accuracy {s['accuracy']:.3f}.", ""]
    L += md_metrics_table(s, classes)
    L += ["", "## Calibration written to the json", "",
          f"- temperature: {c['temperature']:.4f}", f"- uncertain_below: {c['uncertain_below']}",
          "- thresholds: " + ", ".join(f"{k} {v}" for k, v in c["thresholds"].items()), "",
          "## Quality thresholds (train crop statistics)", "",
          "```json", __import__("json").dumps(r["quality_stats"], indent=1), "```", "",
          "Suggested `quality` block: `" + __import__("json").dumps(r["quality_suggested"]) + "`", ""]
    (INTERNAL_DIR / f"ML_EVAL_{r['run']}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[doc] wrote internal/ML_EVAL_{r['run']}.md")


def write_gate_doc(r):
    import json
    L = [f"# ML_EVAL_{r['run']}.md: endoscopy gate evaluation", "",
         f"Arch `{r['arch']}` @ {r['img_size']}; temperature {r['calibration']['temperature']:.3f}; "
         f"threshold {r['calibration']['endoscopy_threshold']}." +
         (" **SMOKE TEST.**" if r.get("smoke_limit") else ""), "",
         "| split | n pos | n neg | positives accepted | negatives rejected | AUC |", "|---|---:|---:|---:|---:|---:|"]
    for k in ["val", "test", "test_stress"]:
        t = r[k]
        L.append(f"| {k} | {t['n_pos']} | {t['n_neg']} | {t['positives_accepted']} | "
                 f"{t['negatives_rejected']} | {t['auc']} |")
    L += ["", "## User Media snapshots (sanity check, not training data)", "", "```",
          json.dumps(r["media_snapshots"], indent=0), "```", ""]
    (INTERNAL_DIR / f"ML_EVAL_{r['run']}.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[doc] wrote internal/ML_EVAL_{r['run']}.md")


if __name__ == "__main__":
    main()
