"""Extract, inventory, map, dedupe, group and split the GI datasets.

    ml\\.venv\\Scripts\\python.exe ml\\src\\prepare_data.py              # everything available
    ml\\.venv\\Scripts\\python.exe ml\\src\\prepare_data.py --sources gastrovision   # one dataset

Steps
1. Extract each complete zip from ml/data/raw/ to ml/data/extracted/<source>/ (size-checked).
2. Scan images; source_class = parent folder name.
3. Perceptual hash (imagehash.phash, 64 bit) of the contract border-cropped image (cached).
4. Map source classes -> super-classes with ml/config/class_map.json (written with the
   proposed defaults if missing). Super-classes with < --min-class images are dropped
   (their images become "exclude": still endoscopy, used as gate positives).
5. Dedupe: union-find over pairs with hamming <= --dup-hamming (any source / class).
   A duplicate cluster whose members disagree on the label is dropped completely;
   otherwise the largest-resolution image is kept.
6. Groups: union-find over pairs with hamming <= --group-hamming among images with the
   same label (components > --max-group re-split at a lower threshold) (frames of the same video/lesion end up together). GastroVision / HyperKvasir
   ship no patient ids in the labelled sets, so this is the leakage control.
7. Split 70/15/15 at group level, stratified per label, seed 42.
8. Write ml/splits/{all_images,train,val,test}.csv, classes.json and
   internal/ML_DATA_INVENTORY.md (with sha256 of test.csv = frozen test set).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import zipfile
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONFIG_DIR, DATA_DIR, INTERNAL_DIR, REPO_DIR, SEED, SPLITS_DIR,  # noqa: E402
                    SUPER_CLASSES, load_json, save_json)

RAW_DIR = DATA_DIR / "raw"
EXTRACT_DIR = DATA_DIR / "extracted"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SOURCES = {
    "hyperkvasir": {"zip": "hyper-kvasir-labeled-images.zip", "size": 3_928_814_344},
    "gastrovision": {"zip": "Gastrovision.zip", "size": 1_793_518_442},
}

# Proposed mapping. Keys are normalised folder names (lower case, only a-z0-9 and '-').
# Values: a super-class key or "exclude". Reasons are in internal/ML_DATA_INVENTORY.md.
DEFAULT_CLASS_MAP = {
    "hyperkvasir": {
        # lower GI landmarks
        "cecum": "no_abnormal_finding",
        "ileum": "no_abnormal_finding",
        "retroflex-rectum": "no_abnormal_finding",
        # lower GI findings
        "hemorrhoids": "exclude",
        "polyps": "polyp",
        "ulcerative-colitis-grade-0-1": "inflammation",
        "ulcerative-colitis-grade-1": "inflammation",
        "ulcerative-colitis-grade-1-2": "inflammation",
        "ulcerative-colitis-grade-2": "inflammation",
        "ulcerative-colitis-grade-2-3": "inflammation",
        "ulcerative-colitis-grade-3": "inflammation",
        # quality of view (bowel prep) -> not a finding
        "bbps-0-1": "exclude",
        "bbps-2-3": "exclude",
        "impacted-stool": "exclude",
        # therapeutic
        "dyed-lifted-polyps": "other_procedure",
        "dyed-resection-margins": "other_procedure",
        # upper GI landmarks
        "pylorus": "no_abnormal_finding",
        "retroflex-stomach": "no_abnormal_finding",
        "z-line": "no_abnormal_finding",
        # upper GI findings
        "barretts": "barretts",
        "barretts-short-segment": "barretts",
        "esophagitis-a": "inflammation",
        "esophagitis-b-d": "inflammation",
    },
    "gastrovision": {
        "accessory-tools": "other_procedure",
        "angiectasia": "bleeding",
        "barrett-s-esophagus": "barretts",
        "blood-in-lumen": "bleeding",
        "cecum": "no_abnormal_finding",
        "colon-diverticula": "exclude",
        "colon-polyps": "polyp",
        "colorectal-cancer": "possible_tumour",
        "duodenal-bulb": "no_abnormal_finding",
        "dyed-lifted-polyps": "other_procedure",
        "dyed-resection-margins": "other_procedure",
        "erythema": "inflammation",
        "esophageal-varices": "varices",
        "esophagitis": "inflammation",
        "gastric-polyps": "polyp",
        "gastroesophageal-junction-normal-z-line": "no_abnormal_finding",
        "ileocecal-valve": "no_abnormal_finding",
        "mucosal-inflammation-large-bowel": "inflammation",
        "normal-esophagus": "no_abnormal_finding",
        "normal-mucosa-and-vascular-pattern-in-the-large-bowel": "no_abnormal_finding",
        "normal-stomach": "no_abnormal_finding",
        "pylorus": "no_abnormal_finding",
        "resected-polyps": "other_procedure",
        "resection-margins": "other_procedure",
        "retroflex-rectum": "no_abnormal_finding",
        "small-bowel-terminal-ileum": "no_abnormal_finding",
        "ulcer": "ulcer",
    },
}


def norm_name(s: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in s.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


# ---------------------------------------------------------------- extraction
def find_zip(name: str) -> Path | None:
    for d in (RAW_DIR, REPO_DIR):
        p = d / name
        if p.exists():
            return p
    return None


def safe_extract(zp: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(zp) as z:
        members = z.infolist()
        for i, m in enumerate(members):
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"Unsafe path in zip: {m.filename}")
            if m.is_dir():
                continue
            if Path(m.filename).suffix.lower() not in IMG_EXT | {".csv", ".txt", ".json", ".xlsx"}:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            with z.open(m) as src, open(target, "wb") as dst:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    dst.write(buf)
            if (i + 1) % 2000 == 0:
                print(f"  extracted {i + 1}/{len(members)}", flush=True)


def extract_sources(sources: list[str], allow_partial: bool) -> list[str]:
    ready = []
    for s in sources:
        info = SOURCES[s]
        dest = EXTRACT_DIR / s
        done = dest / ".extracted_ok"
        if done.exists():
            ready.append(s)
            continue
        zp = find_zip(info["zip"])
        if zp is None:
            print(f"[extract] {s}: {info['zip']} not found, skipping")
            continue
        size = zp.stat().st_size
        if size != info["size"] and not allow_partial:
            print(f"[extract] {s}: {zp} is {size:,} bytes, expected {info['size']:,} "
                  f"(download not finished?) -> skipping")
            continue
        print(f"[extract] {s}: {zp} -> {dest}")
        safe_extract(zp, dest)
        done.write_text(str(size))
        ready.append(s)
    return ready


# ---------------------------------------------------------------- scanning + hashing
def scan(source: str) -> pd.DataFrame:
    root = EXTRACT_DIR / source
    rows = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            rows.append({"path": p.relative_to(DATA_DIR).as_posix(), "source": source,
                         "source_class": norm_name(p.parent.name),
                         "source_folder": p.parent.relative_to(root).as_posix()})
    return pd.DataFrame(rows)


def _hash_one(rel: str):
    import imagehash
    from PIL import Image
    from preprocess import load_and_crop
    try:
        crop, box = load_and_crop(DATA_DIR / rel)
        h = imagehash.phash(Image.fromarray(crop), hash_size=8)
        val = int(str(h), 16)
        thumb = cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA).tobytes().hex()
        return rel, val, crop.shape[1], crop.shape[0], thumb, ""
    except Exception as e:  # unreadable image
        return rel, 0, 0, 0, "", str(e)


def compute_hashes(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    cache_p = DATA_DIR / "phash_cache.csv"
    cols = ["path", "phash", "crop_w", "crop_h", "thumb", "error"]
    cache = pd.read_csv(cache_p) if cache_p.exists() else pd.DataFrame(columns=cols)
    if "thumb" not in cache.columns:      # old cache format -> recompute
        cache = pd.DataFrame(columns=cols)
    cache["error"] = cache["error"].astype(object).fillna("").astype(str)
    have = set(cache["path"])
    todo = [p for p in df["path"] if p not in have]
    if todo:
        print(f"[hash] hashing {len(todo)} images with {workers} workers ...", flush=True)
        out = []
        with Pool(workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_hash_one, todo, chunksize=32)):
                out.append(r)
                if (i + 1) % 2000 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
        new = pd.DataFrame(out, columns=cols)
        new["phash"] = new["phash"].map(lambda v: f"{v:016x}")
        cache = pd.concat([cache, new], ignore_index=True)
        cache.to_csv(cache_p, index=False)
    cache["phash"] = cache["phash"].astype(str)
    cache["thumb"] = cache["thumb"].astype(object).fillna("").astype(str)
    return df.merge(cache, on="path", how="left")


# ---------------------------------------------------------------- hamming / union-find
_POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def popcount64(a: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a)
    return _POP8[a.view(np.uint8)].reshape(a.shape + (8,)).sum(-1)


def close_pairs(h: np.ndarray, thr: int, chunk: int = 1024):
    """All (i, j), i < j, with hamming(h[i], h[j]) <= thr."""
    n = len(h)
    for s in range(0, n, chunk):
        a = h[s:s + chunk, None]
        d = popcount64(np.bitwise_xor(a, h[None, :]))
        ii, jj = np.nonzero(d <= thr)
        ii = ii + s
        keep = jj > ii
        yield from zip(ii[keep].tolist(), jj[keep].tolist())


def thumbs_array(series) -> np.ndarray:
    """8x8 RGB thumbnails (hex) -> float32 [N, 192]."""
    return np.stack([np.frombuffer(bytes.fromhex(t), np.uint8) if t else np.zeros(192, np.uint8)
                     for t in series]).astype(np.float32)


def confirmed_pairs(h: np.ndarray, th: np.ndarray, thr: int, max_colour_diff: float):
    """pHash pairs that also have similar colour layout. pHash alone collides on
    HyperKvasir frames whose crop is dominated by the black corners + green ScopeGuide
    box (e.g. BBPS 0-1 vs 2-3 pairs at hamming <= 4 that are visibly different), so a
    pair also needs mean |8x8 RGB thumbnail difference| <= max_colour_diff."""
    for i, j in close_pairs(h, thr):
        if np.abs(th[i] - th[j]).mean() <= max_colour_diff:
            yield i, j


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def cluster(h: np.ndarray, th: np.ndarray, thr: int, max_size: int, min_thr: int,
            max_colour_diff: float) -> list[list[int]]:
    """Single-linkage components at hamming <= thr. Single linkage chains (one 1600-image
    component at thr 10 on GastroVision), so a component bigger than max_size is
    re-clustered at thr - 1, down to min_thr (the near-duplicate level)."""
    u = UF(len(h))
    for i, j in confirmed_pairs(h, th, thr, max_colour_diff):
        u.union(i, j)
    comps = defaultdict(list)
    for i in range(len(h)):
        comps[u.find(i)].append(i)
    out = []
    for comp in comps.values():
        if len(comp) > max_size and thr > min_thr:
            for sub in cluster(h[comp], th[comp], thr - 1, max_size, min_thr, max_colour_diff):
                out.append([comp[s] for s in sub])
        else:
            out.append(comp)
    return out


# ---------------------------------------------------------------- main logic
def load_class_map(sources_present: list[str], df: pd.DataFrame) -> dict:
    p = CONFIG_DIR / "class_map.json"
    if p.exists():
        cm = load_json(p)
    else:
        cm = {"_comment": "source -> {normalised source class folder: super-class key or 'exclude'}. "
                          "Generated by prepare_data.py; reasons in internal/ML_DATA_INVENTORY.md.",
              "min_class_images": 80}
        for s, m in DEFAULT_CLASS_MAP.items():
            cm[s] = dict(m)
    changed = False
    for s in sources_present:
        cm.setdefault(s, {})
        for c in sorted(df.loc[df.source == s, "source_class"].unique()):
            if c not in cm[s]:
                print(f"[classmap] WARNING: unmapped class {s}/{c} -> 'exclude' (edit class_map.json)")
                cm[s][c] = "exclude"
                changed = True
    if changed or not p.exists():
        save_json(cm, p)
    return cm


def assign_splits(groups: pd.DataFrame, fracs=(0.70, 0.15, 0.15), seed=SEED) -> dict:
    """groups: columns group, label, n. Greedy per-label assignment of groups to splits so
    that image counts approach the fractions. Returns {group: split}."""
    names = ["train", "val", "test"]
    rng = random.Random(seed)
    out = {}
    for label in sorted(groups["label"].unique()):
        g = groups[groups.label == label]
        items = list(zip(g["group"], g["n"]))
        rng.shuffle(items)
        items.sort(key=lambda t: -t[1])          # big groups first (stable after shuffle)
        total = sum(n for _, n in items)
        cur = [0, 0, 0]
        for gid, n in items:
            deficit = [fracs[k] * total - cur[k] for k in range(3)]
            k = int(np.argmax(deficit))
            out[gid] = names[k]
            cur[k] += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--allow-partial-zip", action="store_true")
    ap.add_argument("--dup-hamming", type=int, default=4)
    ap.add_argument("--group-hamming", type=int, default=10)
    ap.add_argument("--max-group", type=int, default=40,
                    help="components larger than this are re-split at a lower hamming threshold")
    ap.add_argument("--dup-colour", type=float, default=12.0,
                    help="max mean |8x8 RGB thumbnail diff| for a near-duplicate pair")
    ap.add_argument("--group-colour", type=float, default=25.0,
                    help="max mean |8x8 RGB thumbnail diff| for a same-group pair")
    ap.add_argument("--min-class", type=int, default=None, help="override class_map min_class_images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-doc", action="store_true", help="do not write internal/ML_DATA_INVENTORY.md")
    args = ap.parse_args()

    ready = extract_sources(args.sources, args.allow_partial_zip)
    if not ready:
        raise SystemExit("No dataset available yet.")
    df = pd.concat([scan(s) for s in ready], ignore_index=True)
    print(f"[scan] {len(df)} images from {ready}")
    df = compute_hashes(df, args.workers)
    bad = df[df["error"].astype(str) != ""]
    if len(bad):
        print(f"[hash] {len(bad)} unreadable images dropped")
    df = df[df["error"].astype(str) == ""].reset_index(drop=True)

    cm = load_class_map(ready, df)
    min_cls = args.min_class if args.min_class is not None else int(cm.get("min_class_images", 80))
    df["mapped"] = [cm[s].get(c, "exclude") for s, c in zip(df.source, df.source_class)]

    # ---- dedupe
    h = np.array([int(x, 16) for x in df["phash"]], dtype=np.uint64)
    uf = UF(len(df))
    npairs = 0
    th_all = thumbs_array(df["thumb"])
    for i, j in confirmed_pairs(h, th_all, args.dup_hamming, args.dup_colour):
        uf.union(i, j)
        npairs += 1
    root = np.array([uf.find(i) for i in range(len(df))])
    df["dup_root"] = root
    df["area"] = df["crop_w"] * df["crop_h"]
    status = np.array(["keep"] * len(df), dtype=object)
    conflict_examples = []
    for r, idx in df.groupby("dup_root").groups.items():
        idx = list(idx)
        if len(idx) == 1:
            continue
        labels = set(df.loc[idx, "mapped"])
        real = labels - {"exclude"}
        if len(real) == 1 and "exclude" in labels:
            # e.g. a BBPS (excluded) frame that is also filed as cecum: keep the labelled copy.
            lab = next(iter(real))
            idx_real = [k for k in idx if df.at[k, "mapped"] == lab]
            best = df.loc[idx_real].sort_values(["area", "path"], ascending=[False, True]).index[0]
            for k in idx:
                if k != best:
                    status[k] = "drop_dup"
        elif len(labels) > 1:
            status[idx] = "drop_conflict"
            if len(conflict_examples) < 15:
                conflict_examples.append(sorted(df.loc[idx, "source"] + "/" + df.loc[idx, "source_class"]))
        else:
            best = df.loc[idx].sort_values(["area", "path"], ascending=[False, True]).index[0]
            for k in idx:
                if k != best:
                    status[k] = "drop_dup"
    df["status"] = status
    cross = df.groupby("dup_root")["source"].nunique()
    n_cross = int((cross > 1).sum())
    print(f"[dedupe] pairs<= {args.dup_hamming}: {npairs}; dropped dup {int((status == 'drop_dup').sum())}, "
          f"conflict {int((status == 'drop_conflict').sum())}; cross-dataset clusters {n_cross}")

    kept = df[df.status == "keep"].copy()
    # ---- drop tiny super-classes
    sc_counts = kept[kept.mapped != "exclude"]["mapped"].value_counts()
    dropped_sc = sorted(c for c, n in sc_counts.items() if n < min_cls)
    kept["label"] = [("exclude" if m in dropped_sc else m) for m in kept["mapped"]]
    if dropped_sc:
        print(f"[classes] dropped super-classes (<{min_cls}): {dropped_sc}")
    classes = [k for k in SUPER_CLASSES if k in set(kept["label"])]
    print(f"[classes] final: {classes}")

    # ---- groups (within label; excluded images grouped within source_class)
    kept = kept.reset_index().rename(columns={"index": "row"})
    kept["glabel"] = np.where(kept.label == "exclude", "exclude:" + kept.source_class, kept.label)
    gid = np.empty(len(kept), dtype=object)
    for gl, sub in kept.groupby("glabel"):
        idx = sub.index.to_numpy()
        hh = np.array([int(x, 16) for x in sub["phash"]], dtype=np.uint64)
        tt = thumbs_array(sub["thumb"])
        for k, comp in enumerate(cluster(hh, tt, args.group_hamming, args.max_group, args.dup_hamming,
                                         args.group_colour)):
            for c in comp:
                gid[idx[c]] = f"{gl}#{k}"
    kept["group"] = gid
    gsize = kept.groupby("group").size()
    print(f"[group] {len(gsize)} groups; size max {gsize.max()}, mean {gsize.mean():.2f}, "
          f">=10: {(gsize >= 10).sum()}")

    groups = kept.groupby("group").agg(label=("glabel", "first"), n=("path", "size")).reset_index()
    smap = assign_splits(groups)
    kept["split"] = kept["group"].map(smap)

    # ---- write
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["path", "source", "source_class", "label", "group"]
    allcols = cols + ["split", "phash", "crop_w", "crop_h"]
    kept.sort_values("path")[allcols].to_csv(SPLITS_DIR / "all_images.csv", index=False)
    for sp in ["train", "val", "test"]:
        sub = kept[(kept.split == sp) & (kept.label != "exclude")].sort_values("path")[cols]
        sub.to_csv(SPLITS_DIR / f"{sp}.csv", index=False, lineterminator="\n")
    test_sha = hashlib.sha256((SPLITS_DIR / "test.csv").read_bytes()).hexdigest()
    save_json({"classes": classes, "sources": ready, "dropped_super_classes": dropped_sc,
               "min_class_images": min_cls, "dup_hamming": args.dup_hamming,
               "group_hamming": args.group_hamming, "test_csv_sha256": test_sha}, SPLITS_DIR / "classes.json")
    df[["path", "source", "source_class", "mapped", "status", "phash"]].to_csv(
        DATA_DIR / "dedupe_report.csv", index=False)
    print(f"[split] test.csv sha256 {test_sha}")

    if not args.no_doc:
        write_inventory(df, kept, cm, classes, dropped_sc, min_cls, test_sha, args, conflict_examples, n_cross)


def write_inventory(df, kept, cm, classes, dropped_sc, min_cls, test_sha, args, conflict_examples, n_cross):
    L = ["# ML_DATA_INVENTORY.md: GI findings datasets (generated by ml/src/prepare_data.py)", ""]
    L.append(f"Sources: {', '.join(sorted(df.source.unique()))}. Images scanned: {len(df)}. "
             f"Kept after dedupe: {len(kept)}.")
    L.append("")
    L.append("## Source classes, counts and mapping")
    L.append("")
    L.append("| source | folder | class (normalised) | images | dup dropped | conflict dropped | -> super-class |")
    L.append("|---|---|---|---:|---:|---:|---|")
    for (s, c), sub in df.groupby(["source", "source_class"]):
        folder = sub["source_folder"].iloc[0]
        mapped = cm[s].get(c, "exclude")
        if mapped in dropped_sc:
            mapped = f"{mapped} (dropped -> exclude)"
        L.append(f"| {s} | {folder} | {c} | {len(sub)} | {(sub.status == 'drop_dup').sum()} | "
                 f"{(sub.status == 'drop_conflict').sum()} | {mapped} |")
    L.append("")
    L.append("## Final classes (index order of the model)")
    L.append("")
    L.append("| index | key | train | val | test | total |")
    L.append("|---:|---|---:|---:|---:|---:|")
    for i, c in enumerate(classes + ["exclude"]):
        sub = kept[kept.label == c]
        n = [int((sub.split == sp).sum()) for sp in ["train", "val", "test"]]
        idx = str(i) if c != "exclude" else "-"
        L.append(f"| {idx} | {c} | {n[0]} | {n[1]} | {n[2]} | {sum(n)} |")
    L.append("")
    if dropped_sc:
        L.append(f"Dropped super-classes (< {min_cls} images after dedupe): {', '.join(dropped_sc)}. "
                 "Their images are kept as `exclude` (gate positives only).")
        L.append("")
    L.append("## Dedupe and grouping")
    L.append("")
    L.append(f"- pHash (64-bit, imagehash) of the contract border-cropped image.")
    L.append(f"- Near-duplicates: hamming <= {args.dup_hamming} AND mean 8x8-thumbnail colour difference "
             f"<= {args.dup_colour} (pHash alone collides on HyperKvasir frames dominated by the black "
             "corners + green ScopeGuide box), any source/class, union-find. A cluster of one real label "
             "plus 'exclude' images keeps the labelled image. "
             f"Dropped as duplicates: {(df.status == 'drop_dup').sum()}; dropped because the "
             f"cluster had conflicting labels: {(df.status == 'drop_conflict').sum()}. "
             f"Clusters spanning both datasets: {n_cross}.")
    if conflict_examples:
        L.append("- Examples of conflicting clusters: " + "; ".join(", ".join(x) for x in conflict_examples[:8]))
    gs = kept.groupby("group").size()
    L.append(f"- Groups: hamming <= {args.group_hamming} and colour diff <= {args.group_colour} within the same "
             "label (union-find; components "
             f"> {args.max_group} images are re-split at a lower threshold, down to {args.dup_hamming}, "
             "because single linkage chains unrelated frames into one giant group). "
             f"{len(gs)} groups, largest {gs.max()}, mean {gs.mean():.2f}, groups with >= 10 images: "
             f"{(gs >= 10).sum()}.")
    L.append("- Split 70/15/15 per label at group level (greedy, seed 42); a group never spans splits.")
    L.append("")
    L.append("## Frozen test set")
    L.append("")
    L.append(f"`ml/splits/test.csv` sha256: `{test_sha}`")
    L.append("")
    L.append("Do not tune anything on the test split. Re-running prepare_data.py with the same zips "
             "and settings reproduces this hash; if it changes, the test set changed.")
    L.append("")
    L.append(MAPPING_REASONS)
    (INTERNAL_DIR / "ML_DATA_INVENTORY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("[doc] wrote internal/ML_DATA_INVENTORY.md")


MAPPING_REASONS = """## Why each mapping

- **no_abnormal_finding**: anatomical landmarks without pathology (cecum, ileum / terminal ileum,
  ileocecal valve, retroflex rectum / stomach, pylorus, z-line / GE junction, duodenal bulb,
  normal oesophagus / stomach / large-bowel mucosa). They are what a "nothing found" image looks like.
- **inflammation**: oesophagitis (all LA grades), ulcerative colitis (all grades; grade 0-1 is
  borderline but is still labelled UC by the experts), erythema, mucosal inflammation of the large bowel.
- **ulcer**: GastroVision "ulcer". Dropped if it has < min_class_images after dedupe.
- **bleeding**: blood in lumen + angiectasia (vascular lesion and a common bleeding source;
  too few to stand alone).
- **polyp**: colon polyps (HyperKvasir "polyps", GastroVision "colon polyps") + gastric polyps.
- **barretts**: Barrett's oesophagus and short-segment Barrett's.
- **varices**: oesophageal varices (GastroVision has very few -> usually dropped).
- **possible_tumour**: colorectal cancer (no upper-GI cancer class exists in either set).
- **other_procedure**: dyed-lifted polyps, dyed resection margins, resected polyps, resection
  margins, accessory tools. Post-intervention / instrument views; kept as their own class so
  they are not mis-filed as disease (and not as "no abnormal finding"). abnormal = false.
- **exclude**: BBPS 0-1 / 2-3 and impacted stool are bowel-preparation *quality* grades, not
  findings (and BBPS 0-1 hides the mucosa). Hemorrhoids (HyperKvasir: 6 images) and colon
  diverticula are too few and not in the agreed key set. Excluded images stay in
  `all_images.csv` and are used as endoscopy positives for the gate.
"""


if __name__ == "__main__":
    main()
