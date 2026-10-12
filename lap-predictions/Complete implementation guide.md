# Complete Implementation Guide: Anomaly Detection for Laparoscopy Snapshots (Predictions Screen)

One document covering the whole build, in three stages, plus integration, release and maintenance.

```
Stage 1: Image classification  →  Stage 2: Classification + Grad-CAM heatmap  →  Stage 3: Object detection (YOLO)  →  Release & maintenance
```

Approach: **transfer learning**. Fine-tune pre-trained models on public laparoscopy datasets (and later your own labeled snapshots). Never train from scratch.

> **Safety:** The output is decision support, not a diagnosis. Every result shows "Decision support only. Requires clinician review." Real clinical use needs ethics approval, clinical validation and regulatory approval (CE/FDA/local authority). Until then, label it a research prototype.

Companion file: `COMPLETE_AGENT_INSTRUCTIONS.md` (task-by-task instructions for your AI agent).

---

## PART A: Foundations

### A1. What "successful" means (agree numbers with a clinician before training)

| Criterion | Starting target |
|---|---|
| Sensitivity for abnormal findings (held-out patients) | ≥ 0.90 |
| Specificity at that sensitivity | ≥ 0.80 |
| Per-class recall, every abnormal class | ≥ 0.70 |
| Calibration | Confidence roughly matches real accuracy |
| External-source performance | Measured and reported (expect a drop) |
| Heatmaps | Highlight tissue, not instruments/borders/text/glare |
| Detection (Stage 3) | mAP@0.5 target agreed with clinician; false alarms per image reported |
| Latency | ≤ 2 s per image on the target PC |
| Safety | "Uncertain" state and disclaimer always present |

If a target isn't met, report honestly what was achieved. Never tune on the test set to hit a number.

### A2. Architecture

```
┌──────────────────────────┐   HTTP (127.0.0.1)   ┌─────────────────────────────────┐
│ Desktop app              │ ───────────────────► │ Inference service (FastAPI)     │
│  Predictions screen      │ ◄─────────────────── │  /predict  classifier (+ CAM)   │
│  upload, analyze, overlay│        JSON          │  /detect   YOLO boxes           │
└──────────────────────────┘                      │  /health   status + versions    │
                                                  └─────────────────────────────────┘
```
- **Approach A (start here):** local FastAPI service launched by the app. Works with any app language.
- **Approach B (later):** embed ONNX models via ONNX Runtime (C#, C++, Java bindings). One installer, no Python at runtime; Grad-CAM and YOLO post-processing must be reimplemented.
- Keep inference **on-premises/offline**. Don't send patient images to a public cloud.

### A3. Data sources (check each license and access terms)

| Dataset | Content | Use |
|---|---|---|
| **GLENDA** | Gynecologic laparoscopy with endometriosis lesion annotations + normal images | Main source of real pathology labels; regions convertible to boxes |
| **LapGyn4** | Gynecologic laparoscopy subsets (anatomy, actions, instruments) | Normal/context images |
| **Cholec80, HeiChole** | Cholecystectomy videos (tools/phases) | Frame volume; "normal" only if verified |
| **Endoscapes, CholecSeg8k** | Cholecystectomy frames with masks/boxes | Stage 3 practice |
| **m2caiSeg (Kaggle)** | Organ/instrument masks with train/trainval/test splits | **Pipeline rehearsal only** (no anomaly labels) |
| **m2cai16-tool-locations** | Tool boxes | Learn the YOLO pipeline |
| **Kvasir / Kvasir-SEG** | GI endoscopy with lesion masks/boxes | Rehearse abnormality detection |
| **Your app's snapshots** | Your camera and lighting | Most valuable; needs ethics approval/consent, de-identification, doctor labels |

Most public laparoscopy sets label tools/phases/anatomy, not pathology. Plan around GLENDA plus your own labeled snapshots.

### A4. Environment and layout

```powershell
python -m venv .venv
.venv\Scripts\activate                       # Linux/macOS: source .venv/bin/activate
pip install torch torchvision                # pytorch.org selector for CUDA builds
pip install timm albumentations scikit-learn pandas matplotlib opencv-python pillow
pip install onnx onnxruntime fastapi uvicorn python-multipart tensorboard grad-cam
pip freeze > requirements.txt
```
Stage 3 adds `pip install ultralytics` (see the license warning in Part D).

```
lap-predictions/
├── data/{raw,clean,processed,yolo}/      # git-ignored
├── src/{dataset.py,train.py,evaluate.py,gradcam.py,export_onnx.py,build_manifest.py}
├── service/main.py
├── models/{classifier_v*.pt, detector_v*.pt, *.onnx, classes.json, config.json, REGISTRY.md}
├── reports/  tests/  notebooks/
└── PROGRESS.md  DECISIONS.md  EXPERIMENTS.md  requirements.txt
```
`.gitignore`: `.venv/`, `data/`, `runs/`, `models/*.pt`, `models/*.onnx`, `__pycache__/`.

Train on **Google Colab** (T4 GPU) or **Kaggle Notebooks**. Colab wipes local disk: keep zips and weights on Drive, copy data to local disk for speed. Public datasets are fine on these services; real patient snapshots need ethics approval first.

---

## PART B: Stage 1, Image classification

### B1. Define classes
Write the class list in `DECISIONS.md` first. Each class must be visible in a single still, have images from **10-15+ patients/videos**, and be supported by data. Start **binary (normal vs abnormal)**, then add fine classes (`endometriosis_lesion`, `adhesion`, `inflammation`, `other_abnormal`) when data allows. `P(abnormal) = 1 − P(normal)`.

### B2. Rehearse on m2caiSeg
It has no anomaly labels. Derive a proxy image-level label from its masks (for example "blood present"), keep its given splits, check no video crosses splits, and run B3-B7 once end to end. This proves the pipeline; it is **not** an anomaly model.

### B3. Manifest
Name files `<patient_or_video_id>__<frame>.jpg` under `data/raw/<source>/<label>/`. Extract video frames at ~1 fps:
```bash
ffmpeg -i video12.mp4 -vf fps=1 -q:v 2 data/raw/cholec80/frames/video12__%06d.jpg
```
```python
# src/build_manifest.py
import pandas as pd
from pathlib import Path
rows = []
for p in Path("data/raw").rglob("*"):
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
    source, label = p.parts[2], p.parts[3]
    rows.append({"filepath": str(p), "source": source, "label_raw": label,
                 "group_id": f"{source}:{p.stem.split('__')[0]}"})
df = pd.DataFrame(rows); df.to_csv("data/manifest.csv", index=False)
print(df.groupby(["source", "label_raw"]).size())
```
Map raw labels to final classes (`LABEL_MAP`), mapping to `normal` only when genuinely normal. Cap huge normal sets per video (for example 60 frames each). Log every mapping decision.

### B4. Clean, split, freeze
```python
import cv2, numpy as np, shutil
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
Path("data/clean").mkdir(exist_ok=True)

def crop_black(img, thr=15):
    ys, xs = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > thr)
    return None if len(ys) == 0 else img[ys.min():ys.max()+1, xs.min():xs.max()+1]

keep, blur = [], []
for i, r in df.iterrows():
    img = cv2.imread(r.filepath); img = None if img is None else crop_black(img)
    if img is None or min(img.shape[:2]) < 100: keep.append(False); blur.append(0); continue
    blur.append(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    out = f"data/clean/{r.group_id.replace(':','-')}__{i}.jpg"
    cv2.imwrite(out, img); df.loc[i, "filepath"] = out; keep.append(True)   # re-saving strips EXIF
df["blur"] = blur; df = df[keep].reset_index(drop=True)
# inspect the 30 lowest 'blur' images, pick a cutoff (often 50-100), then: df = df[df.blur > CUTOFF]

sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
folds = [idx for _, idx in sgkf.split(df, df["label"], groups=df["group_id"])]
df["split"] = "train"; df.loc[folds[0], "split"] = "test"; df.loc[folds[1], "split"] = "val"
assert (df.groupby("group_id")["split"].nunique() == 1).all(), "Leakage!"
print(df.groupby(["label", "split"])["group_id"].nunique().unstack())   # groups per class per split

for r in df.itertuples():
    dst = Path("data/processed") / r.split / r.label; dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(r.filepath, dst / Path(r.filepath).name)
df.to_csv("data/split_manifest.csv", index=False)
```
- **Split by patient/video, never by image.** Near-identical frames in train and test give fake accuracy.
- If an abnormal class has only 1-2 groups in val/test, its metrics are meaningless: use fewer folds, merge classes, or get more data.
- Delete images with visible patient names/timestamps. View 12 random images per class and look for wrong labels, glare, and classes that come from only one source.
- **Freeze the test set** after this step; log a checksum of `split_manifest.csv`.

### B5. Training
Models: **EfficientNet-B0** first (`efficientnet_b0`), then ResNet-50 (`resnet50`); optional comparison with **EndoViT** (Hugging Face `egeozsoy/EndoViT`, a transformer pre-trained on endoscopic images; Grad-CAM is harder on it). ImageNet weights were trained on everyday photos, so fine-tune the **whole network**.

```python
# src/dataset.py
import albumentations as A, numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from PIL import Image
IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
    A.Rotate(limit=25, p=0.5), A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(5, 20, 20, p=0.4),
    A.GaussianBlur(p=0.2), A.Normalize(MEAN, STD), ToTensorV2()])
eval_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(MEAN, STD), ToTensorV2()])
class AlbImageFolder(ImageFolder):
    def __init__(self, root, transform): super().__init__(root); self.tf = transform
    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.tf(image=np.array(Image.open(path).convert("RGB")))["image"], y
```
```python
# src/train.py
import argparse, random, numpy as np, torch, timm
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score
from dataset import AlbImageFolder, train_tf, eval_tf

def main(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = AlbImageFolder(f"{a.data}/train", train_tf); va = AlbImageFolder(f"{a.data}/val", eval_tf)
    classes = tr.classes
    t = np.array([y for _, y in tr.samples]); w = (1.0 / np.bincount(t))[t]
    tl = DataLoader(tr, a.bs, sampler=WeightedRandomSampler(w, len(w)), num_workers=2, pin_memory=True)
    vl = DataLoader(va, a.bs, shuffle=False, num_workers=2, pin_memory=True)
    model = timm.create_model(a.arch, pretrained=True, num_classes=len(classes), drop_rate=0.2).to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
    def run(dl, opt=None):
        train = opt is not None; model.train(train); L, P, G = [], [], []
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.set_grad_enabled(train), torch.autocast(dev, enabled=(dev == "cuda")):
                out = model(x); loss = crit(out, y)
            if train: opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            L.append(loss.item()); P += out.argmax(1).tolist(); G += y.tolist()
        return np.mean(L), balanced_accuracy_score(G, P)
    for p in model.parameters(): p.requires_grad = False                       # Step A: head only
    for p in model.get_classifier().parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for ep in range(a.head_epochs):
        print("[head]", ep, run(tl, opt), run(vl))
    for p in model.parameters(): p.requires_grad = True                        # Step B: full, low LR
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs); best, bad = -1, 0
    for ep in range(a.epochs):
        trl, tra = run(tl, opt); vll, vla = run(vl); sched.step()
        print(f"[full] {ep} train {trl:.3f}/{tra:.2f} val {vll:.3f}/{vla:.2f}")
        if vla > best:
            best, bad = vla, 0
            torch.save({"state": model.state_dict(), "classes": classes, "arch": a.arch}, a.out)
        else:
            bad += 1
            if bad >= a.patience: print("early stop"); break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed"); ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--out", default="models/classifier_v1.pt"); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--head_epochs", type=int, default=5); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
```
Colab run:
```python
from google.colab import drive; drive.mount("/content/drive")
!pip install -q timm albumentations scikit-learn
!mkdir -p /content/lap/data && cp /content/drive/MyDrive/lap_data/processed.zip /content/
!unzip -q -o /content/processed.zip -d /content/lap/data/
!python src/train.py --data /content/lap/data/processed --out /content/drive/MyDrive/lap_models/classifier_v1.pt
```
Reading the log: both losses falling is healthy; train falling while val rises is overfitting (fewer epochs, stronger augmentation, more data); val stuck low means check labels, folders, learning rate, balance.

### B6. Experiments (one change at a time, logged in `EXPERIMENTS.md`)

| Symptom | Try |
|---|---|
| Overfitting | Stronger augmentation, dropout, fewer epochs, more data |
| Low recall on a rare class | More images, merge similar classes, lower threshold |
| Many false positives | More normal images, hard-negative mining, raise threshold |
| Two classes confused | Clinician label review; resolution 288-384 |
| Good val, poor external | Domain shift: add in-house images, stronger colour/blur augmentation, EndoViT |
| Shortcut learning | Crop borders/text, mask instruments, rebalance sources (see Stage 2 heatmaps) |
| Unstable results | Fix seeds, run 3 seeds, report mean ± spread |

Order: backbone comparison → input size → augmentation → data additions. Select by validation per-class recall and balanced accuracy, never by accuracy alone.

### B7. Calibration, threshold, final evaluation
```python
import numpy as np, torch, torch.nn.functional as F
def fit_temperature(logits, labels):              # fit on VAL logits (tensors)
    T = torch.nn.Parameter(torch.ones(1)); opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)
    def closure(): opt.zero_grad(); l = F.cross_entropy(logits / T, labels); l.backward(); return l
    opt.step(closure); return T.item()
def pick_threshold(p_abn, y_abn, target_sens=0.95):
    ths = np.linspace(0.01, 0.99, 99); pos = max(1, (y_abn == 1).sum())
    ok = [t for t in ths if ((p_abn >= t) & (y_abn == 1)).sum() / pos >= target_sens]
    return max(ok) if ok else float(ths.min())
def group_bootstrap(groups, y, pred, metric, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(n):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)]); vals.append(metric(y[idx], pred[idx]))
    return np.percentile(vals, [2.5, 97.5])
```
- Fit temperature and thresholds on **val**; save to `models/config.json` (temperature, abnormal threshold, uncertain cutoff). Apply unchanged to test.
- **Run the frozen test set once.** Report per-class precision/recall/F1, binary sensitivity/specificity, ROC-AUC, PR-AUC, confusion matrix, calibration (reliability curve, ECE), bootstrap 95% CIs resampled by patient/video, and a gallery of the 20 worst errors (false negatives first).
- Evaluate on an external source (different camera/dataset/in-house) if possible; expect a drop. That drop is what to trust.

### B8. Export, serve, UI (Stage 1 release)
```python
import torch, timm, json, numpy as np, onnxruntime as ort
ck = torch.load("models/classifier_v1.pt", map_location="cpu")
m = timm.create_model(ck["arch"], pretrained=False, num_classes=len(ck["classes"])); m.load_state_dict(ck["state"]); m.eval()
x = torch.randn(1, 3, 224, 224)
torch.onnx.export(m, x, "models/classifier_v1.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
json.dump(ck["classes"], open("models/classes.json", "w"))
s = ort.InferenceSession("models/classifier_v1.onnx")
print(np.abs(s.run(None, {"input": x.numpy()})[0] - m(x).detach().numpy()).max())   # expect ~1e-4 or smaller
```
Register the model in `models/REGISTRY.md` (version, date, data checksum, config, metrics). Never overwrite a released file.

```python
# service/main.py  (Stage 1 version; Stage 2 and 3 add to it)
import io, json, numpy as np, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import sys; sys.path.append("../src")
from gradcam import load_model

app = FastAPI(title="Laparoscopy Predictions")
model, classes = load_model("../models/classifier_v1.pt")
cfg = json.load(open("../models/config.json"))          # {"temperature":..,"uncertain_below":..,"abnormal_threshold":..}
MEAN, STD, SIZE = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225]), 224
DISCLAIMER = "Decision support only. Requires clinician review."
MAX_BYTES = 15 * 1024 * 1024

def preprocess(img):
    rgb = np.array(img.convert("RGB").resize((SIZE, SIZE))).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return torch.from_numpy(x), rgb

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"): raise HTTPException(400, "Upload a JPEG or PNG image")
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "Image too large")
    try: img = Image.open(io.BytesIO(data)); img.load()
    except Exception: raise HTTPException(400, "Unreadable image")
    x, rgb = preprocess(img)
    with torch.no_grad(): p = torch.softmax(model(x) / cfg["temperature"], 1)[0].numpy()
    top = int(p.argmax())
    return {"model_version": "classifier_v1",
            "top_label": classes[top] if p[top] >= cfg["uncertain_below"] else "uncertain",
            "confidence": float(p[top]),
            "probabilities": {c: float(v) for c, v in zip(classes, p)},
            "disclaimer": DISCLAIMER}

@app.get("/health")
def health(): return {"status": "ok", "classifier": "classifier_v1"}
```
Run: `cd service && uvicorn main:app --host 127.0.0.1 --port 8000`. Test: `curl -F "file=@snapshot.jpg" http://127.0.0.1:8000/predict`. Preprocessing must be **identical** to training. Add a parity test: the API's probabilities equal the offline evaluation's for sample images.

**Predictions screen v1:** file picker or "pick from this session's snapshots", Analyze button (async call, progress indicator, UI never freezes), ranked probability bars, "uncertain" state below the cutoff, permanent disclaimer, error states (service down, bad file, timeout).

**Stage 1 done when:** targets in A1 are measured (met or honestly not met), the screen works end to end on real snapshots, and the human signs off.

---

## PART C: Stage 2, Classification + Grad-CAM heatmap

No retraining. Grad-CAM shows where the model looked, so it adds explainability and exposes shortcuts.

```python
# src/gradcam.py
import torch, timm, numpy as np, cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = timm.create_model(ck["arch"], pretrained=False, num_classes=len(ck["classes"]))
    m.load_state_dict(ck["state"]); m.eval(); return m, ck["classes"]

def gradcam_overlay(model, tensor, rgb_float, class_idx):
    cam = GradCAM(model=model, target_layers=[model.blocks[-1]])      # EfficientNet in timm
    heat = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    return show_cam_on_image(rgb_float, heat, use_rgb=True), heat

def heat_to_boxes(heat, thr=0.6, min_area=0.01):
    """Cheap region-of-interest boxes from a heatmap (normalized x1,y1,x2,y2). Not a lesion boundary."""
    mask = (heat >= thr).astype("uint8")
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = heat.shape; out = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh >= min_area * w * h: out.append([x / w, y / h, (x + bw) / w, (y + bh) / h])
    return out
```

### C1. Shortcut review (critical)
Generate overlays for 100+ val images, **including correct predictions**. If the heatmaps land on instruments, black borders, watermarks, glare, or dataset-specific cues, the model learned a shortcut. Fix the data (crop, mask, rebalance sources), retrain from B4/B5, and repeat. Have a clinician review ~50 overlays and judge them sensible.

Note: Grad-CAM shows model attention, not a verified lesion boundary. Label it "regions of interest suggested by the model". Transformers (EndoViT) need different CAM methods; keep the CNN as the main model for this stage.

### C2. API v2 (additive)
Extend `/predict` with `heatmap_png_base64` (and optional `regions` from `heat_to_boxes`). Keep every Stage 1 field.
```python
import base64, cv2
from gradcam import gradcam_overlay, heat_to_boxes
# inside /predict, after computing `top`:
overlay, heat = gradcam_overlay(model, x, rgb, top)
ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
result["heatmap_png_base64"] = base64.b64encode(buf).decode()
result["regions"] = heat_to_boxes(heat)
```
Optionally let clients skip it (`?heatmap=false`) to save latency.

### C3. Screen v2
Original / overlay toggle (or side-by-side), opacity slider, caption "regions of interest suggested by the model". Overlay must align with the image at any window size.

**Stage 2 done when:** clinician review of heatmaps is positive, shortcuts are fixed, v1 clients still work unchanged, latency is still acceptable.

---

## PART D: Stage 3, Object detection (YOLO)

> **Licence warning:** Ultralytics YOLO is released under **AGPL-3.0** (a paid enterprise licence exists). Distributing it inside your app can require releasing your app's source. Fine for research/prototype; for a product verify current terms and consider permissively licensed alternatives (YOLOX, MMDetection, torchvision Faster R-CNN/RetinaNet). The pipeline is identical; only the training command changes. Get a human decision before building on it.

### D1. Box labels
Sources: datasets with boxes (Endoscapes, m2cai16-tool-locations, Kvasir-SEG), masks converted to boxes (GLENDA regions, CholecSeg8k), and your own images annotated in **CVAT** or **Label Studio** (free, self-hostable, which suits patient data; export in YOLO format). A clinician annotates or reviews abnormal findings. Write an annotation guideline (what counts as a finding, minimum size, ambiguous cases).

```python
import cv2
def mask_to_yolo(mask_path, cls_id, out_txt, min_area=100):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE); h, w = m.shape
    cnts, _ = cv2.findContours((m > 127).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area: continue
        lines.append(f"{cls_id} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
    open(out_txt, "w").write("\n".join(lines))
```
YOLO label format: one `.txt` per image, one line per box: `class x_center y_center width height` (normalized 0-1). Images with no finding get an empty `.txt` (useful negatives).

### D2. Dataset layout (split by patient/video again; reuse the Stage 1 test patients where possible)
```
data/yolo/images/{train,val,test}/*.jpg
data/yolo/labels/{train,val,test}/*.txt
```
```yaml
# data/yolo/dataset.yaml
path: data/yolo
train: images/train
val: images/val
test: images/test
names: {0: lesion, 1: adhesion}
```
Validate: every image has a label file, boxes are in range, classes are valid, no patient overlap across splits. Save a box-overlay sample grid.

### D3. Train, evaluate, export (Ultralytics shown)
```bash
pip install ultralytics
yolo detect train data=data/yolo/dataset.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 patience=20
yolo detect val   model=runs/detect/train/weights/best.pt data=data/yolo/dataset.yaml split=test
yolo export       model=runs/detect/train/weights/best.pt format=onnx imgsz=640
```
Start with the smallest model; scale up only if accuracy falls short. Copy the best weights to `models/detector_v3.pt`. Evaluate **once** on the frozen test split: mAP@0.5, mAP@0.5:0.95, per-class precision/recall, PR curves, false alarms per image at the chosen confidence threshold (start near 0.25, tune on val for high sensitivity), plus a visual review of predicted boxes by a clinician.

### D4. Serve `/detect`
```python
from ultralytics import YOLO
det = YOLO("../models/detector_v3.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # same validation as /predict (type, size, readable)
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    r = det.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
    return {"model_version": "detector_v3", "image_size": [img.width, img.height],
            "detections": [{"label": det.names[int(b.cls)], "confidence": float(b.conf),
                            "box": [float(v) for v in b.xyxy[0]]} for b in r.boxes],
            "disclaimer": DISCLAIMER}
```
Optional gating: run the Stage 1 classifier first and call the detector when `P(abnormal)` passes the threshold, or always run both and show both.

### D5. Screen v3
Draw boxes over the image (scale from `image_size` to display size, at any window size and zoom), list findings with confidence, per-box show/hide, keep the heatmap toggle from Stage 2. Empty-detection message: "No findings above threshold. This does not mean the image is normal."

**Stage 3 done when:** mAP target met, clinician review of boxes done, boxes align at any size, earlier endpoints unchanged.

---

## PART E: Release, operations and maintenance

### E1. Packaging and lifecycle
- Package the service with **PyInstaller**, or embed ONNX Runtime in the app.
- The app launches the service, polls `/health`, restarts it on failure, stops it on exit. Bind to `127.0.0.1` only.
- Show the model versions in an "About" area. Ship a limitations page.
- Test on a clean machine.

### E2. Feedback loop (build after Stage 1, grow with each stage)
- Agree / Disagree / Correct-label (and box edits for Stage 3) on the Predictions screen.
- Store de-identified records: `image_ref, model_version, predicted, corrected, timestamp` (SQLite is enough). Export them to the training manifest format. Only use patient data with consent and ethics approval.
- Audit log: user, time, image reference, model version, output. No patient identifiers.

### E3. Retraining and versioning
1. Collect confirmed corrections; clinician-review them before they become labels.
2. Retrain; compare against the previous version on the **same frozen test set** (and a growing external set).
3. Release only if it beats the current model on the agreed criteria; record in `REGISTRY.md`.
4. Never overwrite released models; keep rollback weights.
5. Monitor drift (new camera, lighting, procedure): track the "uncertain" rate and the override rate.

### E4. Safety, privacy, regulation
- De-identify images, strip metadata, crop on-screen text. Never commit patient images; don't send them to public clouds.
- State known limitations in the app and README.
- A model used in real procedures is medical-device software: ethics approval, clinical validation, risk management and regulatory approval apply. Until then, a research prototype.

---

## PART F: Timeline, pitfalls, checklist

| Week | Goal |
|---|---|
| 1 | Success criteria, classes, datasets, manifest, clean, split, m2caiSeg rehearsal |
| 2 | Baseline training, experiments, backbone comparison |
| 3 | Calibration, threshold, final test, external check, ONNX export, `/predict`, screen v1 (**Stage 1 done**) |
| 4 | Grad-CAM, shortcut review, API/UI v2 (**Stage 2 done**) |
| 5-6 | Box labels (convert + annotate), YOLO training and evaluation |
| 7 | `/detect`, screen v3, feedback loop, packaging, docs (**Stage 3 done**) |

**Pitfalls:** splitting by image instead of patient/video; judging by accuracy instead of per-class sensitivity (and mAP); tuning on the test set; treating unlabeled frames as normal without checking; domain shift between public data and your camera; shortcut learning (borders, text, instruments, dataset source); trusting Grad-CAM as a lesion boundary; overconfident outputs with no "uncertain" state; different preprocessing in training and serving; too few box annotations; AGPL/dataset licence violations; privacy lapses.

**Master checklist**
- [ ] Success criteria, classes, label mapping, licences recorded
- [ ] Manifest, cleaning, patient-level split, leakage test, test set frozen
- [ ] m2caiSeg rehearsal done
- [ ] Baseline + experiments logged; calibration and thresholds fitted on val
- [ ] Test evaluated once with bootstrap CIs; external check; failure gallery
- [ ] ONNX parity verified; registry updated
- [ ] `/predict` + screen v1 (**Stage 1**)
- [ ] Grad-CAM shortcut review; API/UI v2 (**Stage 2**)
- [ ] Box labels, YOLO trained/evaluated, `/detect` + screen v3 (**Stage 3**)
- [ ] Feedback loop, audit log, packaging, README, limitations page
- [ ] Retraining and versioning procedure documented