# Predictions Screen: Staged CNN Anomaly Detection for Laparoscopy Snapshots

Adds a **Predictions** screen to the desktop application. A doctor uploads a snapshot, the model suggests abnormalities, and the feature grows in three stages:

```
Stage 1: Image classification  →  Stage 2: Classification + heatmap (Grad-CAM)  →  Stage 3: Object detection (YOLO)
   "what is in this image?"          "and where did the model look?"                  "boxes around each anomaly"
```

Approach decided: **transfer learning**. Take a pre-trained model and fine-tune it on public laparoscopy images. No training from scratch.

> **Important:** This is decision support, not diagnosis. Show "suggested finding, clinician review required" on every result. Use on real patients would make this medical-device software (regulatory approval, clinical validation, data-protection compliance). Until then, label it a research prototype.

Companion file: `AGENT_INSTRUCTIONS.md` (feature-by-feature instructions for your AI coding agent).

---

## 1. Staged roadmap

Each stage ends with a **working, shippable Predictions screen**. Later stages extend the earlier ones rather than replace them.

| | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| **Feature** | Image classification | Classification + Grad-CAM heatmap | Object detection (YOLO) |
| **Output** | Class + confidence + probabilities | Stage 1 + heatmap overlay | Bounding boxes + class + confidence per finding |
| **Model** | Pre-trained CNN (EfficientNet-B0 / ResNet-50) fine-tuned | Same model, no retraining | Pre-trained YOLO fine-tuned on box labels |
| **Labels needed** | Image-level labels | Same as Stage 1 | Bounding boxes (or masks converted to boxes) |
| **API** | `POST /predict` | `POST /predict` adds `heatmap` field | `POST /detect` |
| **UI change** | Upload, results list, disclaimer | Add heatmap overlay + opacity slider | Draw boxes over image, list per-box findings |
| **Exit criteria** | Test-set sensitivity target met on held-out patients | Heatmaps reviewed by a clinician on 50+ cases and judged sensible | Test mAP@0.5 target met; clinician review of boxes |
| **Effort** | ~1.5 weeks | ~3-4 days | ~2 weeks (mostly data prep) |

**Why this order works:** Stage 1 proves the data and pipeline. Stage 2 adds explainability at almost no cost, and its heatmaps show whether the model learned real pathology or shortcuts. Stage 3 needs the expensive part (box annotations), so you do it once the pipeline is trusted.

**Design rules that keep the stages compatible:**
- Version the API and the model files (`classifier_v1`, `classifier_v2`, `detector_v3`).
- Only add fields to the response JSON; never rename or remove them.
- Keep the same label list across stages.
- Keep the test set fixed so versions are comparable.

---

## 2. Model choice (transfer learning)

**Stages 1-2 (classification):** pre-trained ImageNet backbone via `timm`.
- Start with **EfficientNet-B0** (good accuracy for size) or **ResNet-50** (simple, easy Grad-CAM).
- Compare 2-3 backbones on the validation set; keep the best. ConvNeXt-Tiny is a strong third option; MobileNetV3 if the target PC is weak.

**Stage 3 (detection):** pre-trained YOLO (COCO weights), fine-tuned on your boxes. Start with the smallest size (`n` or `s`), scale up only if accuracy is short.

**License check before shipping (important for a distributed desktop app):** Ultralytics YOLO is released under **AGPL-3.0** (with a paid enterprise license available). AGPL can require you to open-source your application if you distribute it. Fine for research/prototype; for a product, verify current terms and consider Apache/BSD-licensed alternatives such as **YOLOX**, **MMDetection**, or **torchvision Faster R-CNN / RetinaNet**. The pipeline below is the same for these; only the training command changes.

---

## 3. Where to get images (free)

Always check each dataset's license and access terms. Most are research/non-commercial, some require a signed agreement, and terms can change.

| Dataset | Content | Use for |
|---|---|---|
| **GLENDA** | Gynecologic laparoscopy with endometriosis lesion annotations + normal images | Stage 1 pathology labels; region annotations can be converted to boxes for Stage 3 |
| **LapGyn4** | Gynecologic laparoscopy image subsets (anatomy, actions, instruments) | Normal/context images, extra training data |
| **Cholec80** | 80 cholecystectomy videos, tool + phase labels | Large volume of frames, normal class, pretraining |
| **CholecSeg8k** | Cholec80 frames with pixel masks | Masks → boxes for Stage 3 practice |
| **Endoscapes** | Cholecystectomy frames with segmentation and boxes for anatomy/tools | Stage 3 detection practice |
| **HeiChole** | Cholecystectomy videos with phases, tools, actions | Frame volume |
| **m2cai16-tool-locations** | Tool bounding boxes in laparoscopy frames | Learn the YOLO pipeline on laparoscopic imagery |
| **Kvasir-SEG / Kvasir** | GI endoscopy (not laparoscopy) with lesion masks and boxes | Rehearse the abnormality pipeline end to end |
| **EndoVis (MICCAI)** | Instrument segmentation, scene understanding | Segmentation/detection practice |

Where to search: the dataset's own site (GLENDA/LapGyn4 are hosted by ITEC, University of Klagenfurt), **Kaggle**, **Zenodo**, **Synapse**, **Hugging Face Datasets**, **Papers with Code → Datasets**. Search terms: "laparoscopy dataset", "surgical video dataset", "endometriosis laparoscopy images".

**Reality check:** Most public laparoscopy datasets label tools, phases and anatomy, not pathology. Real abnormality labels are scarce (GLENDA is the main one). Plan to:
1. Use GLENDA (or another pathology set) for your abnormal classes.
2. Use the large datasets for **normal** samples.
3. Add **your own app's snapshots**, doctor-labeled, as soon as possible. This matters most for real performance.

**Privacy:** Snapshots from your app are patient data. Get ethics approval/consent, strip metadata (EXIF/DICOM tags), and crop out any on-screen patient information.

---

## 4. Architecture

```
┌─────────────────────────┐        ┌──────────────────────────────┐
│  Desktop App            │  HTTP  │  Inference Service           │
│  Predictions screen     │───────►│  FastAPI + ONNX Runtime/     │
│  - upload / pick        │  JSON  │  PyTorch                     │
│  - analyze              │◄───────│  /predict  (Stage 1-2)       │
│  - overlay results      │        │  /detect   (Stage 3)         │
└─────────────────────────┘        └──────────────────────────────┘
```

- **Approach A (recommended first):** local FastAPI service on `127.0.0.1`, launched and stopped by the desktop app. Works with any desktop language.
- **Approach B (later):** embed the ONNX model in the app via ONNX Runtime (C#, C++, Java bindings). One installer, no Python at runtime, but Grad-CAM and YOLO post-processing must be reimplemented.
- Keep inference **on-premises/offline**. Do not send patient images to a public cloud.

---

## 5. Implementation, stage by stage

### Phase 0: Setup (Day 1)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install torch torchvision timm albumentations scikit-learn pandas matplotlib \
            opencv-python grad-cam onnx onnxruntime fastapi uvicorn python-multipart
```

Later, for Stage 3: `pip install ultralytics`.

Suggested layout:
```
lap-predictions/
├── data/{raw,processed,yolo}/
├── notebooks/
├── src/{dataset.py,train.py,evaluate.py,gradcam.py,export_onnx.py}
├── service/main.py
├── models/            # classifier_v1.pt, classifier_v2.pt, detector_v3.pt, classes.json
├── AGENT_INSTRUCTIONS.md
├── PROGRESS.md
└── DECISIONS.md
```

Train on **Google Colab** or **Kaggle Notebooks** (free GPU).

---

### STAGE 1: Image classification

**1.1 Define labels.** Write the class list before anything else, e.g. `normal, endometriosis_lesion, adhesion, inflammation, other_abnormal`. Map every source dataset's labels into this list.

**1.2 Prepare data.**
- Extract frames from videos (1 fps is enough):
  ```bash
  ffmpeg -i video.mp4 -vf fps=1 frames/vid001_%05d.jpg
  ```
- Remove black borders, blurry/black frames.
- **Split by patient/video, never by frame** (70/15/15). Near-identical frames in both train and test give fake 99% accuracy.
- Folder layout: `data/processed/{train,val,test}/<class>/*.jpg`
- Check class balance; handle with weighted sampling.

**1.3 Dataset and augmentation** (`src/dataset.py`):

```python
import albumentations as A, numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import ImageFolder
from PIL import Image

IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2),
    A.Rotate(limit=25, p=0.5), A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(5, 20, 20, p=0.4), A.GaussianBlur(p=0.2),
    A.Normalize(MEAN, STD), ToTensorV2()])
val_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(MEAN, STD), ToTensorV2()])

class AlbImageFolder(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root); self.tf = transform
    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.tf(image=np.array(Image.open(path).convert("RGB")))["image"], y
```

**1.4 Fine-tune** (`src/train.py`), two-step transfer learning:

```python
import torch, timm, numpy as np
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
from dataset import AlbImageFolder, train_tf, val_tf

device = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = AlbImageFolder("data/processed/train", train_tf)
val_ds   = AlbImageFolder("data/processed/val", val_tf)

targets = np.array([y for _, y in train_ds.samples])
w = (1.0 / np.bincount(targets))[targets]
train_dl = DataLoader(train_ds, 32, sampler=WeightedRandomSampler(w, len(w)), num_workers=2)
val_dl   = DataLoader(val_ds, 32, shuffle=False, num_workers=2)

model = timm.create_model("efficientnet_b0", pretrained=True,
                          num_classes=len(train_ds.classes)).to(device)
crit = nn.CrossEntropyLoss()

def run(dl, train=False, opt=None):
    model.train(train); L, P, G = [], [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x); loss = crit(out, y)
            if train: opt.zero_grad(); loss.backward(); opt.step()
        L.append(loss.item()); P += out.argmax(1).tolist(); G += y.tolist()
    return np.mean(L), P, G

# Step A: train only the new classifier head
for p in model.parameters(): p.requires_grad = False
for p in model.get_classifier().parameters(): p.requires_grad = True
opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 1e-3)
for ep in range(5): run(train_dl, True, opt)

# Step B: unfreeze all, low learning rate
for p in model.parameters(): p.requires_grad = True
opt = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20)
best = 1e9
for ep in range(20):
    tl, _, _ = run(train_dl, True, opt); vl, P, G = run(val_dl); sched.step()
    print(ep, round(tl, 3), round(vl, 3))
    if vl < best:
        best = vl
        torch.save({"state": model.state_dict(), "classes": train_ds.classes}, "models/classifier_v1.pt")
print(classification_report(G, P, target_names=train_ds.classes))
```

**1.5 Evaluate on the untouched test set:** per-class **sensitivity** (missing an abnormality is the costly error), specificity/precision, confusion matrix, ROC-AUC per class, calibration (temperature scaling if overconfident). Inspect failure cases by eye. Also test on images from a different source than the training data; expect a drop.

**1.6 Export and serve.** Export to ONNX (`torch.onnx.export`, opset 17, dynamic batch) and save `models/classes.json`. Build the FastAPI service:

```python
# service/main.py
import io, numpy as np, torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import sys; sys.path.append("../src")
from gradcam import load_model

app = FastAPI(title="Laparoscopy Predictions")
model, classes = load_model("../models/classifier_v1.pt")
MEAN, STD, SIZE, CONF_MIN = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225]), 224, 0.5

def preprocess(img):
    rgb = np.array(img.convert("RGB").resize((SIZE, SIZE))).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return torch.from_numpy(x), rgb

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Upload a JPEG or PNG image")
    x, rgb = preprocess(Image.open(io.BytesIO(await file.read())))
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].numpy()
    top = int(p.argmax())
    return {"model_version": "classifier_v1",
            "top_label": classes[top] if p[top] >= CONF_MIN else "uncertain",
            "confidence": float(p[top]),
            "probabilities": {c: float(v) for c, v in zip(classes, p)},
            "disclaimer": "Decision support only. Requires clinician review."}

@app.get("/health")
def health(): return {"status": "ok"}
```

Run: `uvicorn main:app --host 127.0.0.1 --port 8000`. Test: `curl -F "file=@snapshot.jpg" http://127.0.0.1:8000/predict`.

**1.7 Predictions screen v1.** File picker or "pick from this session's snapshots", Analyze button (async call, progress indicator), ranked probability bars, "uncertain" state below the confidence threshold, permanent disclaimer, error handling (service down, bad file, timeout).

**Stage 1 done when:** test sensitivity for abnormal classes meets your target on held-out patients, the screen works end to end, and it has been tried on your app's real snapshots.

---

### STAGE 2: Classification + heatmap (Grad-CAM)

No retraining. Add a heatmap of the regions that drove the prediction (`src/gradcam.py`):

```python
import torch, timm, numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def load_model(path):
    ck = torch.load(path, map_location="cpu")
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(ck["classes"]))
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["classes"]

def gradcam_overlay(model, tensor, rgb_float, class_idx):
    cam = GradCAM(model=model, target_layers=[model.blocks[-1]])   # last conv block
    heat = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
    return show_cam_on_image(rgb_float, heat, use_rgb=True), heat
```

Extend `/predict` to return `heatmap_png_base64` (PNG-encode the overlay with OpenCV, then base64). Existing fields stay unchanged.

**Screen v2:** add original / overlay toggle (or side-by-side) and an opacity slider.

**Critical review step:** look at heatmaps on **correct** predictions too. If the model highlights instruments, watermarks, borders or glare instead of tissue, it learned a shortcut. Fix the data (crop, mask, rebalance) and retrain Stage 1 before moving on.

Note: Grad-CAM shows model attention, not a verified lesion boundary. Label it "regions of interest suggested by the model".

**Stage 2 done when:** a clinician has reviewed heatmaps on 50+ cases and judged them sensible, and shortcut problems are fixed.

---

### STAGE 3: Object detection with YOLO

**3.1 Get box labels.** Sources: datasets with boxes (Endoscapes, m2cai16-tool-locations, Kvasir-SEG), masks converted to boxes (GLENDA regions, CholecSeg8k, Kvasir-SEG), and your own images annotated in **CVAT** or **Label Studio** (both free; export in YOLO format). Have a clinician annotate or review abnormal findings.

Mask → YOLO box conversion:

```python
import cv2, os
def mask_to_yolo(mask_path, cls_id, out_txt):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE); h, w = m.shape
    cnts, _ = cv2.findContours((m > 127).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 100: continue           # skip tiny noise
        lines.append(f"{cls_id} {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
    open(out_txt, "w").write("\n".join(lines))
```

YOLO label format: one `.txt` per image, one line per box: `class x_center y_center width height` (normalized 0-1). Images with no abnormality get an empty `.txt` (they act as useful negatives).

**3.2 Dataset layout and config** (split by patient/video again):

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
names:
  0: lesion
  1: adhesion
```

**3.3 Fine-tune a pre-trained YOLO** (Ultralytics shown; check the license note in Section 2):

```bash
pip install ultralytics
yolo detect train data=data/yolo/dataset.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 patience=20
yolo detect val   model=runs/detect/train/weights/best.pt data=data/yolo/dataset.yaml split=test
yolo export       model=runs/detect/train/weights/best.pt format=onnx imgsz=640
```

Copy the best weights to `models/detector_v3.pt`.

**3.4 Evaluate.** mAP@0.5 and mAP@0.5:0.95, per-class precision/recall, PR curves, and a visual review of predicted boxes on the test set. Tune the confidence threshold for high sensitivity (start around 0.25), then check false alarms per image.

**3.5 Serve `/detect`:**

```python
from ultralytics import YOLO
det = YOLO("../models/detector_v3.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    r = det.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
    return {"model_version": "detector_v3", "image_size": [img.width, img.height],
            "detections": [{"label": det.names[int(b.cls)], "confidence": float(b.conf),
                            "box": [float(v) for v in b.xyxy[0]]} for b in r.boxes],
            "disclaimer": "Decision support only. Requires clinician review."}
```

**Screen v3:** draw boxes over the image (scale from `image_size` to the display size), list each finding with confidence, per-box show/hide, keep the heatmap toggle from Stage 2. Optionally run the Stage 1 classifier as a gate first, then the detector.

**Stage 3 done when:** mAP target met, clinician review of boxes done, and the screen renders boxes correctly at any window size.

---

### Feedback loop and hardening (after each stage)

- Agree / Disagree / Correct-label buttons; store corrections as future training data (with consent, de-identified).
- Audit log: user, time, image reference, model version, output.
- Retrain on a schedule; never overwrite a model; keep the fixed test set.
- Monitor drift (new camera, lighting, procedure).
- Package the service with **PyInstaller** or embed via ONNX Runtime; the app spawns it, polls `/health`, and stops it on exit.

---

## 6. Toolkit: what to get and where

| Need | Tool | Where to find it |
|---|---|---|
| Deep learning framework | **PyTorch** | pytorch.org (use the install selector for your CUDA version) |
| Pre-trained classifiers | **timm** (PyTorch Image Models) | github.com/huggingface/pytorch-image-models, `pip install timm` |
| Heatmaps | **pytorch-grad-cam** | github.com/jacobgil/pytorch-grad-cam, `pip install grad-cam` |
| Detection (YOLO) | **Ultralytics** | docs.ultralytics.com, `pip install ultralytics` (mind AGPL) |
| Permissive detection alternatives | YOLOX, MMDetection, torchvision detection | github.com/Megvii-BaseDetection/YOLOX, github.com/open-mmlab/mmdetection, pytorch.org/vision |
| Augmentation | **Albumentations** | albumentations.ai |
| Annotation (boxes) | **CVAT**, **Label Studio** | cvat.ai, labelstud.io (both free; self-hostable, which is better for patient data) |
| Frame extraction | **FFmpeg** | ffmpeg.org |
| Image processing | **OpenCV** | `pip install opencv-python` |
| Metrics | **scikit-learn** | scikit-learn.org |
| Experiment tracking | **TensorBoard**, **Weights & Biases**, **MLflow** | tensorboard (with PyTorch), wandb.ai, mlflow.org |
| Data/model versioning | **DVC**, Git LFS | dvc.org, git-lfs.com |
| Model export/runtime | **ONNX**, **ONNX Runtime** | onnx.ai, onnxruntime.ai (bindings for C#, C++, Java, Python) |
| Serving | **FastAPI** + **Uvicorn** | fastapi.tiangolo.com, uvicorn.org |
| Packaging | **PyInstaller** | pyinstaller.org |
| Free GPU | **Google Colab**, **Kaggle Notebooks** | colab.research.google.com, kaggle.com/code |
| Datasets & pre-trained weights | Kaggle, Zenodo, Hugging Face, Papers with Code | kaggle.com/datasets, zenodo.org, huggingface.co, paperswithcode.com |

**Hardware:** free Colab/Kaggle GPUs are enough for everything here. Locally, an NVIDIA GPU with 8 GB+ VRAM makes iteration faster. Inference for one image runs fine on CPU.

---

## 7. Timeline

| Week | Goal |
|---|---|
| 1 | Labels, datasets, data prep, first classifier |
| 2 | Tune/evaluate classifier, export, API v1, Predictions screen v1 (**Stage 1 done**) |
| 3 | Grad-CAM, API/UI v2, clinician review of heatmaps (**Stage 2 done**) |
| 4-5 | Box labels (convert + annotate), YOLO training and evaluation |
| 6 | `/detect`, screen v3, feedback loop, packaging, docs (**Stage 3 done**) |

---

## 8. Common pitfalls

- **Data leakage:** splitting by frame instead of patient/video.
- **Accuracy-only thinking:** always check per-class sensitivity, and mAP for detection.
- **Domain shift:** a model trained on public data may fail on your app's snapshots. Test early.
- **Shortcut learning:** instruments, watermarks, hospital colour tone. Use Stage 2 heatmaps to catch it.
- **Trusting Grad-CAM as a lesion boundary.**
- **Too few box annotations for Stage 3:** start small, review predictions, add data where it fails (active learning).
- **Overconfidence:** show "uncertain" when confidence is low.
- **Licensing:** Ultralytics AGPL, dataset non-commercial terms.
- **Privacy:** de-identify images, keep inference on-premises, no patient data in the repo or logs.

---

## 9. Checklist

- [ ] Class list defined
- [ ] Datasets downloaded, licenses checked
- [ ] Frames extracted, cleaned, split by patient/video
- [ ] **Stage 1:** classifier fine-tuned, evaluated, exported, `/predict`, screen v1
- [ ] **Stage 2:** Grad-CAM, heatmap in API and screen v2, clinician review, shortcuts fixed
- [ ] **Stage 3:** box labels ready, YOLO fine-tuned and evaluated, `/detect`, screen v3
- [ ] Feedback buttons, audit log, model versioning
- [ ] Packaging and documentation