# AGENT_INSTRUCTIONS.md: Predictions Screen (Laparoscopy Anomaly Detection)

You are a coding agent helping build a **Predictions screen** in an existing desktop application that captures laparoscopy video and snapshots. A doctor uploads a snapshot; an ML model suggests abnormalities.

Read `predictions_screen_cnn_guide.md` first. It is the design reference (roadmap, code samples, datasets, toolkit). This file tells you **how to work**: feature by feature, in order, with acceptance criteria.

---

## 1. Ground rules

1. **One feature at a time.** Finish a feature (all acceptance criteria met) before starting the next. Never work ahead.
2. **Follow the order.** Features are numbered F0 to F15 and grouped into three stages: Stage 1 classification, Stage 2 classification + Grad-CAM heatmap, Stage 3 YOLO detection. Do not start Stage 2 until Stage 1 is accepted, and so on.
3. **Transfer learning only.** Fine-tune pre-trained models. Never train from scratch.
4. **Existing app first.** Before writing code, inspect the desktop app (language, framework, UI toolkit, how it stores snapshots). Record findings in `DECISIONS.md`. Match the app's conventions; do not rewrite unrelated code.
5. **Do not invent facts.** Do not make up dataset URLs, file formats, metric results or library APIs. If unsure, check the official docs or ask.
6. **No patient data in the repo, logs, or commits.** Use `.gitignore` for `data/`, `models/*.pt`, `runs/`. No real patient images in tests. Strip image metadata.
7. **Split data by patient/video, never by frame.** Keep the test set frozen after it is created. Touch it only for final evaluation.
8. **Additive API changes only.** Add response fields; never rename or remove them. Version models (`classifier_v1`, `classifier_v2`, `detector_v3`); never overwrite a released model file.
9. **Safety wording is mandatory.** Every result shown to a user carries: "Decision support only. Requires clinician review." Never call output a diagnosis.
10. **Check licenses.** Record each dataset's license/terms in `DECISIONS.md`. Flag Ultralytics AGPL-3.0 before Stage 3 (see the guide, Section 2) and get a human decision on which detector to use.
11. **Small commits.** One commit per completed task, message format `F<number>: <what>`.
12. **Keep it reproducible.** Fixed random seeds, pinned `requirements.txt`, config in files (not hard-coded paths).

---

## 2. Working protocol (repeat for every feature)

1. **Read** the feature block below and the matching guide section.
2. **Plan** in 3-8 bullets; write them under the feature in `PROGRESS.md`.
3. **Implement** the tasks.
4. **Test** against every acceptance criterion. Run the tests; show real outputs (metrics, screenshots, logs), not claims.
5. **Update** `PROGRESS.md` (status, what changed, metrics, open issues) and `DECISIONS.md` (choices + reasons).
6. **Report** using the template in Section 5, then **stop and wait for approval** at every 🛑 gate.

### Stop and ask the human when
- A dataset needs registration, a signed agreement, or manual download.
- You need clinician input (label definitions, reviewing heatmaps/boxes, choosing a sensitivity target).
- The class list, thresholds, or detector library must be decided.
- A criterion can't be met after reasonable attempts (say what you tried).
- You find licensing, privacy, or safety concerns.
- Anything would change the existing app's architecture significantly.

---

## 3. Tech stack (pin versions in `requirements.txt`)

| Purpose | Tool |
|---|---|
| Framework | PyTorch + torchvision |
| Pre-trained classifiers | `timm` (EfficientNet-B0 first; also try ResNet-50) |
| Augmentation | Albumentations |
| Heatmaps | `grad-cam` (pytorch-grad-cam) |
| Detection (Stage 3) | Ultralytics YOLO **or** a permissive alternative, decided at F10 |
| Metrics | scikit-learn |
| Frame extraction | FFmpeg, OpenCV |
| Annotation | CVAT or Label Studio |
| Export/runtime | ONNX, ONNX Runtime |
| Service | FastAPI + Uvicorn (bind `127.0.0.1`) |
| Tracking | TensorBoard (or W&B/MLflow if the human prefers) |
| Packaging | PyInstaller (or ONNX Runtime embedded in the app) |
| Training compute | Google Colab / Kaggle (free GPU) |

Repo layout (create if missing):
```
data/{raw,processed,yolo}/     # git-ignored
src/{dataset.py,train.py,evaluate.py,gradcam.py,export_onnx.py}
service/main.py
models/                        # versioned weights + classes.json (weights git-ignored / LFS)
tests/
PROGRESS.md  DECISIONS.md  requirements.txt
```

---

## 4. Feature roadmap

### STAGE 1: Image classification

#### F0. Discovery and scaffold
- **Tasks:** Inspect the desktop app (language, UI framework, snapshot storage, build/run steps). Create the repo layout, `requirements.txt`, `.gitignore`, `PROGRESS.md`, `DECISIONS.md`. Decide and record the integration approach (default: local FastAPI service, ONNX export for later embedding).
- **Acceptance:** app stack documented; environment installs from scratch; a `/health` stub responds; Predictions screen location in the app's navigation identified.
- **Human input:** confirm integration approach.

#### F1. Label schema and data acquisition 🛑
- **Tasks:** Propose a class list (default: `normal`, plus the abnormal classes the available data supports). List candidate datasets from the guide (GLENDA, LapGyn4, Cholec80, etc.) with license, size, label type, and access steps. Provide download scripts only where downloads are direct and permitted.
- **Acceptance:** `DECISIONS.md` has the approved class list and a dataset table (source, license, what it contributes to which class). Raw data placed in `data/raw/` with a manifest (`manifest.csv`: file, source, patient/video id, label).
- **Human input:** approve classes; download gated datasets; confirm ethics/consent for any in-house images.

#### F2. Data preparation and patient-level split
- **Tasks:** Extract frames (about 1 fps), crop black borders, drop blurry/black frames, map source labels to the class list, remove metadata, split **by patient/video** into train/val/test (70/15/15) with a fixed seed, write `data/processed/` folders and a split manifest, report class counts.
- **Acceptance:** no patient/video appears in more than one split (automated test); class distribution table saved; a sample-grid image per class for visual check; imbalance noted.
- **Human input:** review the sample grids for label sanity.

#### F3. Baseline classifier (`classifier_v1`)
- **Tasks:** Implement `dataset.py` (augmentations) and `train.py` (two-step fine-tune: head only, then full at low LR; weighted sampling for imbalance; TensorBoard logging; best checkpoint by validation loss). Try EfficientNet-B0, then one more backbone; keep the best.
- **Acceptance:** training runs end to end on Colab/Kaggle or local GPU; curves logged; checkpoint saved as `models/classifier_v1.pt` with the class list; no sign of severe overfitting (report train vs val).
- **Human input:** none, unless compute is a problem.

#### F4. Evaluation report 🛑
- **Tasks:** On the frozen test set: per-class sensitivity, specificity, precision, F1, ROC-AUC, confusion matrix, calibration check. Review failure cases visually. If possible, evaluate on images from a different source than training. Write `reports/stage1_eval.md`.
- **Acceptance:** report includes all metrics + failure-case gallery + a candid list of weaknesses. Sensitivity target for abnormal classes agreed with the human.
- **Human input:** set/approve the sensitivity target; clinician glance at failure cases if available. If the target is not met, iterate (more data, augmentation, backbone, threshold) and re-report.

#### F5. Export and model registry
- **Tasks:** Export ONNX (opset 17, dynamic batch), verify ONNX outputs match PyTorch within tolerance, save `models/classes.json`, add a `models/REGISTRY.md` entry (version, date, data version, metrics).
- **Acceptance:** parity test passes; registry updated; old versions untouched.

#### F6. Inference API v1: `/predict`
- **Tasks:** Implement `service/main.py` with `/health` and `POST /predict` (JPEG/PNG only, size limit, preprocessing identical to training, softmax probabilities, `uncertain` below the confidence threshold, `model_version` and `disclaimer` in the response).
- **Acceptance:** tests cover valid image, wrong type, corrupted file, oversized file; responds in about 1-2 s on CPU; binds to `127.0.0.1`; response schema documented in the README.

#### F7. Predictions screen v1 🛑
- **Tasks:** In the desktop app add the **Predictions** screen: file picker (and "pick from this session's snapshots" if easy), image preview, Analyze button (async, non-blocking, progress indicator), ranked probability bars, "uncertain" state, permanent disclaimer, error states (service down, bad file, timeout). App starts/stops the service or documents how.
- **Acceptance:** works end to end on real snapshots from the app; UI never freezes; all error states handled; screenshots attached to the report.
- **Human input:** UX review. **Stage 1 sign-off gate.**

---

### STAGE 2: Classification + heatmap

#### F8. Grad-CAM engine (`classifier_v2`, same weights)
- **Tasks:** Implement `gradcam.py` (target the last conv block; overlay function; optional threshold → contour → region-of-interest box). Produce heatmaps on 100+ test images, including **correct** predictions.
- **Acceptance:** heatmap gallery saved; a written check for shortcut learning (do heatmaps land on instruments, borders, watermarks, glare?). If shortcuts are found, fix the data (crop/mask/rebalance), retrain via F2-F4, and re-run this feature.
- **Human input:** clinician reviews about 50 heatmaps and judges them sensible or not. 🛑

#### F9. API v2 and screen v2
- **Tasks:** Add `heatmap_png_base64` (and optional region boxes) to `/predict`; keep all v1 fields. Add original / overlay toggle, opacity slider, and the caption "regions of interest suggested by the model" to the screen.
- **Acceptance:** v1 clients still work unchanged; overlay aligns with the image at any window size; latency still acceptable; tests updated.
- **Human input:** UX review. **Stage 2 sign-off gate.**

---

### STAGE 3: Object detection (YOLO)

#### F10. Detection scoping and box labels 🛑
- **Tasks:** Report the detector license options (Ultralytics AGPL-3.0 vs YOLOX/MMDetection/torchvision) and recommend one. Identify box sources: datasets with boxes, masks converted to boxes (use the `mask_to_yolo` helper in the guide), and in-house images annotated in CVAT/Label Studio. Write an annotation guideline (what counts as a finding, minimum size, how to treat ambiguous cases). Build `data/yolo/` with `images/` and `labels/` for train/val/test (split by patient/video again) and `dataset.yaml`.
- **Acceptance:** format validator passes (every image has a label file; boxes in range; classes valid); box-overlay sample grid saved; counts per class reported; no patient overlap between splits.
- **Human input:** choose the detector library; clinician annotates or reviews abnormal-finding boxes.

#### F11. Detector training (`detector_v3`)
- **Tasks:** Fine-tune a pre-trained detector (smallest size first, e.g. `yolov8n`; scale up only if needed). Log to TensorBoard or the tool's built-in logger, use early stopping, save best weights to `models/detector_v3.pt`.
- **Acceptance:** training completes; curves saved; weights registered in `REGISTRY.md`.

#### F12. Detection evaluation 🛑
- **Tasks:** On the frozen test set: mAP@0.5, mAP@0.5:0.95, per-class precision/recall, PR curves, false alarms per image at the chosen confidence threshold, visual review of predictions. Write `reports/stage3_eval.md`.
- **Acceptance:** report complete with an honest weaknesses list; confidence threshold chosen with the human for high sensitivity at tolerable false alarms.
- **Human input:** approve the target and threshold; clinician review of boxes.

#### F13. Detection API v3 and screen v3
- **Tasks:** Add `POST /detect` returning `image_size` and `detections[{label, confidence, box[x1,y1,x2,y2]}]` plus `model_version` and `disclaimer`. In the UI draw boxes scaled to the display size, list findings with confidence, per-box show/hide, keep the heatmap toggle. Optional: run the classifier first as a gate.
- **Acceptance:** boxes align correctly at any window size and zoom; empty-detection state handled ("no findings above threshold; does not mean the image is normal"); previous endpoints unchanged.
- **Human input:** UX review. **Stage 3 sign-off gate.**

---

### Cross-cutting (finish after Stage 1, complete by the end)

#### F14. Feedback loop, audit log, privacy
- **Tasks:** Agree / Disagree / Correct-label controls; store corrections (image reference, predicted vs corrected label, model version, timestamp) for future retraining with de-identification; audit log (user, time, image ref, model version, output); retention/deletion behavior documented.
- **Acceptance:** feedback persists and exports to the training manifest format; logs contain no patient identifiers; documented.

#### F15. Packaging, hardening, documentation
- **Tasks:** Package the service (PyInstaller or embedded ONNX Runtime); app launches it, polls `/health`, restarts on failure, stops on exit; model version shown in an "About" area; README for setup, retraining, versioning; known-limitations page; robustness tests (blur, dark, glare, resolution changes).
- **Acceptance:** clean-machine install and run works; documented retraining procedure; limitations and disclaimer visible.

---

## 5. Report template (send after every feature)

```
Feature: F<n> <name>
Status: done / blocked / needs decision
What I did: <3-6 bullets>
Evidence: <metrics, test output, screenshots, file paths>
Acceptance criteria: <each criterion → pass/fail>
Decisions made (also logged in DECISIONS.md): <list>
Risks / open questions: <list>
Needs from you: <approvals, downloads, clinician review, or "nothing">
Next: F<n+1> (waiting for approval at gate: yes/no)
```

---

## 6. Definition of done for the whole project

- All acceptance criteria for F0-F15 pass, with evidence in `PROGRESS.md`.
- Stage 1, 2, 3 sign-offs recorded.
- Models versioned in `REGISTRY.md`; test set unchanged since F2/F10.
- Every result view shows the disclaimer.
- Licenses and privacy handling documented.
- A new developer can reproduce training and run the app from the README.