# PROGRESS.md: Feature Roadmap & Progress Log

Tracking progress for the **Laparoscopy Anomaly Detection & Predictions System** across Stage 1 (Classification), Stage 2 (Grad-CAM Explainability), Stage 3 (YOLO Detection), and Cross-Cutting features.

---

## Roadmap Overview

| Stage | Focus | Status | Features Completed |
|---|---|---|---|
| **Stage 1** | Image Classification (F0 - F7) | In Progress | 1 / 8 Completed |
| **Stage 2** | Classification + Grad-CAM Heatmaps (F8 - F9) | Planned | 0 / 2 Completed |
| **Stage 3** | Object Detection (F10 - F13) | Planned | 0 / 4 Completed |
| **Cross-Cutting** | Audit, Packaging & Hardening (F14 - F15) | Planned | 0 / 2 Completed |

---

## Detailed Feature Roadmap

### STAGE 1: Image Classification

#### F0. Discovery and Scaffold
- **Status:** `DONE`
- **Plan:**
  1. Inspect desktop application (`EndoscopyApp`), stack (C# .NET 9 WPF), database (`endoscopy.db`), and snapshot handling.
  2. Create repository layout (`data/`, `src/`, `service/`, `models/`, `reports/`, `tests/`).
  3. Set up and verify environment (`requirements.txt`, `tests/check_env.py`).
  4. Implement `/health` REST stub endpoint using FastAPI & Uvicorn (`service/main.py`).
  5. Document integration decisions in `DECISIONS.md` and progress roadmap in `PROGRESS.md`.
- **Evidence:**
  - `python tests/check_env.py` output: `torch 2.14.0+cpu | CUDA available: False`, `timm 1.0.30 | opencv 5.0.0`, `forward pass OK, output shape: (1, 3)`.
  - Service `/health` endpoint verified: returns `{"status": "ok"}` when queried at `http://127.0.0.1:8000/health`.
  - Desktop app architecture documented in `DECISIONS.md`.
- **Acceptance Criteria:**
  - [x] App stack documented in `DECISIONS.md`
  - [x] Environment installs from scratch & verified
  - [x] `/health` stub responds on `http://127.0.0.1:8000/health`
  - [x] Predictions screen location identified in desktop app

---

#### F1. Label Schema and Data Acquisition 🛑
- **Status:** `IN PROGRESS` — class schema approved; provisional catalog and validated raw manifest implemented. Official mask-palette and dataset-term verification remain pending.
- **Plan:**
  1. Define the target classification question and class schema based on available annotations. Approved project target: binary instrument presence (`tool_present` / `anatomy_only`), not normal/abnormal classification; see `DECISIONS.md`.
  2. Audit candidate data against the question and record a provisional dataset catalog; verify authoritative license/access terms and mask palette before use.
  3. Generate and validate `data/raw/manifest.csv` with `python lap-predictions/src/build_f1_manifest.py` from the repository root. Labels remain blank pending verified palette mapping and dataset-use terms.
- **Acceptance Criteria:**
  - [ ] `DECISIONS.md` has an approved class list and verified dataset catalog
  - [x] Raw data is cataloged in `data/raw/manifest.csv`; manifest structure and image/mask pairs validate
  - [x] Human approval obtained for the class list
  - [ ] Dataset terms and any gated-access approval verified
- **Audit findings:** Local M2CAI-seg folders contain overlapping image IDs (`trainval` overlaps the named `train`/`test` folders, and `train`/`test` also overlap). These must not be treated as independent splits. No patient/video IDs or authoritative license/palette information were found in the local folder layout.
- **Blocked:** Do not assign labels or mark F1 complete until the official mask palette, annotation completeness, and dataset use terms are confirmed. The manifest generator does not infer labels.

---

#### F2. Data Preparation and Patient-Level Split
- **Status:** `PLANNED`
- **Plan:**
  1. Extract frames (~1 fps), crop black borders, remove blurry/black frames.
  2. Map labels to target schema and strip EXIF metadata for privacy compliance.
  3. Perform patient/video-level split (70/15/15) with seed=42 into `data/processed/`.
- **Acceptance Criteria:**
  - [ ] Automated test verifies 0 patient overlap across train/val/test splits
  - [ ] Class distribution table generated
  - [ ] Visual sample-grid per class saved

---

#### F3. Baseline Classifier (`classifier_v1`)
- **Status:** `PLANNED`
- **Plan:**
  1. Implement `src/dataset.py` with Albumentations data augmentations.
  2. Build `src/train.py` for two-step transfer learning with EfficientNet-B0 (`timm`).
  3. Log training and validation loss/metrics using TensorBoard.
- **Acceptance Criteria:**
  - [ ] Training script completes end-to-end
  - [ ] Checkpoint saved as `models/classifier_v1.pt` with class mapping
  - [ ] No severe overfitting observed

---

#### F4. Evaluation Report 🛑
- **Status:** `PLANNED`
- **Plan:**
  1. Evaluate `classifier_v1` on frozen test set (Sensitivity, Specificity, Precision, F1, ROC-AUC).
  2. Generate confusion matrix and failure case gallery in `reports/stage1_eval.md`.
- **Acceptance Criteria:**
  - [ ] Comprehensive evaluation report generated
  - [ ] Sensitivity target agreed with clinician/human stakeholder

---

#### F5. Export and Model Registry
- **Status:** `PLANNED`
- **Plan:**
  1. Export `classifier_v1` to ONNX (opset 17, dynamic batch).
  2. Verify numerical parity between PyTorch and ONNX Runtime outputs.
  3. Update `models/REGISTRY.md` with version metadata.
- **Acceptance Criteria:**
  - [ ] Parity test passes (<1e-4 tolerance)
  - [ ] `models/REGISTRY.md` updated

---

#### F6. Inference API v1: `/predict`
- **Status:** `PLANNED`
- **Plan:**
  1. Extend `service/main.py` with `POST /predict` supporting JPEG/PNG upload.
  2. Return softmax probabilities, confidence thresholding (`uncertain`), model version, and mandatory disclaimer.
- **Acceptance Criteria:**
  - [ ] Validated on valid/corrupted/oversized images
  - [ ] Response time <2s on CPU
  - [ ] Schema fully documented in README

---

#### F7. Predictions Screen v1 🛑
- **Status:** `PLANNED`
- **Plan:**
  1. Create Predictions UI component in `EndoscopyApp` connected to `POST /predict`.
  2. Implement async request handling, progress bar, probability visualization, and safety disclaimer.
- **Acceptance Criteria:**
  - [ ] End-to-end functionality working on real app snapshots
  - [ ] Non-blocking UI execution
  - [ ] Stage 1 human sign-off gate completed

---

### STAGE 2: Classification + Heatmap

- [ ] **F8. Grad-CAM Engine (`classifier_v2`)** - `PLANNED`
- [ ] **F9. API v2 & Screen v2 (Heatmap Overlay)** - `PLANNED`

---

### STAGE 3: Object Detection (YOLO)

- [ ] **F10. Detection Scoping & Box Labels 🛑** - `PLANNED`
- [ ] **F11. Detector Training (`detector_v3`)** - `PLANNED`
- [ ] **F12. Detection Evaluation 🛑** - `PLANNED`
- [ ] **F13. Detection API v3 & Screen v3** - `PLANNED`

---

### Cross-Cutting

- [ ] **F14. Feedback Loop & Audit Logging** - `PLANNED`
- [ ] **F15. Packaging & Hardening** - `PLANNED`
