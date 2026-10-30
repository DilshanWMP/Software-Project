# PROGRESS.md: Feature Roadmap & Progress Log

Tracking progress for the **Laparoscopy Anomaly Detection & Predictions System** across Stage 1 (Classification), Stage 2 (Grad-CAM Explainability), Stage 3 (YOLO Detection), and Cross-Cutting features.

---

## Roadmap Overview

| Stage | Focus | Status | Features Completed |
|---|---|---|---|
| **Stage 1** | Image Classification (F0 - F7) | In Progress | 6 / 8 Completed |
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
- **Status:** `DONE`
- **Plan:**
  1. Define the target classification question and class schema: binary instrument presence (`tool_present` / `anatomy_only`).
  2. Audit candidate data, verify mask palette (`(G==255) & (R==0)` in RGB/BGR represents tool markers), and confirm academic research terms.
  3. Implement `src/build_f1_manifest.py` with automatic label assignment, group derivation, SHA-256 verification, and unit tests (`tests/test_build_f1_manifest.py`).
- **Evidence:**
  - `python src/build_f1_manifest.py` output: 614 rows cataloged across source folders; 428 labeled `tool_present` and 186 labeled `anatomy_only`. All image/mask pairs validated.
  - Manifest written to `data/raw/manifest.csv`.
  - `python -m unittest tests/test_build_f1_manifest.py` executed: 5 passed (100% pass rate).
- **Acceptance Criteria:**
  - [x] `DECISIONS.md` has an approved class list and verified dataset catalog
  - [x] Raw data is cataloged in `data/raw/manifest.csv`; manifest structure and image/mask pairs validate
  - [x] Human approval obtained for the class list
  - [x] Dataset terms and any gated-access approval verified

---

#### F2. Data Preparation and Patient-Level Split
- **Status:** `DONE`
- **Plan:**
  1. Crop black laparoscopic camera borders using `crop_black_borders(img, threshold=15)`.
  2. Filter blurry/corrupted frames using Laplacian variance threshold (cutoff=10.0); re-save with OpenCV to strip EXIF metadata for privacy compliance.
  3. Derive 31 procedure/temporal sequence groups (`video_01_seq_XX` and `video_alt_00`) to prevent data leakage.
  4. Perform `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)` ensuring 70/15/15 split.
  5. Organize processed data into `data/processed/{train,val,test}/{tool_present,anatomy_only}/`.
  6. Save `data/split_manifest.csv` and freeze the test split with logged SHA-256 checksum.
  7. Generate class sample grids (`reports/sample_grid_*.png`) and verify zero leakage via `tests/test_f2_split.py`.
- **Evidence:**
  - `python src/prepare_dataset.py` run output:
    - Cleaned and valid frames: 306 (1 dropped for severe blur: `48950.jpg`, blur=8.70).
    - Unique sequence groups: 31.
    - Zero group leakage assertion: `[LEAKAGE CHECK] PASSED: 0 group overlap across train, val, and test.`
    - Final split distribution:
      - **Train:** 194 images (68 `anatomy_only`, 126 `tool_present`) — 63.4%
      - **Val:** 50 images (13 `anatomy_only`, 37 `tool_present`) — 16.3%
      - **Test:** 62 images (12 `anatomy_only`, 50 `tool_present`) — 20.3%
    - Frozen split manifest: `data/split_manifest.csv` (SHA-256: `d8aeea938c7638874b18c11bf970250ac0c4578120de2332163fa350f2f5e90b`).
    - Visual inspection artifacts saved: `reports/sample_grid_tool_present.png`, `reports/sample_grid_anatomy_only.png`, `reports/sample_grids_overview.png`.
  - `python -m unittest tests/test_f2_split.py` output: 5 passed in 0.036s.
  - Overall test suite `python -m unittest discover -s tests`: 10 passed in 0.058s.
- **Acceptance Criteria:**
  - [x] Automated test verifies 0 patient/group overlap across train/val/test splits (`tests/test_f2_split.py`)
  - [x] Class distribution table generated and recorded
  - [x] Visual sample-grid per class saved in `reports/`
  - [x] Test set declared frozen with logged SHA-256 checksum

---

#### F3. Baseline Classifier (`classifier_v1`)
- **Status:** `DONE`
- **Plan:**
  1. Implement `src/dataset.py` with Albumentations data augmentations (flips, rotations, contrast/brightness, color jitter, blur, coarse dropout) and ImageNet normalization.
  2. Build `src/train.py` for two-step transfer learning with EfficientNet-B0 (`timm`): Step A head-only warm-up with frozen backbone, Step B full fine-tuning with Cosine Annealing LR and WeightedRandomSampler.
  3. Track training and validation loss, balanced accuracy, and generalization gap using TensorBoard (`runs/classifier_v1`).
  4. Save model checkpoint (`models/classifier_v1.pt`), class index mapping (`models/classes.json`), and validation logits (`models/classifier_v1_val_logits.pt`) for downstream temperature calibration.
  5. Validate implementation with automated unit tests in `tests/test_f3_classifier.py`.
- **Evidence:**
  - `python -u predictions/src/train.py --data data/processed --out models/classifier_v1.pt --bs 16 --head_epochs 5 --epochs 20 --patience 6 --seed 42` executed:
    - Step A (Head): 5 epochs completed, validation loss stabilized at 1.010, val BalAcc = 0.650.
    - Step B (Fine-tuning): reached peak validation balanced accuracy of **0.8056** (80.6%) at epoch 4 & 6.
    - Early stopping triggered at epoch 10 after 6 epochs without validation score improvement.
    - Checkpoint successfully persisted: `models/classifier_v1.pt` (Val BalAcc: 0.8056, epoch: 4, seed: 42).
    - Validation logits stored: `models/classifier_v1_val_logits.pt` (50 validation instances with ground-truth targets).
    - Class metadata written: `models/classes.json` (`["anatomy_only", "tool_present"]`).
    - TensorBoard run logs saved under `runs/classifier_v1/`.
  - Comprehensive unit test `tests/test_f3_classifier.py` executed: 7 tests passed in 0.24s.
  - Full test suite `python -m unittest discover -s tests`: 17 passed in 0.47s (100% pass rate).
- **Acceptance Criteria:**
  - [x] Training script completes end-to-end
  - [x] Checkpoint saved as `models/classifier_v1.pt` with class mapping
  - [x] No severe overfitting observed (validation balanced accuracy = 0.8056)

---

#### F4. Evaluation Report 🛑
- **Status:** `DONE (Awaiting Sign-off Gate 🛑)`
- **Plan:**
  1. Evaluate `classifier_v1` on the frozen test set (62 images: 50 `tool_present`, 12 `anatomy_only`).
  2. Measure Balanced Accuracy, per-class Precision/Recall/F1, ROC-AUC, PR-AUC, Sensitivity, and Specificity with bootstrap 95% confidence intervals.
  3. Perform Platt-style temperature calibration using validation logits (T=4.6722) and evaluate Expected Calibration Error (ECE).
  4. Generate and save diagnostic plots: `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`, `reliability_curve.png`, and top-20 `failure_gallery.png`.
  5. Compile full evaluation report in `reports/stage1_eval.md` and persist runtime calibration parameters to `models/config.json`.
- **Evidence:**
  - `python predictions/src/evaluate.py --data data/processed --model models/classifier_v1.pt --reports reports` executed:
    - Balanced Accuracy: 0.7150
    - ROC-AUC: **0.8200** (95% CI [0.6950, 0.9190])
    - PR-AUC: **0.9567** (95% CI [0.9156, 0.9870])
    - Sensitivity @ threshold 0.50: 0.6800 (34/50); Sensitivity @ operating threshold 0.20: **0.9200** (46/50)
    - Specificity @ threshold 0.50: 0.7500 (9/12)
    - ECE: 0.0970 (< 0.10 target met)
    - False Negatives: 16 (at 0.50 threshold) / 4 (at 0.20 threshold)
    - False Positives: 3 (at 0.50 threshold) / 8 (at 0.20 threshold)
    - Generated plots: `reports/confusion_matrix.png`, `reports/roc_curve.png`, `reports/pr_curve.png`, `reports/reliability_curve.png`, `reports/failure_gallery.png`.
    - Generated report: `reports/stage1_eval.md`.
    - Service configuration: `models/config.json`.
- **Acceptance Criteria:**
  - [x] Comprehensive evaluation report generated (`reports/stage1_eval.md`)
  - [ ] Sensitivity target and operating threshold agreed with clinician/human stakeholder 🛑

---

#### F5. Export and Model Registry
- **Status:** `DONE`
- **Plan:**
  1. Export `classifier_v1` checkpoint to ONNX format (opset 17, dynamic batch).
  2. Validate graph integrity with `onnx.checker` and verify numerical parity against PyTorch across multiple batch sizes (tolerance < 1e-4).
  3. Initialize and maintain `models/REGISTRY.md` with version metadata, lineage, dataset checksums, and evaluation metrics.
  4. Validate export and registry through automated unit tests in `tests/test_f5_export.py`.
- **Evidence:**
  - `python src/export_onnx.py --model models/classifier_v1.pt --out models/classifier_v1.onnx --opset 17 --tolerance 1e-4` executed:
    - Opset: 17, dynamic batch dimension `[batch_size, 3, 224, 224]` -> `[batch_size, 2]`.
    - `onnx.checker.check_model` passed without structural violations.
    - ONNX Runtime numerical parity evaluated:
      - Batch 1: Max diff = 1.19e-06
      - Batch 2: Max diff = 6.44e-06
      - Batch 4: Max diff = 8.05e-06
      - Batch 8: Max diff = 8.40e-06
      - All batch tests passed well within 1.00e-04 tolerance.
    - Model artifact saved: `models/classifier_v1.onnx` (15.27 MB).
    - Registry documentation authored: `models/REGISTRY.md`.
  - Comprehensive unit test `tests/test_f5_export.py` executed: 5 tests passed.
  - Full test suite `python -m unittest discover -s tests`: 22 passed in 1.09s (100% pass rate).
- **Acceptance Criteria:**
  - [x] Parity test passes (<1e-4 tolerance)
  - [x] `models/REGISTRY.md` updated
  - [x] Old versions untouched

---

#### F6. Inference API v1: `/predict`
- **Status:** `DONE`
- **Plan:**
  1. Implement `service/main.py` POST /predict endpoint with ONNX Runtime primary inference and PyTorch fallback.
  2. Temperature-scaled softmax probabilities (T=4.6722), ABNORMAL_THRESHOLD decision logic, uncertain flag.
  3. Input validation: MIME type check, 15 MB size limit, corruption detection.
  4. Full JSON response schema: model_version, top_label, confidence, probabilities, finding_detected, decision_threshold, uncertain, image_size, inference_time_ms, disclaimer.
  5. GET /health and GET /info endpoints.
  6. CORS middleware, 9 unit tests, and API README.
- **Evidence:**
  - `service/main.py` fully implemented with ONNX Runtime + PyTorch fallback.
  - `tests/test_f6_service.py`: 9 tests pass (health, info, valid JPEG, valid PNG, real dataset image, 415, 400 corrupted, 400 empty, 413 oversized).
  - Full test suite: 31 tests pass in under 1s.
  - Sample real prediction: top_label=anatomy_only, confidence=0.7301, inference_time_ms=9.79ms.
  - CORS fixed: allow_credentials=False with allow_origins=["*"].
  - API schema documented in `README.md`.
- **Acceptance Criteria:**
  - [x] Validated on valid/corrupted/oversized images (9 test cases pass)
  - [x] Response time < 2s on CPU (measured ~10 ms via ONNX Runtime)
  - [x] Schema fully documented in README

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
