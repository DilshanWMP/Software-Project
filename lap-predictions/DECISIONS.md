# DECISIONS.md: Architectural & Technical Decision Log

This document tracks all architectural choices, data governance policies, licensing checks, dataset selections, and technical decisions made during the development of the **Laparoscopy Anomaly Detection & Predictions System**.

---

## 1. Desktop Application Discovery (F0)

- **Application Stack:**
  - **Language / Framework:** C# .NET 9.0-windows WPF desktop application (`EndoscopyApp.csproj`).
  - **MVVM Architecture:** Uses `CommunityToolkit.Mvvm` (v8.4.0) and `MaterialDesignThemes` (v4.9.0).
  - **Video & Image Capture:** Integrated via `OpenCvSharp4` (`VideoCaptureService.cs`).
  - **Database & Media Storage:** SQLite database (`endoscopy.db`) managed via `Microsoft.Data.Sqlite` (`DatabaseService.cs`).
  - **Media Cataloging:** `MediaFiles` SQLite table tracks `PatientId`, `FilePath`, `FileType`, `CreatedAt`. Snapshots are persisted to local disk with database records.
- **Predictions Screen Location:**
  - The Predictions UI will be integrated into `PatientMediaView` / `MediaViewerView` as an "Analyze Snapshot" panel/modal, with potential for a top-level tab in `MainWindow`.

---

## 2. Integration & Deployment Architecture (F0)

- **Primary Integration Mechanism:** Local FastAPI microservice running on `127.0.0.1:8000`.
  - **Rationale:** Decouples the Python ML inference runtime (PyTorch / `timm` / OpenCV) from the C# WPF frontend. Enables rapid model updates and heatmaps generation without recompiling the WPF desktop app.
- **Export / Embedded Alternative:** ONNX format (opset 17) for execution via `Microsoft.ML.OnnxRuntime` in C# for offline/embedded deployment.

---

## 3. Data Governance & Safety Rules

- **Patient Privacy & Anonymization:**
  - Zero real patient data in git commits or public repositories.
  - `data/` directory, model checkpoints (`models/*.pt`), and training runs are strictly git-ignored.
  - Image EXIF metadata is stripped upon dataset ingestion.
- **Data Splitting Policy:**
  - Data MUST be split by **patient/video ID**, never by frame, to prevent data leakage.
  - Fixed random seed = `42`.
  - Split distribution: **70% Train / 15% Validation / 15% Test**.
  - The Test set is frozen immediately after creation and touched ONLY for final evaluation.
- **Safety Disclaimer Requirement:**
  - Every inference result displayed to a user MUST display:
    > *"Decision support only. Requires clinician review."*
  - Outputs are never framed as definitive diagnoses.
- **API Versioning:**
  - Additive changes only to API contracts.
  - Versioned model artifacts (`classifier_v1`, `classifier_v2`, `detector_v3`).

---

## 4. Technical Stack & Licensing Governance

- **Frameworks:** PyTorch + torchvision, `timm` (EfficientNet-B0 baseline), Albumentations, ONNX Runtime.
- **Explainability:** `pytorch-grad-cam` for Grad-CAM heatmap overlay generation.
- **Licensing Flag:** Ultralytics YOLO is dual-licensed under AGPL-3.0 / Commercial. Before Stage 3 (F10), license implications must be reviewed and a permissive alternative (e.g. Torchvision RetinaNet / Faster R-CNN or YOLOX) evaluated if AGPL-3.0 is incompatible with app distribution.

---

## 5. Decision Log

| Date | Feature | Decision | Rationale | Status |
|---|---|---|---|---|
| 2026-09-30 | F0 | Use FastAPI local REST service at `http://127.0.0.1:8000` | Modular separation of PyTorch backend & C# WPF desktop app | Approved |
| 2026-09-30 | F0 | EfficientNet-B0 (`timm`) as baseline classifier | Fast CPU inference, strong transfer learning benchmark | Approved |
| 2026-09-30 | F0 | Patient/video-level split with fixed seed=42 | Prevents frame leakage between train/val/test splits | Approved |
