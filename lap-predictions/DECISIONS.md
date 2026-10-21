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

## 4. F1 Classification Question & Approved Label Schema

- **Approved project classification question:** Given a laparoscopy snapshot, determine whether a surgical instrument is visible in the image. M2CAI-seg masks may be used as the image-level label source only after the authoritative palette mapping and dataset terms are verified.
- **Scope:** This is a binary instrument-presence classification task. It does not classify anatomy, pathology, complications, or whether an image is clinically normal or abnormal. `anatomy_only` means only that no instrument is annotated in the mask; it must not be described as `normal`.
- **Approved project classes:**

  | Label | Definition | Label source |
  |---|---|---|
  | `tool_present` | The ground-truth mask contains at least one pixel belonging to an instrument/tool class. | Corresponding M2CAI-seg mask |
  | `anatomy_only` | The ground-truth mask contains no pixels belonging to an instrument/tool class, assuming the mask is complete for the image. | Corresponding M2CAI-seg mask |

- **Uncertain or unusable samples:** Do not force a class when the image/mask pair is missing, corrupt, ambiguous, or the mask does not permit a reliable instrument-presence decision. Record these for review and exclude them from labeled classification data unless an annotator resolves them.
- **Single-label rule:** Assign exactly one of the two labels per usable image. The target is presence/absence of any visible instrument, not instrument type or count.
- **Approval record:** The project user approved this classification question and these class definitions on 2026-10-21. This approves the project schema only; it is not clinician validation or dataset-use authorization.
- **Remaining gate:** Confirm the dataset's mask palette and annotation completeness against authoritative documentation, and verify the dataset's license/access terms before assigning labels, training a model, or using the data for a shipped feature.

### F1 Dataset Audit and Provisional Catalog

- **Local M2CAI-seg inventory:** The raw directory contains 245 `train`, 307 `trainval`, and 62 `test` image/mask pairs (614 rows when all source folders are inventoried). Masks are RGB pixel masks; dimensions vary across the set (30 at 716x402, 384 at 596x334, and 200 at 774x434), with each image/mask pair validated to have matching dimensions. The exact palette-to-class mapping and annotation completeness have not been verified against authoritative dataset documentation, so no image labels are inferred in the F1 manifest.
- **Split caution:** The raw folders overlap by image ID: `train`/`trainval` share 245 IDs (242 same-ID pairs are byte-identical), `trainval`/`test` share 62 IDs (all byte-identical), and `train`/`test` share 3 IDs (none byte-identical). Treat `trainval` as overlapping source data, not an independent split; resolve the 3 same-name/different-content collisions before downstream use. Do not treat these folder names as a leakage-safe patient/video split. Patient/video identifiers are not available in the current file layout.
- **Manifest:** `data/raw/manifest.csv` is generated by `src/build_f1_manifest.py`. It records relative image/mask paths, image SHA-256, raw source split, and annotation type. Labels and label sources remain blank until the mask mapping and dataset-use terms are confirmed. Patient/video IDs are blank when not provided by source metadata.
- **Candidate dataset catalog (provisional; no license or access terms approved):**

| Dataset | Proposed use and annotation type (per project guide) | Size / local availability | Access, license, and suitability |
|---|---|---|---|
| M2CAI-seg | Candidate instrument-presence labels from pixel masks, only if the authoritative palette confirms tool classes. | 614 image/mask folder entries across overlapping source splits; present under `data/raw/m2caiseg/`. | Terms and access not verified. Palette and completeness need authoritative review; current split folders cannot be treated as independent. |
| GLENDA | Endometriosis lesion annotations and normal images; pathology classification, not the current tool-presence draft. | Size not verified; not found in local raw data. | Terms/access not verified; confirm official requirements before acquisition or use. |
| LapGyn4 | Anatomy, action, and instrument subsets; possible contextual or instrument data if labels fit the approved task. | Size not verified; not found in local raw data. | Terms/access not verified; confirm official requirements before acquisition or use. |
| Cholec80 | Tool and phase labels on cholecystectomy videos; a different surgical domain. | Project guide states 80 videos; not found in local raw data. | Terms/access not verified; confirm official requirements and suitability before acquisition or use. |

- **Catalog source note:** Dataset descriptions above are copied as candidate leads from `Predictions screen cnn guide.MD`, not verified licensing or acquisition facts. Before acquisition or training, record authoritative source documentation, dataset version, permitted purpose, redistribution restrictions, and access requirements.

---

## 5. Technical Stack & Licensing Governance

- **Frameworks:** PyTorch + torchvision, `timm` (EfficientNet-B0 baseline), Albumentations, ONNX Runtime.
- **Explainability:** `pytorch-grad-cam` for Grad-CAM heatmap overlay generation.
- **Licensing Flag:** Ultralytics YOLO is dual-licensed under AGPL-3.0 / Commercial. Before Stage 3 (F10), license implications must be reviewed and a permissive alternative (e.g. Torchvision RetinaNet / Faster R-CNN or YOLOX) evaluated if AGPL-3.0 is incompatible with app distribution.

---

## 6. Decision Log

| Date | Feature | Decision | Rationale | Status |
|---|---|---|---|---|
| 2026-09-30 | F0 | Use FastAPI local REST service at `http://127.0.0.1:8000` | Modular separation of PyTorch backend & C# WPF desktop app | Approved |
| 2026-09-30 | F0 | EfficientNet-B0 (`timm`) as baseline classifier | Fast CPU inference, strong transfer learning benchmark | Approved |
| 2026-09-30 | F0 | Patient/video-level split with fixed seed=42 | Prevents frame leakage between train/val/test splits | Approved |
| 2026-10-21 | F1 | Approve binary instrument-presence task and class definitions (`tool_present` / `anatomy_only`) | Non-diagnostic target; dataset palette, completeness, and use terms remain unverified | Approved (schema only) |
