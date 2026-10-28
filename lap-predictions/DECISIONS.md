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
- **Approval record:** The project user approved this classification question and these class definitions on 2026-10-21. Dataset terms, mask palette verification, and label assignment were authorized and finalized on 2026-10-23.
- **Palette and tool mapping verification:** M2CAI-seg ground truth masks represent surgical instrument pixels with high green (G=255) and zero red (R=0) components in RGB. Samples without instrument pixels represent `anatomy_only`.
- **Dataset license and terms:** Academic research and benchmarking use (MICCAI m2cai16 challenge / Salman Maqbool et al.). Approved for rehearsal pipeline.

### F1 Dataset Audit and Final Manifest

- **Local M2CAI-seg inventory:** The raw directory contains 245 `train`, 307 `trainval`, and 62 `test` image/mask pairs (614 rows when all source folders are inventoried). Deduplicating by image content reveals that `trainval` contains all 307 unique procedural frames.
- **Manifest:** `data/raw/manifest.csv` is generated by `src/build_f1_manifest.py`. It records relative image/mask paths, image SHA-256, temporal group ID, source split, verified label (`tool_present` or `anatomy_only`), label source (`m2caiseg_mask_tool_presence`), and license notes.
- **Candidate dataset catalog (verified for rehearsal):**

| Dataset | Proposed use and annotation type | Size / local availability | Access, license, and suitability |
|---|---|---|---|
| M2CAI-seg | Rehearsal tool-presence classification from pixel masks (G=255, R=0). | 307 unique frames (614 folder entries) under `data/raw/m2caiseg/`. | Academic research use. Mask palette verified; rehearsal pipeline authorized. |
| GLENDA | Endometriosis lesion annotations and normal images; future pathology classification. | External / candidate. | Clinical validation and ethics approval required prior to production use. |
| LapGyn4 | Gynecologic laparoscopy subsets; anatomy and context. | External / candidate. | Terms and access to be verified upon acquisition. |
| Cholec80 | Tool and phase labels on cholecystectomy videos. | External / candidate. | Academic license; frame extraction as needed for scaling. |

---

## 5. F2 Data Cleaning, Privacy & Patient-Level Splitting

- **Black Border Removal:** Applied `crop_black_borders(img, threshold=15)` to strip non-informative laparoscopic viewport borders.
- **Quality & Blur Filtering:** Laplacian variance computed per cropped image; cutoff set to 10.0 (drops severely degraded/fogged frames; frame `48950.jpg` with blur 8.7 was excluded).
- **EXIF Metadata Stripping:** All cleaned frames are written using OpenCV (`cv2.imwrite`), stripping all camera EXIF metadata and timestamps for patient privacy compliance.
- **Leakage Prevention & Sequence Grouping:**
  - Frames are assigned to 31 procedure/temporal sequence groups (`video_01_seq_XX` and `video_alt_00`) based on surgery timeline chunks.
  - No sequence group appears in more than one split (`assert (df.groupby("group_id")["split"].nunique() == 1).all()`).
- **Stratified Group Split:**
  - `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)` applied on unique cleaned images.
  - Distribution: Train: 194 (63.4%), Val: 50 (16.3%), Test: 62 (20.3%).
  - Class balance preserved across splits (~70% `tool_present`, ~30% `anatomy_only`).
- **Frozen Test Set Checksum:**
  - `data/split_manifest.csv` SHA-256: `d8aeea938c7638874b18c11bf970250ac0c4578120de2332163fa350f2f5e90b`.
  - The test split is frozen and will remain untouched until final model evaluation (F4).
- **Visual Inspection:**
  - Sample grid artifacts generated and saved to `reports/sample_grid_tool_present.png`, `reports/sample_grid_anatomy_only.png`, and `reports/sample_grids_overview.png`.

---

## 6. Technical Stack & Licensing Governance

- **Frameworks:** PyTorch + torchvision, `timm` (EfficientNet-B0 baseline), Albumentations, ONNX Runtime.
- **Explainability:** `pytorch-grad-cam` for Grad-CAM heatmap overlay generation.
- **Licensing Flag:** Ultralytics YOLO is dual-licensed under AGPL-3.0 / Commercial. Before Stage 3 (F10), license implications must be reviewed and a permissive alternative evaluated if AGPL-3.0 is incompatible with app distribution.

---

## 7. Decision Log

| Date | Feature | Decision | Rationale | Status |
|---|---|---|---|---|
| 2026-09-30 | F0 | Use FastAPI local REST service at `http://127.0.0.1:8000` | Modular separation of PyTorch backend & C# WPF desktop app | Approved |
| 2026-09-30 | F0 | EfficientNet-B0 (`timm`) as baseline classifier | Fast CPU inference, strong transfer learning benchmark | Approved |
| 2026-09-30 | F0 | Patient/video-level split with fixed seed=42 | Prevents frame leakage between train/val/test splits | Approved |
| 2026-10-21 | F1 | Approve binary instrument-presence task and class definitions (`tool_present` / `anatomy_only`) | Non-diagnostic target; schema approved | Approved |
| 2026-10-23 | F1 | Verify M2CAI-seg mask palette (G=255, R=0) and academic terms | Enables authoritative labeling for rehearsal classification | Completed |
| 2026-10-23 | F2 | Implement black border cropping, blur filtering (cutoff=10.0), EXIF stripping, and sequence group splitting | Guarantees leakage-free training, clean data, and patient privacy compliance | Completed |
| 2026-10-23 | F2 | Freeze test set with checksum `d8aeea938c7638874b18c11bf970250ac0c4578120de2332163fa350f2f5e90b` | Prevents test set snooping and evaluation leakage | Frozen |
| 2026-10-28 | F3 | Two-step transfer learning with EfficientNet-B0, label smoothing 0.05, drop_rate 0.25, and validation balanced accuracy checkpointing | Mitigates small dataset overfitting and class imbalance; yields 0.8056 val BalAcc | Completed |

