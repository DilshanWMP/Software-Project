# COMPLETE_AGENT_INSTRUCTIONS.md: Agent Guidelines for the Full Predictions Screen Build

You are a coding agent building the **Predictions screen** for a desktop app that captures laparoscopy video and snapshots. A doctor uploads a snapshot; ML models suggest abnormalities. The build has three stages, then release:

```
Stage 1 classification (T0-T14) → Stage 2 Grad-CAM heatmap (T15-T18) → Stage 3 YOLO detection (T19-T24) → Release (T25-T27)
```

**Read first:** `COMPLETE_IMPLEMENTATION_GUIDE.md` (design, code, metrics). This file says **how to work**: one task at a time, with acceptance criteria and human gates. Do not start a stage before the previous stage's sign-off.

---

## 1. Ground rules (non-negotiable)

1. **One task at a time, in order.** Finish a task (all acceptance criteria met, evidence shown) before starting the next. Never work ahead.
2. **Transfer learning only.** Fine-tune pre-trained models; never train from scratch.
3. **Split by patient/video, never by image.** The classification test set is frozen at T4; the detection test split is frozen at T20. Each is used **once** for final evaluation (T10, T23). Never tune on it.
4. **Do not invent facts.** No made-up dataset URLs, file layouts, metrics, library APIs or results. Report only numbers you actually computed. If unsure, check official docs or ask.
5. **No patient data in the repo, logs, commits or chat.** `data/`, `models/*.pt`, `models/*.onnx`, `runs/` stay git-ignored. Strip metadata. Do not upload real patient snapshots to cloud services without the human confirming ethics approval/consent.
6. **Safety wording is mandatory.** Every user-visible result shows: "Decision support only. Requires clinician review." Never call output a diagnosis. Heatmaps are "regions of interest suggested by the model", not lesion boundaries.
7. **Check licences.** Record every dataset's source, licence and terms in `DECISIONS.md`. Flag **Ultralytics AGPL-3.0** before Stage 3 and get a human decision on the detector library.
8. **Additive changes only.** Version models (`classifier_v1`, `detector_v3`...); never overwrite a released model. Add API fields; never rename or remove them.
9. **Reproducible.** Fixed seeds, pinned `requirements.txt`, configs in files, no hard-coded absolute paths.
10. **Existing app first.** Inspect the desktop app before integrating; match its conventions; don't refactor unrelated code.
11. **Honest reporting.** Report failures, weak classes and limits plainly. If a success criterion isn't met, say so.
12. **Small commits.** One commit per completed task, message `T<n>: <what>`. If you create commits, end them with the attribution lines your environment specifies.

---

## 2. Working protocol (every task)

1. **Read** the task block and the matching guide section.
2. **Plan** in 3-8 bullets in `PROGRESS.md`.
3. **Implement.**
4. **Verify** against every acceptance criterion; run tests; show real outputs (tables, logs, screenshots), not claims.
5. **Update** `PROGRESS.md` (status, evidence), `DECISIONS.md` (choices + reasons), `EXPERIMENTS.md` (training runs).
6. **Report** using the template in Section 5. **Stop and wait** at every 🛑 gate.

### Stop and ask the human when
- A dataset needs registration, a signed agreement, or a manual download.
- Classes, label mappings, success targets, thresholds or the detector library must be decided.
- Clinician input is needed (label review, heatmap/box review, error review).
- Real patient or in-house data would be used.
- A criterion can't be met after reasonable attempts (say what you tried).
- Any licensing, privacy or safety concern appears.
- A change would significantly alter the existing app's architecture.

---

## 3. Tech stack

| Purpose | Tool |
|---|---|
| Framework | PyTorch + torchvision |
| Classifiers | `timm` (EfficientNet-B0 first; ResNet-50; optional EndoViT) |
| Augmentation | Albumentations |
| Heatmaps | `grad-cam` (pytorch-grad-cam) |
| Detection | Ultralytics YOLO **or** a permissive alternative (decided at T19) |
| Data/metrics | pandas, scikit-learn, matplotlib, OpenCV, Pillow |
| Frames | FFmpeg |
| Annotation | CVAT or Label Studio (self-hosted) |
| Export/runtime | ONNX, ONNX Runtime |
| Service | FastAPI + Uvicorn (bind `127.0.0.1`) |
| Tracking | TensorBoard or `EXPERIMENTS.md` |
| Packaging | PyInstaller or ONNX Runtime embedded in the app |
| Training compute | Google Colab / Kaggle (free GPU), weights saved to Drive |

Layout: `data/{raw,clean,processed,yolo}/`, `src/`, `service/`, `models/`, `reports/`, `tests/`, `PROGRESS.md`, `DECISIONS.md`, `EXPERIMENTS.md`, `requirements.txt`.

---

## 4. Task list

### STAGE 1: Image classification

#### T0. Discovery and scaffold
- **Do:** Inspect the desktop app (language, UI framework, snapshot storage path/format/resolution, overlays/black borders, how screens are added, ability to call HTTP/start processes, target PC specs). Create layout, `.gitignore`, `requirements.txt`, tracking files, `/health` stub. Verify the environment (`torch`, `timm`, a pretrained forward pass). Document the Colab/Kaggle GPU dry run.
- **Accept:** App stack in `DECISIONS.md`; environment installs from scratch; `/health` returns ok; first commit made.
- **Human input:** confirm the integration approach (default: local FastAPI service; ONNX kept for embedding later).

#### T1. Success criteria, classes, data sources 🛑
- **Do:** Propose classes (start binary normal/abnormal; fine classes only where data supports them). Propose numeric targets (guide A1). List candidate datasets with licence, size, label type, access steps and which class each feeds.
- **Accept:** `DECISIONS.md` has approved classes, approved targets and a dataset table. Gated datasets are marked "needs human download".
- **Human input:** approve classes and targets; download gated data; confirm ethics/consent for in-house images.

#### T2. Rehearsal on m2caiSeg
- **Do:** Inspect `data/raw/m2caiseg` (structure, split names, image/mask counts, file-name patterns, mask encoding). Build a manifest preserving the given splits. Derive a **proxy** image-level label from masks (confirmed with the human). Check that no video crosses splits. Run T5-T8 on it.
- **Accept:** Manifest and split tables saved; leakage check passes; rehearsal model trains and evaluates; the report states this is a **pipeline test, not an anomaly model**.
- **Human input:** confirm the proxy label.

#### T3. Manifest and label mapping (real data)
- **Do:** Organize `data/raw/<source>/<label>/<group>__<frame>.jpg`; extract video frames (~1 fps); write `build_manifest.py`; map raw labels to approved classes (`normal` only when genuinely normal); cap oversize normal sets per video.
- **Accept:** `data/manifest.csv` complete; counts per source/class reported; every mapping decision logged.

#### T4. Clean, split, freeze 🛑
- **Do:** Crop black borders; drop unreadable/tiny/blurry frames (choose the cutoff by inspecting the lowest-scoring images); remove images with on-screen text; strip metadata. `StratifiedGroupKFold` by `group_id` (~70/15/15). Write `data/processed/{train,val,test}/<class>/` and `split_manifest.csv`. Produce sample grids.
- **Accept:** Automated test: no `group_id` in more than one split. Tables saved for images **and groups** per class/split. Classes with <~10 groups or 1-2 groups in val/test flagged. Test set declared frozen (checksum logged).
- **Human input:** review grids; decide what to do about thin classes.

#### T5. Training pipeline
- **Do:** Implement `dataset.py` and `train.py` per the guide (two-step fine-tuning, mixed precision, balanced sampling, early stopping on balanced accuracy, seeded, CLI args). Unit-test the loader.
- **Accept:** Runs on a small local subset and on Colab/Kaggle GPU; logs saved; checkpoint stores weights, classes and architecture.

#### T6. Baseline (`classifier_v1`)
- **Do:** Train EfficientNet-B0; evaluate on **val only**; log in `EXPERIMENTS.md`; save weights to Drive.
- **Accept:** Curves saved; train/val gap reported; no unexplained collapse (a class never predicted).

#### T7. Experiments
- **Do:** One change at a time: ResNet-50 (optional EndoViT), input size, augmentation, fixes for observed weaknesses (guide B6). Final candidates over 3 seeds. Select by validation per-class recall and balanced accuracy.
- **Accept:** `EXPERIMENTS.md` lists every run; chosen config justified; test set untouched.

#### T8. Calibration and thresholds
- **Do:** Fit temperature on val; choose the abnormal threshold to meet the agreed sensitivity; choose the "uncertain" cutoff; save `models/config.json`.
- **Accept:** Reliability curve and ECE before/after saved; values documented.

#### T9. Grad-CAM shortcut check 🛑
- **Do:** Overlays for 50-100 val images, including **correct** predictions; assess whether attention is on tissue or on instruments/borders/text/glare/dataset cues.
- **Accept:** Gallery and written assessment saved. If shortcuts exist, fix data, retrain from T4/T5, repeat.
- **Human input:** clinician or human reviews the gallery.

#### T10. Final classification test (once) 🛑
- **Do:** Run the frozen test set once with the chosen model, temperature and thresholds. Report per-class precision/recall/F1, binary sensitivity/specificity, ROC-AUC, PR-AUC, confusion matrix, calibration, bootstrap 95% CIs by patient/video, external-source results if available, and the 20 worst errors (false negatives first). Write `reports/stage1_eval.md` with a candid weaknesses section.
- **Accept:** Every criterion marked met / not met with the measured number. **No retuning on test.**
- **Human input:** go/no-go.

#### T11. Export and registry
- **Do:** ONNX export (opset 17, dynamic batch); verify parity with PyTorch; write `classes.json`; add `REGISTRY.md` entry.
- **Accept:** Parity test passes; older versions untouched.

#### T12. API `/predict`
- **Do:** Implement `GET /health` and `POST /predict` (JPEG/PNG only, size limit, preprocessing identical to training, calibrated softmax, "uncertain" below cutoff, `model_version` and `disclaimer`). Bind `127.0.0.1`.
- **Accept:** Tests for valid, wrong-type, corrupted, oversized and tiny images; parity test against offline evaluation; ≤ ~2 s on CPU; schema documented.

#### T13. Predictions screen v1 🛑
- **Do:** Add the screen: picker (and session snapshots if easy), preview, async Analyze with progress indicator, ranked probability bars, "uncertain" state, permanent disclaimer, error states (service down, bad file, timeout).
- **Accept:** Works end to end on real app snapshots; UI never freezes; screenshots attached.
- **Human input:** UX review. **Stage 1 sign-off.**

#### T14. Stage 1 hardening
- **Do:** Robustness tests (blur, dark, glare, resolution change, screenshot vs raw frame); README for setup/retraining/versioning; limitations page.
- **Accept:** Tests pass; docs reproduce training and run.

### STAGE 2: Classification + heatmap

#### T15. Grad-CAM engine
- **Do:** Implement `gradcam.py` (`gradcam_overlay`, `heat_to_boxes`) for the chosen CNN; benchmark added latency.
- **Accept:** Unit tests (output shape, range, alignment with input); latency measured.

#### T16. Heatmap review 🛑
- **Do:** Generate galleries on 100+ images, correct and incorrect predictions. Document shortcut findings. Fix and retrain if needed (back through T4-T11 as required).
- **Accept:** Clinician reviews ~50 overlays and judges them sensible; shortcut fixes verified.
- **Human input:** clinician review.

#### T17. API v2
- **Do:** Add `heatmap_png_base64` and `regions` to `/predict`, optional `?heatmap=false`. Keep all v1 fields.
- **Accept:** v1 clients unchanged (regression test); latency acceptable; schema updated.

#### T18. Screen v2 🛑
- **Do:** Original/overlay toggle, opacity slider, caption "regions of interest suggested by the model".
- **Accept:** Overlay aligns at any window size; screenshots attached.
- **Human input:** UX review. **Stage 2 sign-off.**

### STAGE 3: Object detection

#### T19. Detection scoping 🛑
- **Do:** Report detector licence options (Ultralytics AGPL-3.0 vs YOLOX/MMDetection/torchvision) with a recommendation. Identify box sources. Write an annotation guideline.
- **Accept:** Decision recorded in `DECISIONS.md`.
- **Human input:** choose the detector library; arrange annotation by a clinician.

#### T20. Box dataset
- **Do:** Build `data/yolo/` (`images/`, `labels/`, `dataset.yaml`) from converted masks, datasets with boxes, and annotated in-house images. Split by patient/video; keep test patients consistent with Stage 1 where possible. Freeze the detection test split.
- **Accept:** Validator passes (every image has a label file, boxes in range, classes valid); overlay grid saved; per-class counts reported; no patient overlap; checksum logged.
- **Human input:** clinician reviews abnormal-finding boxes.

#### T21. Detector training (`detector_v3`)
- **Do:** Fine-tune a pre-trained detector, smallest size first; early stopping; save weights; register in `REGISTRY.md`.
- **Accept:** Training completes; curves saved; experiments logged.

#### T22. Detector improvement
- **Do:** Error analysis on val; add data where it fails (active learning); tune resolution/augmentation; choose the confidence threshold on val for high sensitivity.
- **Accept:** Documented iterations; test split untouched.

#### T23. Detection final test (once) 🛑
- **Do:** mAP@0.5, mAP@0.5:0.95, per-class precision/recall, PR curves, false alarms per image at the chosen threshold, visual review of predictions. Write `reports/stage3_eval.md` with weaknesses.
- **Accept:** Each target met / not met with the real number; threshold agreed.
- **Human input:** approve targets and threshold; clinician reviews boxes.

#### T24. API v3 and screen v3 🛑
- **Do:** Add `POST /detect` (`image_size`, `detections[{label, confidence, box}]`, `model_version`, `disclaimer`). In the UI draw boxes scaled to the display size, list findings with confidence, per-box show/hide, keep the heatmap toggle. Empty-result message: "No findings above threshold. This does not mean the image is normal."
- **Accept:** Boxes align at any window size/zoom; earlier endpoints unchanged; tests updated.
- **Human input:** UX review. **Stage 3 sign-off.**

### RELEASE

#### T25. Feedback loop, audit log, privacy
- **Do:** Agree/Disagree/Correct-label (and box edits) controls; store de-identified corrections (`image_ref, model_version, predicted, corrected, timestamp`); audit log (user, time, image ref, model version, output); document retention and deletion.
- **Accept:** Feedback persists and exports in the training-manifest format; logs contain no patient identifiers.

#### T26. Packaging and operations
- **Do:** Package the service (PyInstaller or embedded ONNX Runtime); app launches it, polls `/health`, restarts on failure, stops on exit; show model versions in an About area; write the retraining and rollback procedure; drift-monitoring notes (uncertain rate, override rate).
- **Accept:** Clean-machine install and run works; procedures documented.

#### T27. Final documentation and handover 🛑
- **Do:** README (setup, retraining, versioning, API reference), limitations page, licence and privacy summary, regulatory note (research prototype until validated and approved).
- **Accept:** A new developer can reproduce training and run the app from the README.
- **Human input:** final sign-off.

---

## 5. Report template (after every task)

```
Task: T<n> <name>
Status: done / blocked / needs decision
What I did: <3-6 bullets>
Evidence: <metrics, tables, logs, screenshots, file paths>
Acceptance criteria: <each criterion → pass / fail>
Decisions made (also logged in DECISIONS.md): <list>
Risks / open questions: <list>
Needs from you: <approvals, downloads, clinician review, or "nothing">
Next: T<n+1> (waiting at gate: yes/no)
```

---

## 6. Definition of done

- All acceptance criteria for T0-T27 pass, with evidence in `PROGRESS.md`.
- Success criteria reported as met or not met with real numbers and confidence intervals.
- Each test set used exactly once; split-manifest checksums unchanged since T4 and T20.
- All models versioned in `REGISTRY.md`; ONNX parity verified; API changes additive only.
- Every result view shows the disclaimer; heatmaps and boxes are labelled as suggestions.
- Licences (datasets, Ultralytics) and privacy handling documented.
- Human sign-offs recorded for Stage 1 (T13), Stage 2 (T18), Stage 3 (T24) and final handover (T27).