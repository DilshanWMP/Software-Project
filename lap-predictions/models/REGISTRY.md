# Model Registry (`models/REGISTRY.md`)

This registry tracks all trained, validated, and exported model versions for the **Laparoscopy Anomaly Detection & Predictions System**.

### Versioning Rules
1. **Never overwrite a released model file.**
2. Models are versioned monotonically: `classifier_v1` (Stage 1), `classifier_v2` (Stage 2 Grad-CAM), `detector_v3` (Stage 3 YOLO).
3. Every entry links to training dataset checksums, validation metrics, test set evaluation reports, and runtime configurations.
4. **Safety Mandate:** All model outputs are decision support prototypes requiring qualified clinician review.

---

## Active & Registered Models

### `classifier_v1` (Stage 1 Image Classification)

| Attribute | Specification |
|:---|:---|
| **Model Version** | `classifier_v1` |
| **Stage** | Stage 1 (Binary Instrument-Presence Classification) |
| **Release Date** | 2026-10-28 |
| **Architecture** | EfficientNet-B0 (`efficientnet_b0` in `timm`) |
| **Frameworks** | PyTorch 2.14.0 + ONNX Runtime 1.30.0 |
| **Input Specifications** | Dynamic batch $\times$ 3 $\times$ 224 $\times$ 224 (RGB, ImageNet normalized) |
| **Output Specifications** | Dynamic batch $\times$ 2 (raw unscaled logits) |
| **Classes** | `0: anatomy_only`, `1: tool_present` (saved in `models/classes.json`) |
| **Dataset Version** | M2CAI-seg cleaned rehearsal dataset (306 unique frames) |
| **Split Checksum** | `data/split_manifest.csv` (SHA-256: `d8aeea938c7638874b18c11bf970250ac0c4578120de2332163fa350f2f5e90b`) |
| **Split Counts** | Train: 194 frames (63.4%), Val: 50 frames (16.3%), Test: 62 frames (20.3%) |
| **PyTorch Weights** | [`models/classifier_v1.pt`](classifier_v1.pt) (Best Val BalAcc: 0.8056 at epoch 4 & 6) |
| **Validation Logits** | [`models/classifier_v1_val_logits.pt`](classifier_v1_val_logits.pt) |
| **ONNX Export** | [`models/classifier_v1.onnx`](classifier_v1.onnx) (Opset 17, dynamic batch, 15.27 MB) |
| **ONNX Parity Diff** | Max absolute difference: **$8.40 \times 10^{-6}$** ($< 1.0 \times 10^{-4}$ tolerance passed) |
| **Temperature Calibration** | Platt scaling temperature $T = 4.6722$ (ECE = 0.0970) |
| **Runtime Config** | [`models/config.json`](config.json) (`temperature: 4.6722`, `abnormal_threshold: 0.20 / 0.50`) |
| **Test Evaluation Report** | [`reports/stage1_eval.md`](../reports/stage1_eval.md) |

#### Test Set Performance (`62` held-out frames)
- **Balanced Accuracy:** 0.7150
- **ROC-AUC:** 0.8200 (95% CI [0.6950, 0.9190])
- **PR-AUC (Average Precision):** 0.9567 (95% CI [0.9156, 0.9870])
- **Sensitivity @ Operating Threshold $0.20$:** **0.9200 (92.0%)** (46 / 50 tools detected, 4 missed)
- **Sensitivity @ Default Threshold $0.50$:** 0.6800 (68.0%) (34 / 50 tools detected, 16 missed)
- **Specificity @ Threshold $0.50$:** 0.7500 (75.0%) (9 / 12 anatomy frames correctly cleared)
- **Expected Calibration Error (ECE):** 0.0970 (target $< 0.10$ met)

---
