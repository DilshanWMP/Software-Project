# Stage 1 Final Evaluation Report

**Generated:** 2026-10-28 20:27  

**Evaluation split:** frozen test set  

**Safety note:** Decision support only. Requires clinician review.

---

## Model & Config

- Architecture: `efficientnet_b0`
- Checkpoint: `classifier_v1.pt`
- Training seed: 42
- Best Val Bal Acc (training): 0.8056

---

## Overall Metrics

| Metric | Value | 95 % CI |
|--------|-------|---------|
| Balanced Accuracy | 0.7150 | — |
| ROC-AUC | 0.8200 | [0.6950, 0.9190] |
| PR-AUC (AP) | 0.9567 | [0.9156, 0.9870] |
| Sensitivity @ threshold 0.50 | 0.6800 | — |
| Specificity @ threshold 0.50 | 0.7500 | — |
| ECE (calibration) | 0.0970 | — |
| Temperature (T) | 4.6722 | — |

---

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
    |-------|-----------|--------|----|---------|
    | anatomy_only | 0.360 | 0.750 | 0.486 | 12 |
| tool_present | 0.919 | 0.680 | 0.782 | 50 |

---

## Confusion Matrix

```
[[ 9  3]
 [16 34]]
```

---

## Threshold & Decision Config

- Target sensitivity ≥ 0.9
- Chosen threshold: **0.500**

```json
    {
  "temperature": 4.672170162200928,
  "abnormal_threshold": 0.5,
  "uncertain_below": 0.6,
  "classes": [
    "anatomy_only",
    "tool_present"
  ],
  "model_version": "classifier_v1"
}
    ```

---

## Plots

- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`
- `reliability_curve.png`
- `failure_gallery.png`

---

## Failure Analysis

- Total False Negatives (missed tool): **16**
- Total False Positives (false alarm): **3**
- See `failure_gallery.png` for the 20 worst errors (FN first).

---

## Weaknesses & Limitations

- Dataset is small (306 unique frames, 62 test images). Confidence intervals are wide.
- Sequences (groups) are temporally derived from frame numbers — true patient-level
  splits are unavailable in M2CAI-seg; leakage across procedures is possible.
- Model is trained on cholecystectomy frames only; generalisation to gynaecologic
  laparoscopy (the clinical use-case) is unknown and likely degraded.
- `anatomy_only` class is under-represented (~30 %) — sensitivity/specificity are
  estimated on a skewed test set.
- This is a rehearsal / pipeline test, not a validated anomaly detector.
  The model predicts **instrument presence**, not clinical pathology.
- No external-source evaluation was performed (no second dataset available).
- **Clinician sign-off is required before Stage 2 or any clinical use.**

---

## Acceptance Criteria

| Criterion | Target | Measured | Pass? |
|-----------|--------|----------|-------|
| Sensitivity (tool_present) | ≥ 0.90 | 0.680 | ❌ |
| ROC-AUC | ≥ 0.80 | 0.820 | ✅ |
| Calibration (ECE < 0.10) | < 0.10 | 0.097 | ✅ |
| Confusion matrix saved | yes | yes | ✅ |
| Failure gallery (top-20) | yes | yes | ✅ |

---
