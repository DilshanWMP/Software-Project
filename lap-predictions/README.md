# lap-predictions — Inference Service API

Offline FastAPI microservice for laparoscopy snapshot anomaly detection.  
Runs at `http://127.0.0.1:8000`. No internet required; model runs entirely on CPU.

---

## Quick Start

```powershell
# 1. Activate virtual environment
cd lap-predictions
.venv\Scripts\Activate.ps1

# 2. Start the service
uvicorn service.main:app --host 127.0.0.1 --port 8000

# 3. Test (in another terminal)
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@snapshot.jpg;type=image/jpeg"
```

---

## Endpoints

### `GET /health`

Returns service liveness and model runtime status.

**Response:**
```json
{
  "status": "ok",
  "model_version": "classifier_v1",
  "runtime": "onnxruntime"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` (model loaded) or `"degraded"` (model missing) |
| `model_version` | string | Active model identifier |
| `runtime` | string | `"onnxruntime"`, `"pytorch"`, or `"none"` |

---

### `GET /info`

Returns model metadata and calibration configuration.

**Response:**
```json
{
  "model_version": "classifier_v1",
  "classes": ["anatomy_only", "tool_present"],
  "temperature": 4.6722,
  "abnormal_threshold": 0.5,
  "uncertain_below": 0.6,
  "input_dimensions": [224, 224, 3],
  "runtime": "onnxruntime",
  "disclaimer": "Decision support only. Requires clinician review."
}
```

---

### `POST /predict`

Upload a laparoscopy snapshot (JPEG or PNG) and receive a classification prediction.

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Field name: `file`
- Accepted formats: `image/jpeg`, `image/png`
- Max size: **15 MB**

**Example (curl):**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@snapshot.jpg;type=image/jpeg"
```

**Example (Python):**
```python
import httpx

with open("snapshot.jpg", "rb") as f:
    resp = httpx.post(
        "http://127.0.0.1:8000/predict",
        files={"file": ("snapshot.jpg", f, "image/jpeg")}
    )
print(resp.json())
```

**Response (200 OK):**
```json
{
  "model_version": "classifier_v1",
  "top_label": "anatomy_only",
  "confidence": 0.7301,
  "probabilities": {
    "anatomy_only": 0.7301,
    "tool_present": 0.2699
  },
  "finding_detected": false,
  "decision_threshold": 0.5,
  "uncertain": false,
  "image_size": [416, 334],
  "inference_time_ms": 9.79,
  "disclaimer": "Decision support only. Requires clinician review."
}
```

**Response Schema:**

| Field | Type | Description |
|---|---|---|
| `model_version` | string | Model identifier from `models/config.json` |
| `top_label` | string | Top predicted class (or `"uncertain"` if confidence < `uncertain_below`) |
| `confidence` | float | Probability of top class after temperature scaling [0–1] |
| `probabilities` | object | Per-class softmax probabilities (temperature-scaled, sum to 1.0) |
| `finding_detected` | boolean | `true` if `P(tool_present) >= decision_threshold` |
| `decision_threshold` | float | Threshold used for `finding_detected` (from `models/config.json`) |
| `uncertain` | boolean | `true` if top confidence < `uncertain_below` (low-confidence flag) |
| `image_size` | [int, int] | Original uploaded image dimensions `[width, height]` |
| `inference_time_ms` | float | Wall-clock inference time in milliseconds |
| `disclaimer` | string | **Always present:** `"Decision support only. Requires clinician review."` |

**Error Responses:**

| Status | Condition |
|---|---|
| `400 Bad Request` | Empty file or corrupted/unreadable image |
| `413 Content Too Large` | File exceeds 15 MB limit |
| `415 Unsupported Media Type` | File is not JPEG or PNG |
| `503 Service Unavailable` | Model not loaded (checkpoint file missing) |

---

## Model Configuration (`models/config.json`)

```json
{
  "temperature": 4.6722,
  "abnormal_threshold": 0.5,
  "uncertain_below": 0.6,
  "classes": ["anatomy_only", "tool_present"],
  "model_version": "classifier_v1"
}
```

| Key | Description |
|---|---|
| `temperature` | Temperature scaling factor (T=4.6722, calibrated on val set, ECE=0.097) |
| `abnormal_threshold` | `P(tool_present)` threshold for `finding_detected = true` |
| `uncertain_below` | Top confidence below this → label becomes `"uncertain"` |

---

## Model Artifacts

| File | Description |
|---|---|
| `models/classifier_v1.onnx` | Primary inference model (ONNX opset 17, 15.27 MB) |
| `models/classifier_v1.pt` | PyTorch checkpoint (fallback if ONNX not available) |
| `models/classes.json` | Class list: `["anatomy_only", "tool_present"]` |
| `models/config.json` | Calibration and threshold configuration |
| `models/REGISTRY.md` | Full model versioning registry |

---

## Performance (CPU)

| Metric | Value |
|---|---|
| Inference latency (ONNX) | ~10 ms per image |
| Test ROC-AUC | 0.82 (95% CI [0.695, 0.919]) |
| Test PR-AUC | 0.957 |
| Sensitivity @ th=0.20 | 92.0% (46/50) |
| ECE (calibration error) | 0.097 (< 0.10 target) |

---

> **Safety Notice:** This service is a **decision support prototype only**.  
> It has not undergone clinical validation and must not be used for medical diagnosis.  
> All results require clinician review before any clinical action is taken.

