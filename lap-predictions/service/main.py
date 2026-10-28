import io
import json
import time
import os
import sys
from pathlib import Path
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Resolve project directories
SERVICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVICE_DIR.parent
MODELS_DIR = REPO_ROOT / "models"

# Constants
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DISCLAIMER_TEXT = "Decision support only. Requires clinician review."

# Load Configuration and Classes
config_path = MODELS_DIR / "config.json"
classes_path = MODELS_DIR / "classes.json"

if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
else:
    CONFIG = {
        "temperature": 1.0,
        "abnormal_threshold": 0.5,
        "uncertain_below": 0.6,
        "classes": ["anatomy_only", "tool_present"],
        "model_version": "classifier_v1",
    }

CLASSES = CONFIG.get("classes", ["anatomy_only", "tool_present"])
MODEL_VERSION = CONFIG.get("model_version", "classifier_v1")
TEMPERATURE = float(CONFIG.get("temperature", 1.0))
ABNORMAL_THRESHOLD = float(CONFIG.get("abnormal_threshold", 0.5))
UNCERTAIN_BELOW = float(CONFIG.get("uncertain_below", 0.6))

# Load Model: Prefer ONNX Runtime, Fallback to PyTorch
onnx_path = MODELS_DIR / f"{MODEL_VERSION}.onnx"
pt_path = MODELS_DIR / f"{MODEL_VERSION}.pt"

ort_session = None
pt_model = None

if onnx_path.exists():
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        print(f"[Service] Loaded ONNX model from: {onnx_path}")
    except Exception as e:
        print(f"[Service] Could not initialize ONNX Runtime ({e}), falling back to PyTorch.")

if ort_session is None and pt_path.exists():
    try:
        import torch
        import timm
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        pt_model = timm.create_model(ckpt.get("arch", "efficientnet_b0"), pretrained=False, num_classes=len(CLASSES))
        pt_model.load_state_dict(ckpt["state"])
        pt_model.eval()
        print(f"[Service] Loaded PyTorch model from: {pt_path}")
    except Exception as e:
        print(f"[Service] Could not load PyTorch checkpoint: {e}")

# Initialize FastAPI App
app = FastAPI(
    title="Laparoscopy Predictions Service",
    description="Offline microservice for Laparoscopy anomaly detection and instrument presence classification.",
    version="1.0.0",
)

# CORS Middleware (allows desktop app / local browser communication)
# NOTE: allow_credentials=False required when allow_origins=["*"]; browsers
# reject credentialed requests to wildcard origins (CORS spec §3.2.5).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Resize, convert to float RGB in [0, 1], normalize, and transpose to NCHW."""
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    rgb_arr = np.array(img_resized, dtype=np.float32) / 255.0  # HWC

    norm = (rgb_arr - MEAN) / STD
    tensor = np.expand_dims(norm.transpose(2, 0, 1), axis=0).astype(np.float32)  # 1CHW
    return tensor, rgb_arr


def run_model_inference(tensor_np: np.ndarray) -> np.ndarray:
    """Run model inference via ONNX Runtime or PyTorch."""
    if ort_session is not None:
        outputs = ort_session.run(["output"], {"input": tensor_np})
        return outputs[0]  # shape (1, 2)
    elif pt_model is not None:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(tensor_np)
            logits = pt_model(t).numpy()
        return logits
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference model is not loaded. Ensure classifier checkpoint exists in models/."
        )


@app.get("/health")
def health():
    """Health check endpoint."""
    model_loaded = (ort_session is not None) or (pt_model is not None)
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_version": MODEL_VERSION,
        "runtime": "onnxruntime" if ort_session is not None else ("pytorch" if pt_model is not None else "none"),
    }


@app.get("/info")
def info():
    """Return model metadata, classes, and calibration settings."""
    return {
        "model_version": MODEL_VERSION,
        "classes": CLASSES,
        "temperature": TEMPERATURE,
        "abnormal_threshold": ABNORMAL_THRESHOLD,
        "uncertain_below": UNCERTAIN_BELOW,
        "input_dimensions": [IMG_SIZE, IMG_SIZE, 3],
        "runtime": "onnxruntime" if ort_session is not None else "pytorch",
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict instrument/anomaly presence from an uploaded laparoscopy snapshot."""
    # 1. Validate file extension and MIME type
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{ext or file.content_type}'. Please upload a JPEG or PNG image.",
        )

    # 2. Read bytes and enforce size limits
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file size ({len(contents) / (1024*1024):.1f} MB) exceeds maximum allowed size of 15MB.",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )

    # 3. Decode image and check corruption
    try:
        raw_img = Image.open(io.BytesIO(contents))
        raw_img.verify()
        # Re-open after verify as per PIL documentation
        img = Image.open(io.BytesIO(contents))
        orig_w, orig_h = img.size
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image file: {str(e)}",
        )

    # 4. Preprocess
    start_time = time.perf_counter()
    tensor_np, _ = preprocess_image(img)

    # 5. Forward Pass
    logits = run_model_inference(tensor_np)[0]  # shape (num_classes,)

    # 6. Temperature-scaled softmax probabilities
    t = max(0.01, TEMPERATURE)
    scaled_logits = logits / t
    # Numerically stable softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)
    prob_dict = {cls_name: float(p) for cls_name, p in zip(CLASSES, probs)}

    # 7. Classification Decision Logic
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    top_label = CLASSES[top_idx]

    # Evaluate positive finding
    tool_idx = CLASSES.index("tool_present") if "tool_present" in CLASSES else 1
    p_finding = float(probs[tool_idx])
    finding_detected = bool(p_finding >= ABNORMAL_THRESHOLD)

    # Low-confidence thresholding
    is_uncertain = bool(top_prob < UNCERTAIN_BELOW)
    if is_uncertain:
        top_label = "uncertain"

    inference_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "model_version": MODEL_VERSION,
        "top_label": top_label,
        "confidence": round(top_prob, 4),
        "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
        "finding_detected": finding_detected,
        "decision_threshold": ABNORMAL_THRESHOLD,
        "uncertain": is_uncertain,
        "image_size": [orig_w, orig_h],
        "inference_time_ms": round(inference_ms, 2),
        "disclaimer": DISCLAIMER_TEXT,
    }
