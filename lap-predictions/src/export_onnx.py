import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
from pathlib import Path
import json
import numpy as np
import torch
import timm
import onnx
import onnxruntime as ort

def export_onnx(model_path: str, out_path: str, opset: int = 17, tolerance: float = 1e-4):
    model_path = Path(model_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading PyTorch checkpoint from: {model_path}")
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    arch = ckpt["arch"]

    model = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    print(f"Exporting {arch} to ONNX (opset {opset}, dynamic batch) at: {out_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    # 1. Check ONNX model structure
    print("Validating exported ONNX model structure with onnx.checker...")
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure check passed.")

    # 2. Check numerical parity across multiple batch sizes
    print("Checking numerical parity with ONNX Runtime...")
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    for batch_size in [1, 2, 4, 8]:
        test_input = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            torch_out = model(test_input).numpy()

        ort_inputs = {"input": test_input.numpy()}
        ort_out = session.run(["output"], ort_inputs)[0]

        max_diff = float(np.max(np.abs(torch_out - ort_out)))
        mean_diff = float(np.mean(np.abs(torch_out - ort_out)))
        print(f"  Batch size {batch_size:2d} -> Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

        assert max_diff < tolerance, f"Parity check failed for batch {batch_size}: {max_diff} >= {tolerance}"

    print(f"\nParity verification PASSED (all batch sizes < {tolerance:e} tolerance).")
    print(f"Successfully saved ONNX model to: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
    return str(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export classifier_v1 PyTorch checkpoint to ONNX")
    parser.add_argument("--model",     default="../../models/classifier_v1.pt", help="Input PyTorch checkpoint path")
    parser.add_argument("--out",       default="../../models/classifier_v1.onnx", help="Output ONNX file path")
    parser.add_argument("--opset",     type=int, default=17, help="ONNX opset version")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Maximum allowed discrepancy")
    args = parser.parse_args()

    export_onnx(args.model, args.out, opset=args.opset, tolerance=args.tolerance)

