"""
Environment and Model Verification Test
Checks core packages and performs a dry-run forward pass with EfficientNet-B0.
"""

import torch
import timm
import cv2
import fastapi
import onnxruntime
import sklearn


def verify():
    print("=== Environment Verification ===")
    print(f"PyTorch version     : {torch.__version__}")
    print(f"CUDA available      : {torch.cuda.is_available()}")
    print(f"timm version        : {timm.__version__}")
    print(f"OpenCV version      : {cv2.__version__}")
    print(f"FastAPI version     : {fastapi.__version__}")
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")

    print("\n=== Testing Pretrained Backbone Forward Pass ===")

    # pretrained=True may download the weights if they are not cached.
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=3,
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 3), (
        f"Unexpected output shape: {tuple(output.shape)}"
    )

    print(
        f"Forward pass successful. "
        f"Output tensor shape: {tuple(output.shape)}"
    )
    print("Environment is READY for Stage 1.")


if __name__ == "__main__":
    verify()