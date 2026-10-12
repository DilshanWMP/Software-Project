"""
Environment and Model Verification Test
Checks core packages and performs a dry-run forward pass with EfficientNet-B0.
"""
import torch
import timm
import cv2
import fastapi

def verify():
    print("=== Environment Verification ===")
    print(f"PyTorch version : {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    print(f"timm version    : {timm.__version__}")
    print(f"OpenCV version  : {cv2.__version__}")
    print(f"FastAPI version : {fastapi.__version__}")

    print("\n=== Testing Pretrained Backbone Forward Pass ===")
    # Instantiate EfficientNet-B0 backbone (3 dummy classes for testing)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=3)
    model.eval()

    # Create dummy batch (BatchSize=1, Channels=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 3), f"Unexpected output shape: {output.shape}"
    print(f"Forward pass successful. Output tensor shape: {output.shape}")
    print("Environment is READY for Stage 1.")

if __name__ == "__main__":
    verify()
