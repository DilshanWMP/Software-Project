import torch, timm, cv2, onnxruntime, sklearn
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("timm", timm.__version__, "| opencv", cv2.__version__)
m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=3)  # downloads weights
x = torch.randn(1, 3, 224, 224)
print("forward pass OK, output shape:", tuple(m(x).shape))