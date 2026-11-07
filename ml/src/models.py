"""timm backbones + ONNX export wrappers (internal/AI_CONTRACT.md section 1).

Findings wrapper:  input RGB 0..1 [1,3,S,S] -> (logits [1,C], cam [1,C,h,w])
Gate wrapper:      input RGB 0..1 [1,3,G,G] -> logits [1,2]

ImageNet mean/std normalisation happens inside the wrapper (so inside the ONNX graph).
CAM is exact for backbones whose head is  GAP -> (dropout) -> Linear:
    logits[c] = mean_hw(cam[c]) + bias[c]
`check_cam_exact` verifies that numerically.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Architectures we support. cam_ok = head is GAP -> dropout -> Linear (exact CAM).
ARCHS = {
    "efficientnet_b0": {"cam_ok": True},
    "efficientnet_b2": {"cam_ok": True},
    "efficientnet_b3": {"cam_ok": True},
    "resnet50": {"cam_ok": True},
    # MobileNetV3 head is GAP -> 1x1 conv -> act -> Linear, so CAM is NOT exact.
    # Only used for the gate, which has no cam output.
    "mobilenetv3_small_100": {"cam_ok": False},
}


def build_model(arch: str, num_classes: int, pretrained: bool = True,
                drop_rate: float = 0.2) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"Unsupported arch {arch}; choose from {sorted(ARCHS)}")
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes,
                             drop_rate=drop_rate)


def _classifier(model: nn.Module) -> nn.Linear:
    head = model.get_classifier()
    if not isinstance(head, nn.Linear):
        raise TypeError(f"Expected nn.Linear classifier, got {type(head)}")
    return head


class _Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class FindingsExport(nn.Module):
    """RGB 0..1 -> (logits, cam). Model must be in eval mode."""

    def __init__(self, model: nn.Module):
        super().__init__()
        arch = getattr(model, "default_cfg", {}).get("architecture", "")
        if arch and arch in ARCHS and not ARCHS[arch]["cam_ok"]:
            raise ValueError(f"{arch} head is not GAP->Linear; CAM would not be exact")
        self.norm = _Normalize()
        self.model = model
        self.fc = _classifier(model)

    def forward(self, x):
        feats = self.model.forward_features(self.norm(x))          # [B,K,h,w]
        logits = self.model.forward_head(feats)                      # GAP->drop->fc
        cam = torch.einsum("ck,bkhw->bchw", self.fc.weight, feats)  # no bias, no relu
        return logits, cam


class GateExport(nn.Module):
    """RGB 0..1 -> logits [B,2] (0 = not_endoscopy, 1 = endoscopy)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.norm = _Normalize()
        self.model = model

    def forward(self, x):
        return self.model(self.norm(x))


class Normalized(nn.Module):
    """Training-time wrapper: same normalisation as the exported graph."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.norm = _Normalize()
        self.model = model

    def forward(self, x):
        return self.model(self.norm(x))


@torch.no_grad()
def check_cam_exact(model: nn.Module, size: int = 224, tol: float = 1e-4) -> float:
    """Return max |logits - (mean_hw(cam) + bias)|; raise if above tol."""
    model.eval()
    wrap = FindingsExport(model).eval()
    x = torch.rand(2, 3, size, size)
    logits, cam = wrap(x)
    recon = cam.mean(dim=(2, 3)) + wrap.fc.bias
    diff = (logits - recon).abs().max().item()
    if diff > tol:
        raise AssertionError(f"CAM not exact: max diff {diff}")
    return diff


if __name__ == "__main__":
    for a, info in ARCHS.items():
        m = build_model(a, 9, pretrained=False)
        if info["cam_ok"]:
            print(a, "cam exact, max diff", check_cam_exact(m))
        else:
            print(a, "gate-only (no CAM)")
