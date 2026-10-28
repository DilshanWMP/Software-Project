import unittest
import json
from pathlib import Path
import numpy as np
import torch
import timm
import onnx
import onnxruntime as ort

import sys
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "predictions" / "src"))

from export_onnx import export_onnx

class TestF5Export(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = REPO_ROOT / "models"
        cls.ckpt_path = cls.models_dir / "classifier_v1.pt"
        cls.onnx_path = cls.models_dir / "classifier_v1.onnx"
        cls.config_path = cls.models_dir / "config.json"
        cls.classes_path = cls.models_dir / "classes.json"
        cls.registry_path = cls.models_dir / "REGISTRY.md"

    def test_onnx_model_exists_and_valid(self):
        self.assertTrue(self.onnx_path.exists(), f"ONNX file not found at {self.onnx_path}")
        onnx_model = onnx.load(str(self.onnx_path))
        onnx.checker.check_model(onnx_model)

        # Check graph input and output shapes/names
        graph = onnx_model.graph
        self.assertEqual(len(graph.input), 1)
        self.assertEqual(graph.input[0].name, "input")
        self.assertEqual(len(graph.output), 1)
        self.assertEqual(graph.output[0].name, "output")

    def test_onnx_runtime_dynamic_batch(self):
        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        for bs in [1, 3, 5]:
            dummy = np.random.randn(bs, 3, 224, 224).astype(np.float32)
            out = session.run(["output"], {"input": dummy})[0]
            self.assertEqual(out.shape, (bs, 2))

    def test_pytorch_onnx_parity(self):
        self.assertTrue(self.ckpt_path.exists(), f"PyTorch checkpoint not found at {self.ckpt_path}")
        ckpt = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)

        model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"]))
        model.load_state_dict(ckpt["state"])
        model.eval()

        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])

        np.random.seed(42)
        for bs in [1, 2, 4]:
            x_np = np.random.randn(bs, 3, 224, 224).astype(np.float32)
            with torch.no_grad():
                torch_out = model(torch.from_numpy(x_np)).numpy()
            ort_out = session.run(["output"], {"input": x_np})[0]

            max_diff = float(np.max(np.abs(torch_out - ort_out)))
            self.assertLess(max_diff, 1e-4, f"Parity difference {max_diff} exceeds 1e-4 tolerance for batch size {bs}")

    def test_classes_and_config_metadata(self):
        self.assertTrue(self.classes_path.exists(), "classes.json missing")
        self.assertTrue(self.config_path.exists(), "config.json missing")

        with open(self.classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.assertEqual(classes, ["anatomy_only", "tool_present"])
        self.assertEqual(config["classes"], classes)
        self.assertEqual(config["model_version"], "classifier_v1")
        self.assertIn("temperature", config)
        self.assertIn("abnormal_threshold", config)

    def test_registry_documentation(self):
        self.assertTrue(self.registry_path.exists(), "REGISTRY.md missing")
        content = self.registry_path.read_text(encoding="utf-8")
        self.assertIn("classifier_v1", content)
        self.assertIn("classifier_v1.onnx", content)
        self.assertIn("efficientnet_b0", content)
        self.assertIn("0.8200", content)

if __name__ == "__main__":
    unittest.main()

