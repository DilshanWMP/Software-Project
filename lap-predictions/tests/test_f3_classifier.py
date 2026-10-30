import unittest
import json
from pathlib import Path
import numpy as np
import torch
import timm

import sys
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "predictions" / "src"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset import AlbImageFolder, train_tf, eval_tf
from train import make_loaders

class TestF3Classifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = REPO_ROOT / "data" / "processed"
        cls.models_dir = REPO_ROOT / "models"
        cls.checkpoint_path = cls.models_dir / "classifier_v1.pt"
        cls.logits_path = cls.models_dir / "classifier_v1_val_logits.pt"
        cls.classes_path = cls.models_dir / "classes.json"

    def test_transforms_shapes_and_types(self):
        dummy_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        tr = train_tf(image=dummy_img)["image"]
        ev = eval_tf(image=dummy_img)["image"]

        self.assertIsInstance(tr, torch.Tensor)
        self.assertIsInstance(ev, torch.Tensor)
        self.assertEqual(tr.shape, (3, 224, 224))
        self.assertEqual(ev.shape, (3, 224, 224))
        self.assertEqual(tr.dtype, torch.float32)

    def test_dataset_loading(self):
        train_path = self.data_dir / "train"
        self.assertTrue(train_path.exists(), f"{train_path} does not exist")
        ds = AlbImageFolder(str(train_path), train_tf)
        self.assertGreater(len(ds), 0)
        self.assertEqual(set(ds.classes), {"anatomy_only", "tool_present"})

        sample_tensor, label = ds[0]
        self.assertEqual(sample_tensor.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])

    def test_make_loaders(self):
        train_loader, val_loader, classes = make_loaders(str(self.data_dir), bs=8)
        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)
        self.assertEqual(classes, ["anatomy_only", "tool_present"])

        for x, y in train_loader:
            self.assertEqual(x.shape[1:], (3, 224, 224))
            self.assertEqual(len(y), x.shape[0])
            break

    def test_model_forward_pass(self):
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        dummy_input = torch.randn(2, 3, 224, 224)
        out = model(dummy_input)
        self.assertEqual(out.shape, (2, 2))

    def test_checkpoint_saved_and_valid(self):
        self.assertTrue(self.checkpoint_path.exists(), "classifier_v1.pt checkpoint does not exist")
        ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        for key in ["state", "classes", "arch", "val_acc", "epoch", "seed"]:
            self.assertIn(key, ckpt)
        self.assertEqual(ckpt["classes"], ["anatomy_only", "tool_present"])
        self.assertGreaterEqual(ckpt["val_acc"], 0.70)
        
        # Test loading weights and executing forward pass
        model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["classes"]))
        model.load_state_dict(ckpt["state"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.randn(1, 3, 224, 224))
        self.assertEqual(logits.shape, (1, 2))

    def test_val_logits_saved(self):
        self.assertTrue(self.logits_path.exists(), "classifier_v1_val_logits.pt does not exist")
        data = torch.load(str(self.logits_path), map_location="cpu", weights_only=False)
        self.assertIn("logits", data)
        self.assertIn("targets", data)
        self.assertIn("classes", data)
        self.assertEqual(data["logits"].shape[0], 50)  # 50 val samples
        self.assertEqual(data["targets"].shape[0], 50)

    def test_classes_json(self):
        self.assertTrue(self.classes_path.exists(), "classes.json does not exist")
        with open(self.classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        self.assertEqual(classes, ["anatomy_only", "tool_present"])

if __name__ == "__main__":
    unittest.main()
