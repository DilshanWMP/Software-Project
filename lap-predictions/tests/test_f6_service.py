import io
import unittest
from pathlib import Path
from PIL import Image
import numpy as np

import sys
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "service"))

from fastapi.testclient import TestClient
from main import app, MAX_FILE_SIZE, DISCLAIMER_TEXT

class TestF6Service(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.data_dir = REPO_ROOT / "data" / "processed" / "test"

    def _create_dummy_image_bytes(self, fmt="JPEG", size=(300, 300)):
        img = Image.fromarray(np.random.randint(0, 256, (*size, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        buf.seek(0)
        return buf.getvalue()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model_version"], "classifier_v1")

    def test_info_endpoint(self):
        resp = self.client.get("/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_version"], "classifier_v1")
        self.assertEqual(data["classes"], ["anatomy_only", "tool_present"])
        self.assertIn("temperature", data)
        self.assertIn("abnormal_threshold", data)
        self.assertEqual(data["disclaimer"], DISCLAIMER_TEXT)

    def test_predict_valid_jpeg(self):
        img_bytes = self._create_dummy_image_bytes("JPEG")
        resp = self.client.post(
            "/predict",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Schema checks
        self.assertEqual(data["model_version"], "classifier_v1")
        self.assertIn(data["top_label"], ["anatomy_only", "tool_present", "uncertain"])
        self.assertIsInstance(data["confidence"], float)
        self.assertGreater(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertIn("probabilities", data)
        self.assertEqual(set(data["probabilities"].keys()), {"anatomy_only", "tool_present"})

        prob_sum = sum(data["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

        self.assertIn("finding_detected", data)
        self.assertIn("image_size", data)
        self.assertEqual(data["image_size"], [300, 300])
        self.assertEqual(data["disclaimer"], DISCLAIMER_TEXT)

        # Performance check (< 2000 ms)
        self.assertLess(data["inference_time_ms"], 2000.0)

    def test_predict_valid_png(self):
        img_bytes = self._create_dummy_image_bytes("PNG")
        resp = self.client.post(
            "/predict",
            files={"file": ("test.png", img_bytes, "image/png")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["model_version"], "classifier_v1")

    def test_predict_real_dataset_image(self):
        real_img_path = next((self.data_dir / "tool_present").glob("*.jpg"), None)
        if real_img_path is not None:
            with open(real_img_path, "rb") as f:
                real_bytes = f.read()
            resp = self.client.post(
                "/predict",
                files={"file": (real_img_path.name, real_bytes, "image/jpeg")},
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["model_version"], "classifier_v1")
            self.assertIn("probabilities", data)

    def test_predict_unsupported_media_type(self):
        resp = self.client.post(
            "/predict",
            files={"file": ("notes.txt", b"plain text content", "text/plain")},
        )
        self.assertEqual(resp.status_code, 415)

    def test_predict_corrupted_image(self):
        resp = self.client.post(
            "/predict",
            files={"file": ("corrupt.jpg", b"not a real jpeg header", "image/jpeg")},
        )
        self.assertEqual(resp.status_code, 400)

    def test_predict_empty_file(self):
        resp = self.client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(resp.status_code, 400)

    def test_predict_oversized_file(self):
        # Patch MAX_FILE_SIZE temporarily to avoid allocating 15MB in tests
        import main as svc
        original = svc.MAX_FILE_SIZE
        svc.MAX_FILE_SIZE = 50 * 1024  # 50 KB for test speed
        try:
            oversized_bytes = b"0" * (svc.MAX_FILE_SIZE + 1)
            resp = self.client.post(
                "/predict",
                files={"file": ("large.jpg", oversized_bytes, "image/jpeg")},
            )
            self.assertEqual(resp.status_code, 413)
        finally:
            svc.MAX_FILE_SIZE = original

if __name__ == "__main__":
    unittest.main()

