"""Quick checks of the contract preprocessing (run: ml\.venv\Scripts\python.exe -m unittest discover -s ml\tests)."""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import crop_array, crop_box, load_and_crop, to_tensor  # noqa: E402


class TestPreprocess(unittest.TestCase):
    def test_black_border_cropped(self):
        img = np.zeros((100, 200, 3), np.uint8)
        img[10:90, 30:170] = 120
        self.assertEqual(crop_box(img), (30, 10, 169, 89))
        rgb, box = crop_array(img)
        self.assertEqual(rgb.shape, (80, 140, 3))

    def test_no_border(self):
        img = np.full((50, 60, 3), 200, np.uint8)
        self.assertEqual(crop_box(img), (0, 0, 59, 49))

    def test_all_black_not_cropped(self):
        self.assertEqual(crop_box(np.zeros((40, 40, 3), np.uint8)), (0, 0, 39, 39))

    def test_min_keep(self):
        img = np.zeros((100, 100, 3), np.uint8)
        img[40:60, 40:60] = 200          # 20% < 25% -> keep full image
        self.assertEqual(crop_box(img), (0, 0, 99, 99))
        img[38:63, 38:63] = 200          # 25 px = 25% -> crop
        self.assertEqual(crop_box(img), (38, 38, 62, 62))

    def test_min_fraction_strict(self):
        img = np.zeros((100, 100, 3), np.uint8)
        img[:, 10:90] = 100
        img[50, 0:10] = 255              # column 0..9: 1/100 = 0.01 <= 0.02 -> ignored
        img[0:2, 95] = 255               # col 95: 0.02 not > 0.02 -> ignored
        self.assertEqual(crop_box(img), (10, 0, 89, 99))

    def test_threshold_strict(self):
        img = np.full((30, 30, 3), 20, np.uint8)   # gray == 20 is NOT > 20
        img[5:25, 5:25] = 21
        self.assertEqual(crop_box(img), (5, 5, 24, 24))

    def test_tensor_and_file(self):
        img = np.zeros((60, 80, 3), np.uint8)
        img[5:55, 10:70] = (10, 100, 200)          # BGR
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            cv2.imwrite(str(p), img)
            rgb, box = load_and_crop(p)
        self.assertEqual(box, (10, 5, 69, 54))
        self.assertEqual(tuple(rgb[0, 0]), (200, 100, 10))
        t = to_tensor(rgb, 32)
        self.assertEqual(t.shape, (1, 3, 32, 32))
        self.assertEqual(t.dtype, np.float32)
        self.assertAlmostEqual(float(t[0, 0, 3, 3]), 200 / 255, places=6)


if __name__ == "__main__":
    unittest.main()
