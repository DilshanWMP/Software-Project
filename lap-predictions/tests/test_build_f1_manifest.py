import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.build_f1_manifest import build_rows, validate_rows, write_manifest


class BuildF1ManifestTests(unittest.TestCase):
    def test_builds_unlabeled_row_for_valid_image_mask_pair(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            image_path = raw_root / "m2caiseg/train/images/frame001.jpg"
            mask_path = raw_root / "m2caiseg/train/groundtruth/frame001_gt.png"
            image_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(image_path)
            Image.new("RGB", (8, 8)).save(mask_path)

            rows = build_rows(raw_root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["relative_path"], "m2caiseg/train/images/frame001.jpg")
            self.assertEqual(rows[0]["annotation_path"], "m2caiseg/train/groundtruth/frame001_gt.png")
            self.assertEqual(len(rows[0]["image_sha256"]), 64)
            self.assertEqual(rows[0]["label"], "")
            self.assertEqual(validate_rows(rows, raw_root), [])

    def test_reports_missing_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            image_path = raw_root / "m2caiseg/train/images/frame001.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(image_path)

            rows = build_rows(raw_root)
            errors = validate_rows(rows, raw_root)

            self.assertEqual(len(errors), 1)
            self.assertIn("missing annotation path", errors[0])

    def test_rejects_labels_outside_the_proposed_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            image_path = raw_root / "m2caiseg/train/images/frame001.jpg"
            mask_path = raw_root / "m2caiseg/train/groundtruth/frame001_gt.png"
            image_path.parent.mkdir(parents=True)
            mask_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(image_path)
            Image.new("RGB", (8, 8)).save(mask_path)

            row = build_rows(raw_root)[0]
            row["label"] = "normal"

            errors = validate_rows([row], raw_root)

            self.assertTrue(any("outside the proposed draft schema" in error for error in errors))

    def test_writes_expected_manifest_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested/manifest.csv"
            write_manifest([], output_path)

            self.assertTrue(output_path.is_file())
            self.assertEqual(
                output_path.read_text(encoding="utf-8").strip(),
                "dataset_id,relative_path,annotation_path,image_id,image_sha256,patient_or_video_id,"
                "source_split,label,label_source,annotation_type,license_or_access_note",
            )

    def test_assigns_labels_based_on_mask_colors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_root = Path(temp_dir)
            
            # Case 1: Mask with tool pixels (G=255, R=0, e.g. cyan [0, 255, 255])
            img1 = raw_root / "m2caiseg/train/images/frame001.jpg"
            mask1 = raw_root / "m2caiseg/train/groundtruth/frame001_gt.png"
            img1.parent.mkdir(parents=True)
            mask1.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(img1)
            tool_mask = Image.new("RGB", (8, 8), color=(0, 255, 255))
            tool_mask.save(mask1)

            # Case 2: Mask without tool pixels (e.g. background black [0, 0, 0])
            img2 = raw_root / "m2caiseg/train/images/frame002.jpg"
            mask2 = raw_root / "m2caiseg/train/groundtruth/frame002_gt.png"
            Image.new("RGB", (8, 8)).save(img2)
            anatomy_mask = Image.new("RGB", (8, 8), color=(85, 170, 0))
            anatomy_mask.save(mask2)

            rows = build_rows(raw_root, assign_labels=True)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["label"], "tool_present")
            self.assertEqual(rows[0]["label_source"], "m2caiseg_mask_tool_presence")
            self.assertEqual(rows[1]["label"], "anatomy_only")
            self.assertEqual(rows[1]["label_source"], "m2caiseg_mask_tool_presence")
            self.assertEqual(validate_rows(rows, raw_root), [])


if __name__ == "__main__":
    unittest.main()

