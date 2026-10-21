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


if __name__ == "__main__":
    unittest.main()
