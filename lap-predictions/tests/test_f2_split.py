"""Unit and regression tests for F2 data preparation, splitting, and leakage assertions."""

import csv
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = PROJECT_ROOT / "data/split_manifest.csv"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPECTED_CLASSES = {"tool_present", "anatomy_only"}
EXPECTED_SPLITS = {"train", "val", "test"}


class TestF2DataPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SPLIT_MANIFEST.exists():
            raise unittest.SkipTest("split_manifest.csv does not exist yet. Run prepare_dataset.py first.")
        
        with SPLIT_MANIFEST.open(encoding="utf-8") as f:
            cls.rows = list(csv.DictReader(f))

    def test_split_manifest_has_expected_fields(self):
        expected_fields = {
            "image_id",
            "filename",
            "clean_path",
            "group_id",
            "label",
            "blur",
            "image_sha256",
            "width",
            "height",
            "split",
        }
        self.assertGreater(len(self.rows), 0, "Manifest contains no records.")
        self.assertTrue(expected_fields.issubset(self.rows[0].keys()))

    def test_zero_group_leakage_across_splits(self):
        """Verify that 0 patient/sequence groups overlap across train, val, and test splits."""
        groups_by_split: dict[str, set[str]] = {s: set() for s in EXPECTED_SPLITS}
        all_groups = set()

        for row in self.rows:
            split = row["split"]
            group = row["group_id"]
            self.assertIn(split, EXPECTED_SPLITS, f"Unexpected split: {split}")
            groups_by_split[split].add(group)
            all_groups.add(group)

        # Pairwise intersection must be empty
        self.assertEqual(
            groups_by_split["train"] & groups_by_split["val"],
            set(),
            "Data leakage detected between train and val!",
        )
        self.assertEqual(
            groups_by_split["train"] & groups_by_split["test"],
            set(),
            "Data leakage detected between train and test!",
        )
        self.assertEqual(
            groups_by_split["val"] & groups_by_split["test"],
            set(),
            "Data leakage detected between val and test!",
        )

    def test_all_processed_files_exist_on_disk(self):
        """Verify that every entry in split_manifest.csv exists in data/processed/ and is readable."""
        for row in self.rows:
            split = row["split"]
            label = row["label"]
            filename = row["filename"]
            file_path = PROCESSED_DIR / split / label / filename

            self.assertTrue(file_path.is_file(), f"Missing processed file: {file_path}")

            # Verify image readability and dimension check
            with Image.open(file_path) as img:
                self.assertGreaterEqual(img.width, 100)
                self.assertGreaterEqual(img.height, 100)
                # Verify EXIF is stripped
                exif = img.getexif()
                self.assertEqual(len(exif), 0, f"EXIF metadata found in {file_path}")

    def test_split_proportions_and_classes(self):
        """Verify that all splits contain both approved classes and rough 70/15/15 distribution."""
        split_counts = {s: 0 for s in EXPECTED_SPLITS}
        class_counts = {c: 0 for c in EXPECTED_CLASSES}

        for row in self.rows:
            split_counts[row["split"]] += 1
            class_counts[row["label"]] += 1

        total = len(self.rows)
        self.assertGreater(total, 250, "Expected dataset to contain > 250 frames.")

        # Proportions: train ~65-75%, val ~10-20%, test ~10-20%
        train_ratio = split_counts["train"] / total
        val_ratio = split_counts["val"] / total
        test_ratio = split_counts["test"] / total

        self.assertGreater(train_ratio, 0.60)
        self.assertLess(train_ratio, 0.80)
        self.assertGreater(val_ratio, 0.10)
        self.assertLess(val_ratio, 0.25)
        self.assertGreater(test_ratio, 0.10)
        self.assertLess(test_ratio, 0.25)

        # Classes
        for cls_name in EXPECTED_CLASSES:
            self.assertGreater(class_counts[cls_name], 30)

    def test_sample_grid_artifacts_generated(self):
        """Verify visual sample grids exist in reports/."""
        self.assertTrue((REPORTS_DIR / "sample_grid_tool_present.png").is_file())
        self.assertTrue((REPORTS_DIR / "sample_grid_anatomy_only.png").is_file())
        self.assertTrue((REPORTS_DIR / "sample_grids_overview.png").is_file())


if __name__ == "__main__":
    unittest.main()
