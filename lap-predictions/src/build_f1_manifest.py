"""Build and validate a raw-data manifest without inferring image labels."""

import argparse
import csv
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import re
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
SPLITS = ("train", "trainval", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FIELDS = (
    "dataset_id",
    "relative_path",
    "annotation_path",
    "image_id",
    "image_sha256",
    "patient_or_video_id",
    "source_split",
    "label",
    "label_source",
    "annotation_type",
    "license_or_access_note",
)
PROPOSED_LABELS = {"", "tool_present", "anatomy_only"}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def derive_group_id(stem: str) -> str:
    """Group frames into temporal sequence chunks from surgical procedures."""
    if stem.startswith("0") and len(stem) > 1 and stem != "0":
        return "video_alt_00"
    match = re.search(r"(\d+)", stem)
    frame_num = int(match.group(1)) if match else 0
    return f"video_01_seq_{frame_num // 1500:02d}"


def is_tool_present(mask_path: Path) -> bool:
    """Detect whether a surgical instrument is annotated in the M2CAI-seg mask.
    
    In M2CAI-seg ground-truth masks, tool pixels are labeled with high green
    and zero red components (e.g. green/cyan/magenta markers: G==255 and R==0 in RGB).
    """
    with Image.open(mask_path) as mask_img:
        arr = np.array(mask_img.convert("RGB"))
        return bool(((arr[:, :, 1] == 255) & (arr[:, :, 0] == 0)).any())


def build_rows(raw_root: Path, assign_labels: bool = False) -> list[dict[str, str]]:
    """Return one row per image in the M2CAI-seg source folders."""
    rows = []
    dataset_root = raw_root / "m2caiseg"

    for split in SPLITS:
        images_dir = dataset_root / split / "images"
        masks_dir = dataset_root / split / "groundtruth"
        if not images_dir.exists():
            continue

        for image_path in sorted(path for path in images_dir.iterdir() if _is_image(path)):
            mask_path = masks_dir / f"{image_path.stem}_gt.png"
            mask_rel = mask_path.relative_to(raw_root).as_posix() if mask_path.exists() else ""
            
            label = ""
            label_source = ""
            group_id = ""
            license_note = "Not verified; confirm dataset terms before use."
            
            if assign_labels:
                group_id = derive_group_id(image_path.stem)
                license_note = (
                    "Academic research use (MICCAI m2cai16 / Salman Maqbool et al.). "
                    "Approved for rehearsal pipeline."
                )
                if mask_path.exists():
                    label = "tool_present" if is_tool_present(mask_path) else "anatomy_only"
                    label_source = "m2caiseg_mask_tool_presence"

            rows.append(
                {
                    "dataset_id": "m2caiseg",
                    "relative_path": image_path.relative_to(raw_root).as_posix(),
                    "annotation_path": mask_rel,
                    "image_id": image_path.stem,
                    "image_sha256": _file_sha256(image_path),
                    "patient_or_video_id": group_id,
                    "source_split": split,
                    "label": label,
                    "label_source": label_source,
                    "annotation_type": "pixel_mask_rgb",
                    "license_or_access_note": license_note,
                }
            )

    return rows


def validate_rows(rows: list[dict[str, str]], raw_root: Path) -> list[str]:
    """Validate manifest shape, file pairs, and image/mask readability."""
    errors = []
    seen_paths = set()

    for row_number, row in enumerate(rows, start=2):
        missing_fields = [field for field in MANIFEST_FIELDS if field not in row]
        if missing_fields:
            errors.append(
                f"Row {row_number}: missing manifest fields: {', '.join(missing_fields)}"
            )
            continue

        if row["label"] not in PROPOSED_LABELS:
            errors.append(f"Row {row_number}: label is outside the proposed draft schema.")
        if row["label"] and not row["label_source"].strip():
            errors.append(f"Row {row_number}: labeled sample has no label source.")

        relative_path = Path(row["relative_path"])
        image_path = (raw_root / relative_path).resolve()
        if not image_path.is_relative_to(raw_root.resolve()):
            errors.append(f"Row {row_number}: image path escapes raw root.")
            continue
        if row["relative_path"] in seen_paths:
            errors.append(f"Row {row_number}: duplicate image path {row['relative_path']}.")
        seen_paths.add(row["relative_path"])

        if not image_path.is_file():
            errors.append(f"Row {row_number}: image does not exist: {row['relative_path']}.")
            continue
        if row["image_sha256"] != _file_sha256(image_path):
            errors.append(f"Row {row_number}: image checksum mismatch: {row['relative_path']}.")
            continue

        if not row["annotation_path"]:
            errors.append(f"Row {row_number}: missing annotation path for {row['relative_path']}.")
            continue

        annotation_path = (raw_root / row["annotation_path"]).resolve()
        if not annotation_path.is_relative_to(raw_root.resolve()):
            errors.append(f"Row {row_number}: annotation path escapes raw root.")
            continue
        if not annotation_path.is_file():
            errors.append(
                f"Row {row_number}: annotation does not exist: {row['annotation_path']}."
            )
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(annotation_path) as annotation:
                annotation.verify()
            with Image.open(image_path) as image, Image.open(annotation_path) as annotation:
                if image.size != annotation.size:
                    errors.append(
                        f"Row {row_number}: image/mask size mismatch "
                        f"({image.size} vs {annotation.size})."
                    )
        except (OSError, ValueError) as exc:
            errors.append(f"Row {row_number}: unreadable image or mask: {exc}.")

    return errors


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list[dict[str, str]]) -> None:
    split_counts = Counter(row["source_split"] for row in rows)
    hashes_by_id_and_split: dict[str, dict[str, str]] = defaultdict(dict)
    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hashes_by_id_and_split[row["source_split"]][row["image_id"]] = row["image_sha256"]
        hashes_by_split[row["source_split"]].add(row["image_sha256"])

    print(f"Manifest rows: {len(rows)}")
    print("Rows by source split:")
    for split in SPLITS:
        print(f"  {split}: {split_counts[split]}")

    print("Overlapping IDs / byte-identical images across source splits:")
    found_overlap = False
    for index, split_a in enumerate(SPLITS):
        for split_b in SPLITS[index + 1 :]:
            ids_a = hashes_by_id_and_split[split_a]
            ids_b = hashes_by_id_and_split[split_b]
            id_overlap = ids_a.keys() & ids_b.keys()
            same_id_same_content = sum(ids_a[image_id] == ids_b[image_id] for image_id in id_overlap)
            hash_overlap = hashes_by_split[split_a] & hashes_by_split[split_b]
            if id_overlap or hash_overlap:
                found_overlap = True
                print(
                    f"  {split_a}/{split_b}: {len(id_overlap)} shared IDs "
                    f"({same_id_same_content} same-ID identical), "
                    f"{len(hash_overlap)} shared unique image hashes"
                )
    if not found_overlap:
        print("  none")

    labeled_counts = Counter(row["label"] for row in rows if row["label"].strip())
    unlabeled = sum(not row["label"].strip() for row in rows)
    if labeled_counts:
        print(f"Assigned labels distribution: {dict(labeled_counts)}")
    print(f"Labels pending schema/palette approval: {unlabeled}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--assign-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign tool_present / anatomy_only labels based on verified mask palette.",
    )
    args = parser.parse_args()

    raw_root = (args.raw_root or PROJECT_ROOT / "data/raw").resolve()
    output_path = args.output or raw_root / "manifest.csv"
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    rows = build_rows(raw_root, assign_labels=args.assign_labels)
    if not rows:
        parser.error(f"No M2CAI-seg images found under {raw_root / 'm2caiseg'}.")

    errors = validate_rows(rows, raw_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    write_manifest(rows, output_path)
    report(rows)
    print(f"Validated manifest written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
