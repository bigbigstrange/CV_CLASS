"""KSDD2 (Kolektor Surface-Defect Dataset 2) indexing and split helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from config.paths import KSDD2_ROOT
from data.schema import DefectRecord


def _mask_has_defect(mask_path: Path) -> bool:
    mask = np.array(Image.open(mask_path))
    return bool((mask > 0).any())


def _build_record(
    split: str,
    sample_id: str,
    root: Path,
    is_segmented: bool,
) -> DefectRecord | None:
    image_path = root / split / f"{sample_id}.png"
    if not image_path.exists():
        return None

    mask_path = root / split / f"{sample_id}_GT.png"
    has_mask = mask_path.exists()
    if has_mask:
        is_defect = _mask_has_defect(mask_path)
    else:
        mask_path = None
        is_defect = False

    return DefectRecord(
        dataset="ksdd2",
        split=split,
        sample_id=sample_id,
        image_path=image_path,
        mask_path=mask_path,
        has_mask=has_mask,
        is_defect=is_defect,
        is_segmented=is_segmented,
    )


def load_split(
    split: str,
    root: Path = KSDD2_ROOT,
    weak_split: str = "split_weakly_246",
) -> list[DefectRecord]:
    """
    Load KSDD2 train or test split.

    For train, `weak_split` controls how many samples expose pixel masks:
    - split_weakly_0   : 2085 segmented (all-negative masks + positives)
    - split_weakly_246 : all 2331 train samples segmented (full supervision)
    Test split always uses full masks.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"KSDD2 split directory not found: {split_dir}")

    if split == "test":
        sample_ids = sorted(
            p.stem for p in split_dir.glob("*.png") if not p.name.endswith("_GT.png")
        )
        segmented = {sample_id: True for sample_id in sample_ids}
    else:
        pyb_path = root / f"{weak_split}.pyb"
        if not pyb_path.exists():
            raise FileNotFoundError(f"KSDD2 weak split file not found: {pyb_path}")
        with pyb_path.open("rb") as handle:
            train_samples, _ = pickle.load(handle)
        segmented = {str(sample_id): bool(has_mask) for sample_id, has_mask in train_samples}
        sample_ids = sorted(segmented.keys(), key=int)

    records: list[DefectRecord] = []
    for sample_id in sample_ids:
        record = _build_record(split, sample_id, root, segmented.get(sample_id, split == "test"))
        if record is not None:
            records.append(record)
    return records


def load_train_test(
    root: Path = KSDD2_ROOT,
    weak_split: str = "split_weakly_246",
) -> tuple[list[DefectRecord], list[DefectRecord]]:
    return load_split("train", root, weak_split), load_split("test", root, weak_split)


def summarize(records: list[DefectRecord]) -> dict[str, int | float]:
    defect_images = sum(1 for record in records if record.is_defect)
    segmented = sum(1 for record in records if record.is_segmented)
    return {
        "images": len(records),
        "segmented_images": segmented,
        "defect_images": defect_images,
        "ok_images": len(records) - defect_images,
        "defect_ratio": round(defect_images / len(records), 4) if records else 0.0,
    }
