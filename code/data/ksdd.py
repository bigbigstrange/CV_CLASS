"""KSDD (Kolektor Surface-Defect Dataset) indexing and split helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from config.paths import KSDD_ROOT, KSDD_SPLITS_ROOT
from data.schema import DefectRecord

SPLIT_FILE = KSDD_SPLITS_ROOT / "split.pyb"


def _mask_has_defect(mask_path: Path) -> bool:
    mask = np.array(Image.open(mask_path))
    return bool((mask > 0).any())


def list_kos_ids(root: Path = KSDD_ROOT) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("kos"))


def iter_kos_images(kos_id: str, root: Path = KSDD_ROOT) -> list[DefectRecord]:
    kos_dir = root / kos_id
    records: list[DefectRecord] = []
    for image_path in sorted(kos_dir.glob("Part*.jpg")):
        mask_path = kos_dir / f"{image_path.stem}_label.bmp"
        if not mask_path.exists():
            continue
        records.append(
            DefectRecord(
                dataset="ksdd",
                split="all",
                sample_id=f"{kos_id}/{image_path.stem}",
                image_path=image_path,
                mask_path=mask_path,
                has_mask=True,
                is_defect=_mask_has_defect(mask_path),
                is_segmented=True,
            )
        )
    return records


def load_all(root: Path = KSDD_ROOT) -> list[DefectRecord]:
    records: list[DefectRecord] = []
    for kos_id in list_kos_ids(root):
        records.extend(iter_kos_images(kos_id, root))
    return records


def load_official_splits(split_file: Path = SPLIT_FILE) -> tuple[list[list[str]], list[list[str]], list[str]]:
    """Return (train_folds, test_folds, all_kos_ids) from the official JIM 2019 split file."""
    if not split_file.exists():
        raise FileNotFoundError(
            f"Official KSDD split file not found: {split_file}. "
            "Download from https://data.vicos.si/datasets/KSDD/KolektorSDD-training-splits.zip"
        )
    with split_file.open("rb") as handle:
        train_folds, test_folds, all_items = pickle.load(handle)
    return train_folds, test_folds, all_items


def load_fold(
    fold: int = 0,
    root: Path = KSDD_ROOT,
    split_file: Path = SPLIT_FILE,
) -> tuple[list[DefectRecord], list[DefectRecord]]:
    """Load one official 3-fold split. Images from the same kos never cross train/test."""
    train_folds, test_folds, _ = load_official_splits(split_file)
    if fold not in (0, 1, 2):
        raise ValueError("KSDD official split only defines fold 0, 1, or 2")

    train: list[DefectRecord] = []
    test: list[DefectRecord] = []
    train_kos = set(train_folds[fold])
    test_kos = set(test_folds[fold])

    for kos_id in train_kos:
        for record in iter_kos_images(kos_id, root):
            train.append(
                DefectRecord(
                    dataset=record.dataset,
                    split="train",
                    sample_id=record.sample_id,
                    image_path=record.image_path,
                    mask_path=record.mask_path,
                    has_mask=record.has_mask,
                    is_defect=record.is_defect,
                    is_segmented=record.is_segmented,
                )
            )

    for kos_id in test_kos:
        for record in iter_kos_images(kos_id, root):
            test.append(
                DefectRecord(
                    dataset=record.dataset,
                    split="test",
                    sample_id=record.sample_id,
                    image_path=record.image_path,
                    mask_path=record.mask_path,
                    has_mask=record.has_mask,
                    is_defect=record.is_defect,
                    is_segmented=record.is_segmented,
                )
            )

    return train, test


def summarize(records: list[DefectRecord]) -> dict[str, int | float]:
    kos_ids = {record.group_id for record in records}
    defect_images = sum(1 for record in records if record.is_defect)
    return {
        "images": len(records),
        "kos_samples": len(kos_ids),
        "defect_images": defect_images,
        "ok_images": len(records) - defect_images,
        "defect_ratio": round(defect_images / len(records), 4) if records else 0.0,
    }
