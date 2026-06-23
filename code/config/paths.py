"""Central dataset paths for the defect detection project."""

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = CODE_ROOT / "Datasets"

KSDD_ROOT = DATASETS_ROOT / "kolektor缺陷数据集"
KSDD_SPLITS_ROOT = DATASETS_ROOT / "KSDD-splits"
KSDD2_ROOT = DATASETS_ROOT / "KolektorSDD2"
