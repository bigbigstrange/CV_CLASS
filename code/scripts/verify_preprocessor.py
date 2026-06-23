"""Quick check that unified preprocessing works on KSDD and KSDD2."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config.dataset_config import UNIFIED_INPUT_SIZE
from data import ksdd, ksdd2
from data.preprocessor import create_unified_preprocessor


def _check_sample(name: str, record, preprocessor) -> None:
    sample = preprocessor(record)
    image = sample["image"]
    mask = sample["mask"]
    height, width = preprocessor.output_shape

    print(f"\n{name}")
    print(f"  id      : {sample['sample_id']}")
    print(f"  label   : {sample['label']}")
    print(f"  image   : shape={image.shape}, dtype={image.dtype}, "
          f"min={image.min():.3f}, max={image.max():.3f}")
    print(f"  mask    : shape={mask.shape}, unique={sorted(set(mask.flatten().tolist()))}")
    assert image.shape == (height, width)
    assert mask.shape == (height, width)
    assert image.dtype.name == "float32"
    assert mask.dtype.name == "float32"


def main() -> None:
    preprocessor = create_unified_preprocessor()
    print(f"unified size (w x h): {UNIFIED_INPUT_SIZE}")
    print(f"output shape (h x w): {preprocessor.output_shape}")

    ksdd_train, _ = ksdd.load_fold(fold=0)
    ksdd2_train, _ = ksdd2.load_train_test()

    _check_sample("KSDD", ksdd_train[0], preprocessor)
    _check_sample("KSDD2", ksdd2_train[0], preprocessor)

    ksdd_defect = next(record for record in ksdd_train if record.is_defect)
    ksdd2_defect = next(record for record in ksdd2_train if record.is_defect)
    _check_sample("KSDD defect", ksdd_defect, preprocessor)
    _check_sample("KSDD2 defect", ksdd2_defect, preprocessor)

    print("\nPreprocessor check passed.")


if __name__ == "__main__":
    main()
