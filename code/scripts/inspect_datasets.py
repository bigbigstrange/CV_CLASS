"""Verify local datasets and print split / format summary."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config.dataset_config import KSDD2_INPUT_SIZE, KSDD_DEFAULT_FOLD, KSDD_INPUT_SIZE
from config.paths import KSDD2_ROOT, KSDD_ROOT, KSDD_SPLITS_ROOT
from data import ksdd, ksdd2
from PIL import Image


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def inspect_ksdd() -> None:
    _print_header("KSDD")
    print(f"root: {KSDD_ROOT}")
    all_records = ksdd.load_all()
    summary = ksdd.summarize(all_records)
    sample = all_records[0]
    image = Image.open(sample.image_path)
    mask = Image.open(sample.mask_path)

    print("format:")
    print("  image: PartN.jpg, grayscale, ~500 x 1263")
    print("  label: PartN_label.bmp, pixel mask, same size as image")
    print("  grouping: kos01..kos50 (physical commutator), 8 views each")
    print("stats:", summary)
    print(f"sample: {sample.sample_id}, mode={image.mode}, size={image.size}")
    print(f"label : mode={mask.mode}, size={mask.size}")

    train, test = ksdd.load_fold(KSDD_DEFAULT_FOLD)
    print("\nofficial 3-fold split (JIM 2019, split by kos):")
    print(f"  fold {KSDD_DEFAULT_FOLD}: train={ksdd.summarize(train)}, test={ksdd.summarize(test)}")
    print(f"  split file: {KSDD_SPLITS_ROOT / 'split.pyb'}")
    print(f"recommended input size: {KSDD_INPUT_SIZE} (width x height)")


def inspect_ksdd2() -> None:
    _print_header("KSDD2")
    print(f"root: {KSDD2_ROOT}")
    train, test = ksdd2.load_train_test()
    sample = train[0]
    image = Image.open(sample.image_path)

    print("format:")
    print("  image: {id}.png, RGB, ~230 x 630")
    print("  label: {id}_GT.png, all-zero=OK, nonzero=defect")
    print("  split: fixed train/ test folders + split_weakly_*.pyb")
    print("stats:")
    print("  train:", ksdd2.summarize(train))
    print("  test :", ksdd2.summarize(test))
    print(f"sample: {sample.sample_id}, mode={image.mode}, size={image.size}")
    print(f"recommended input size: {KSDD2_INPUT_SIZE} (width x height)")


def inspect_project_plan() -> None:
    _print_header("Suggested two-phase workflow")
    print("Phase 1 - system build on KSDD:")
    print("  - use official 3-fold CV (by kos) for reproducible baseline")
    print("  - all 399 images have masks; good for end-to-end pipeline debugging")
    print("Phase 2 - model optimization on KSDD2:")
    print("  - pretrain / finetune on train split (2331 images, 246 defective)")
    print("  - evaluate on test split (1004 images, 110 defective)")
    print("  - use split_weakly_246 for full pixel supervision during finetune")
    print("Cross-dataset note:")
    print("  - unify to grayscale + resize before mixing pipelines")
    print("  - keep KSDD fold-test as final small-sample benchmark")


def main() -> None:
    inspect_ksdd()
    inspect_ksdd2()
    inspect_project_plan()


if __name__ == "__main__":
    main()
