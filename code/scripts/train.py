"""Unified training entry for phase 1 and phase 2 (scheme A)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config.train_config import build_train_config
from data.torch_dataset import (
    build_eval_loaders,
    build_finetune_loaders,
    build_phase1_loaders,
    build_pretrain_loaders,
)
from engine.trainer import SegmentationTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train defect segmentation model")
    parser.add_argument(
        "--stage",
        choices=["phase1", "pretrain", "finetune"],
        required=True,
        help="phase1=KSDD, pretrain=KSDD2, finetune=mixed KSDD2+KSDD",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Explicit checkpoint path. Default: auto-resume from last.pt if exists.",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable auto-resume from outputs/checkpoints/<stage>/last.pt",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Train from scratch (same as --no-auto-resume).",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auto_resume = not (args.no_auto_resume or args.fresh)
    overrides = {
        "ksdd_fold": args.fold,
        "device": args.device,
        "resume_from": args.resume_from,
        "auto_resume": auto_resume,
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["learning_rate"] = args.lr

    config = build_train_config(args.stage, **overrides)
    set_seed(config.seed)
    print(f"[{args.stage}] model={config.model_type} input={config.input_size}")

    if config.resume_from is not None:
        print(f"[{args.stage}] auto-resume from {config.resume_from}")
    else:
        print(f"[{args.stage}] training from scratch")

    trainer = SegmentationTrainer(config)

    try:
        if args.stage == "phase1":
            train_loader, val_loader = build_phase1_loaders(config)
            print(
                f"[phase1] KSDD fold {config.ksdd_fold}: "
                f"train={len(train_loader.dataset)}, test={len(val_loader.dataset)}"
            )
            trainer.fit(train_loader, val_loader)

        elif args.stage == "pretrain":
            train_loader, val_loader = build_pretrain_loaders(config)
            print(
                f"[pretrain] KSDD2: train={len(train_loader.dataset)}, "
                f"test={len(val_loader.dataset)}"
            )
            trainer.fit(train_loader, val_loader)

        elif args.stage == "finetune":
            train_loader, ksdd2_test_loader, ksdd_test_loader = build_finetune_loaders(config)
            if config.resume_from is None:
                pretrain_ckpt = config.checkpoint_dir.parent / "pretrain" / "best.pt"
                if pretrain_ckpt.exists():
                    trainer.load_checkpoint(
                        pretrain_ckpt,
                        resume_training=False,
                        reset_best_f1=True,
                    )
                    print(f"[finetune] loaded pretrain weights: {pretrain_ckpt}")
                else:
                    print("[finetune] warning: pretrain checkpoint not found, training from scratch")

            print(
                f"[finetune] mixed train={len(train_loader.dataset)}, "
                f"KSDD2 test={len(ksdd2_test_loader.dataset)}, "
                f"KSDD test={len(ksdd_test_loader.dataset)}"
            )
            trainer.fit(train_loader, ksdd2_test_loader)

            phase1_ckpt = config.checkpoint_dir.parent / "phase1" / "best.pt"
            if phase1_ckpt.exists():
                segdec_config = build_train_config(
                    "phase1",
                    ksdd_fold=config.ksdd_fold,
                    device=config.device,
                    auto_resume=False,
                )
                segdec_trainer = SegmentationTrainer(segdec_config)
                segdec_trainer.load_checkpoint(phase1_ckpt, resume_training=False)
                _, ksdd_test_loader_segdec = build_eval_loaders(segdec_config)
                ksdd_metrics = segdec_trainer.evaluate(ksdd_test_loader_segdec)
                print(f"[finetune] KSDD fold-test (SegDecNet): {ksdd_metrics}")
            else:
                ksdd_metrics = trainer.evaluate(ksdd_test_loader)
                print(f"[finetune] KSDD fold-test (SSN fallback): {ksdd_metrics}")

    except KeyboardInterrupt:
        print(f"[{args.stage}] interrupted. Re-run the same command to continue training.")


if __name__ == "__main__":
    main()
