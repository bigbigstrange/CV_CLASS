"""Evaluate Tier 1 checkpoints: SSN on KSDD2, SegDecNet on KSDD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from config.train_config import CHECKPOINT_ROOT, build_train_config, resolve_checkpoint_path
from data.torch_dataset import build_eval_loaders
from engine.metrics import IMAGE_SCORE_MODES, THRESHOLD_CRITERIA
from engine.trainer import SegmentationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tier 1 defect detection models")
    parser.add_argument(
        "--ssn-checkpoint",
        type=Path,
        default=None,
        help="SuperSimpleNet checkpoint for KSDD2. Default: finetune/best.pt with fallbacks.",
    )
    parser.add_argument(
        "--segdec-checkpoint",
        type=Path,
        default=None,
        help="SegDecNet checkpoint for KSDD. Default: phase1/best.pt.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--threshold-criterion",
        choices=sorted(THRESHOLD_CRITERIA),
        default=None,
        help="Threshold search objective. Default: stage config (f2 for SSN).",
    )
    parser.add_argument(
        "--image-score-mode",
        choices=sorted(IMAGE_SCORE_MODES),
        default=None,
        help="Image-level score source. Default: stage config (hybrid_max for SSN).",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=None,
        help="Target FPR for Recall@FPR metric (default: 0.01).",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run legacy decision+f1 baseline for SSN and print delta.",
    )
    return parser.parse_args()


def ssn_fallbacks() -> list[Path]:
    finetune_dir = CHECKPOINT_ROOT / "finetune"
    pretrain_dir = CHECKPOINT_ROOT / "pretrain"
    return [
        finetune_dir / "best.pt",
        finetune_dir / "last.pt",
        pretrain_dir / "best.pt",
        pretrain_dir / "last.pt",
    ]


def segdec_fallbacks() -> list[Path]:
    phase1_dir = CHECKPOINT_ROOT / "phase1"
    return [
        phase1_dir / "best.pt",
        phase1_dir / "last.pt",
    ]


def _apply_eval_overrides(config, args: argparse.Namespace) -> None:
    if args.threshold_criterion is not None:
        config.threshold_criterion = args.threshold_criterion
    if args.image_score_mode is not None:
        config.image_score_mode = args.image_score_mode
    if args.target_fpr is not None:
        config.target_fpr = args.target_fpr


def _print_detection_summary(name: str, metrics: dict) -> None:
    ap = metrics.get("ap_det", metrics.get("ap", 0.0))
    print(
        f"{name}  AP-det={ap:.4f}  Recall={metrics.get('recall', 0):.4f}  "
        f"Precision={metrics.get('precision', 0):.4f}  F1={metrics.get('f1', 0):.4f}  "
        f"F2={metrics.get('f2', 0):.4f}  "
        f"Recall@FPR={metrics.get('recall_at_fpr', 0):.4f}  "
        f"thr={metrics.get('threshold', 0):.2f}  "
        f"score={metrics.get('image_score_mode', 'decision')}  "
        f"crit={metrics.get('threshold_criterion', 'f1')}"
    )


def main() -> None:
    args = parse_args()

    ssn_config = build_train_config(
        "finetune",
        ksdd_fold=args.fold,
        device=args.device,
        auto_resume=False,
        resume_from=None,
    )
    segdec_config = build_train_config(
        "phase1",
        ksdd_fold=args.fold,
        device=args.device,
        auto_resume=False,
        resume_from=None,
    )
    _apply_eval_overrides(ssn_config, args)

    ssn_ckpt = resolve_checkpoint_path(
        args.ssn_checkpoint or CHECKPOINT_ROOT / "finetune" / "best.pt",
        fallbacks=ssn_fallbacks(),
    )
    segdec_ckpt = resolve_checkpoint_path(
        args.segdec_checkpoint or CHECKPOINT_ROOT / "phase1" / "best.pt",
        fallbacks=segdec_fallbacks(),
    )

    ksdd2_loader, ksdd_loader = build_eval_loaders(ssn_config)
    _, ksdd_loader_segdec = build_eval_loaders(segdec_config)

    ssn_trainer = SegmentationTrainer(ssn_config)
    ssn_trainer.load_checkpoint(ssn_ckpt, resume_training=False)
    ksdd2_metrics = ssn_trainer.evaluate(ksdd2_loader, search_threshold=True)

    ksdd2_baseline_metrics = None
    if args.compare_baseline:
        baseline_config = build_train_config(
            "finetune",
            ksdd_fold=args.fold,
            device=args.device,
            auto_resume=False,
            resume_from=None,
            threshold_criterion="f1",
            image_score_mode="decision",
            threshold_min=0.3,
            threshold_max=0.7,
            threshold_step=0.05,
        )
        baseline_trainer = SegmentationTrainer(baseline_config)
        baseline_trainer.load_checkpoint(ssn_ckpt, resume_training=False)
        ksdd2_baseline_metrics = baseline_trainer.evaluate(ksdd2_loader, search_threshold=True)

    segdec_trainer = SegmentationTrainer(segdec_config)
    segdec_trainer.load_checkpoint(segdec_ckpt, resume_training=False)
    ksdd_metrics = segdec_trainer.evaluate(ksdd_loader_segdec, search_threshold=True)

    results = {
        "tier1": True,
        "ksdd2_model": "SuperSimpleNet",
        "ksdd_model": "SegDecNet",
        "ssn_checkpoint": str(ssn_ckpt),
        "segdec_checkpoint": str(segdec_ckpt),
        "ssn_eval_threshold": ssn_trainer.eval_threshold,
        "segdec_eval_threshold": segdec_trainer.eval_threshold,
        "ssn_eval_config": {
            "threshold_criterion": ssn_config.threshold_criterion,
            "image_score_mode": ssn_config.image_score_mode,
            "target_fpr": ssn_config.target_fpr,
        },
        "primary_metrics": {
            "ksdd2": "ap_det (image-level Average Precision, KSDD2 benchmark)",
            "ksdd": "ap (image-level Average Precision, SegDecNet benchmark)",
            "recall_at_fpr": f"Recall when FPR <= {ssn_config.target_fpr:.2%} on normal samples",
        },
        "ksdd2_test": ksdd2_metrics,
        "ksdd_fold_test": ksdd_metrics,
    }
    if ksdd2_baseline_metrics is not None:
        results["ksdd2_test_baseline"] = ksdd2_baseline_metrics

    print("\n=== Primary benchmark metrics ===")
    ksdd2_ap = ksdd2_metrics.get("ap_det", ksdd2_metrics.get("ap", 0.0))
    ksdd_ap = ksdd_metrics.get("ap", ksdd_metrics.get("ap_det", 0.0))
    print(f"KSDD2  AP-det = {ksdd2_ap:.4f}  |  I-AUROC = {ksdd2_metrics.get('i_auroc', 0):.4f}")
    print(f"KSDD   AP     = {ksdd_ap:.4f}  |  I-AUROC = {ksdd_metrics.get('i_auroc', 0):.4f}")
    print(f"KSDD2  AP-loc = {ksdd2_metrics.get('ap_loc', 0):.4f}  |  P-AUROC = {ksdd2_metrics.get('p_auroc', 0):.4f}")
    print(f"KSDD   AP-loc = {ksdd_metrics.get('ap_loc', 0):.4f}  |  P-AUROC = {ksdd_metrics.get('p_auroc', 0):.4f}")

    print("\n=== Recall-oriented detection (SSN) ===")
    _print_detection_summary("KSDD2 ", ksdd2_metrics)
    if ksdd2_baseline_metrics is not None:
        _print_detection_summary("Base  ", ksdd2_baseline_metrics)
        delta_recall = ksdd2_metrics["recall"] - ksdd2_baseline_metrics["recall"]
        delta_fpr_recall = (
            ksdd2_metrics.get("recall_at_fpr", 0.0)
            - ksdd2_baseline_metrics.get("recall_at_fpr", 0.0)
        )
        print(
            f"Delta  Recall={delta_recall:+.4f}  "
            f"Recall@FPR={delta_fpr_recall:+.4f}  "
            f"(new vs decision+f1 baseline)"
        )
    print()

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
