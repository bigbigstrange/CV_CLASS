"""Training hyperparameters for phase 1 and phase 2 (scheme A)."""

from dataclasses import dataclass, field
from pathlib import Path

from config.dataset_config import (
    KSDD2_MASK_DILATE,
    KSDD_INPUT_SIZE,
    KSDD_MASK_DILATE,
    KSDD_DEFAULT_FOLD,
    KSDD2_DEFAULT_WEAK_SPLIT,
    SSN_INPUT_SIZE,
    UNIFIED_INPUT_SIZE,
)
from config.paths import CODE_ROOT

OUTPUT_ROOT = CODE_ROOT / "outputs"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"


@dataclass
class TrainConfig:
    stage: str = "phase1"
    model_type: str = "segdec"
    input_size: tuple[int, int] = KSDD_INPUT_SIZE
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    ksdd_fold: int = KSDD_DEFAULT_FOLD
    ksdd2_weak_split: str = KSDD2_DEFAULT_WEAK_SPLIT
    ksdd_oversample: int = 6
    pos_weight: float = 3.0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    focal_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    seg_pos_weight: float = 3.0
    decision_pos_weight: float = 1.0
    decision_loss_weight: float = 1.0
    mask_dilate_ksdd: int = KSDD_MASK_DILATE
    mask_dilate_ksdd2: int = KSDD2_MASK_DILATE
    eval_threshold: float = 0.5
    min_defect_pixels: int = 8
    threshold_search: bool = True
    threshold_min: float = 0.3
    threshold_max: float = 0.7
    threshold_step: float = 0.05
    threshold_criterion: str = "f1"
    threshold_min_recall: float = 0.90
    image_score_mode: str = "decision"
    image_score_topk: int = 8
    target_fpr: float = 0.01
    pixel_ap_max_samples: int = 500_000
    ssn_config: dict = field(default_factory=dict)
    ssn_clip_grad: bool = True
    seed: int = 42
    device: str = "cuda"
    checkpoint_dir: Path = field(default_factory=lambda: CHECKPOINT_ROOT / "phase1")
    resume_from: Path | None = None
    auto_resume: bool = True
    save_every: int = 5
    best_metric: str = "f1"


STAGE_DEFAULTS: dict[str, dict] = {
    "phase1": {
        "epochs": 20,
        "learning_rate": 1e-4,
        "model_type": "segdec",
        "input_size": KSDD_INPUT_SIZE,
        "checkpoint_dir": CHECKPOINT_ROOT / "phase1",
    },
    "pretrain": {
        "epochs": 30,
        "learning_rate": 1e-4,
        "model_type": "ssn",
        "input_size": SSN_INPUT_SIZE,
        "batch_size": 4,
        "best_metric": "ap_det",
        "threshold_criterion": "f2",
        "image_score_mode": "hybrid_max",
        "threshold_min": 0.1,
        "threshold_max": 0.9,
        "threshold_step": 0.02,
        "checkpoint_dir": CHECKPOINT_ROOT / "pretrain",
    },
    "finetune": {
        "epochs": 15,
        "learning_rate": 5e-5,
        "model_type": "ssn",
        "input_size": SSN_INPUT_SIZE,
        "batch_size": 4,
        "best_metric": "ap_det",
        "threshold_criterion": "f2",
        "image_score_mode": "hybrid_max",
        "threshold_min": 0.1,
        "threshold_max": 0.9,
        "threshold_step": 0.02,
        "checkpoint_dir": CHECKPOINT_ROOT / "finetune",
    },
}


def resolve_resume_checkpoint(
    checkpoint_dir: Path,
    resume_from: Path | None,
    auto_resume: bool,
) -> Path | None:
    if resume_from is not None:
        return resume_from
    if not auto_resume:
        return None
    last_ckpt = checkpoint_dir / "last.pt"
    if last_ckpt.exists():
        return last_ckpt
    return None


def resolve_checkpoint_path(checkpoint: Path, fallbacks: list[Path] | None = None) -> Path:
    """Resolve checkpoint to an existing absolute path."""
    candidates = [
        checkpoint,
        Path.cwd() / checkpoint,
        CODE_ROOT / checkpoint,
    ]
    if checkpoint.is_absolute():
        candidates.insert(0, checkpoint)

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path.resolve()

    if fallbacks:
        for path in fallbacks:
            if path.exists():
                print(f"[checkpoint] fallback -> {path.resolve()}")
                return path.resolve()

    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}\nTried:\n  {tried}")


def build_train_config(stage: str, **overrides) -> TrainConfig:
    if stage not in STAGE_DEFAULTS:
        raise ValueError(f"Unknown stage: {stage}. Choose from {list(STAGE_DEFAULTS)}")
    config = TrainConfig(stage=stage, **STAGE_DEFAULTS[stage])
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config.resume_from = resolve_resume_checkpoint(
        config.checkpoint_dir,
        config.resume_from,
        config.auto_resume,
    )
    return config
