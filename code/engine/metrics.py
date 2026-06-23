"""Evaluation metrics for segmentation and image-level defect detection."""

from __future__ import annotations

import numpy as np
import torch

IMAGE_SCORE_MODES = {"decision", "seg_max", "seg_topk_mean", "hybrid_max"}
THRESHOLD_CRITERIA = {"f1", "f2", "recall_at_target"}


def f_beta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    beta_sq = beta * beta
    denom = beta_sq * precision + recall + 1e-8
    return (1.0 + beta_sq) * precision * recall / denom


def compute_image_det_scores(
    seg_logits: torch.Tensor,
    decision_logits: torch.Tensor | None,
    score_mode: str = "decision",
    topk: int = 8,
) -> torch.Tensor:
    """Aggregate image-level detection scores from seg map and/or decision head."""
    seg_probs = torch.sigmoid(seg_logits.detach().cpu())
    seg_flat = seg_probs.flatten(1)
    seg_max = seg_flat.amax(dim=1)

    if score_mode == "seg_max" or decision_logits is None:
        return seg_max

    decision_scores = torch.sigmoid(decision_logits.detach().cpu()).reshape(-1)
    if score_mode == "decision":
        return decision_scores
    if score_mode == "seg_topk_mean":
        k = min(topk, seg_flat.size(1))
        topk_vals, _ = seg_flat.topk(k, dim=1)
        return topk_vals.mean(dim=1)
    if score_mode == "hybrid_max":
        return torch.maximum(decision_scores, seg_max)

    raise ValueError(
        f"Unknown image_score_mode: {score_mode}. Choose from {sorted(IMAGE_SCORE_MODES)}"
    )


def _metrics_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    preds = (scores >= threshold).astype(np.float64)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f_beta_score(precision, recall, beta=2.0),
        "accuracy": accuracy,
    }


def _threshold_objective(
    metrics: dict[str, float],
    criterion: str,
    min_recall: float,
) -> float:
    if criterion == "f1":
        return metrics["f1"]
    if criterion == "f2":
        return metrics["f2"]
    if criterion == "recall_at_target":
        if metrics["recall"] >= min_recall:
            return metrics["precision"]
        return -1.0
    raise ValueError(
        f"Unknown threshold_criterion: {criterion}. Choose from {sorted(THRESHOLD_CRITERIA)}"
    )


def recall_at_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """Recall when false-positive rate on normal samples is capped at target_fpr."""
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    neg_scores = scores[labels <= 0.5]
    pos_scores = scores[labels > 0.5]
    if neg_scores.size == 0 or pos_scores.size == 0:
        return {
            "recall_at_fpr": 0.0,
            "recall_at_fpr_threshold": float("nan"),
            "recall_at_fpr_actual_fpr": float("nan"),
        }

    threshold = float(np.quantile(neg_scores, 1.0 - target_fpr))
    recall = float((pos_scores >= threshold).mean())
    actual_fpr = float((neg_scores >= threshold).mean())
    return {
        "recall_at_fpr": recall,
        "recall_at_fpr_threshold": threshold,
        "recall_at_fpr_actual_fpr": actual_fpr,
    }


def _image_predictions(
    probs: torch.Tensor,
    threshold: float,
    min_defect_pixels: int,
) -> torch.Tensor:
    preds = (probs >= threshold).float()
    pixel_counts = preds.view(preds.size(0), -1).sum(dim=1)
    prob_max = probs.view(probs.size(0), -1).amax(dim=1)
    return ((pixel_counts >= min_defect_pixels) | (prob_max >= threshold)).float()


@torch.no_grad()
def compute_batch_metrics(
    logits: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    min_defect_pixels: int = 8,
    decision_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) >= 1).float().sum(dim=(1, 2, 3))
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))

    if decision_logits is not None:
        image_preds = (torch.sigmoid(decision_logits) >= threshold).float()
    else:
        image_preds = _image_predictions(probs, threshold, min_defect_pixels)
    image_labels = labels.float()

    tp = ((image_preds == 1) & (image_labels == 1)).sum().float()
    fp = ((image_preds == 1) & (image_labels == 0)).sum().float()
    fn = ((image_preds == 0) & (image_labels == 1)).sum().float()
    tn = ((image_preds == 0) & (image_labels == 0)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        "iou": float(iou.mean().item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1": float(f1.item()),
        "accuracy": float(accuracy.item()),
        "threshold": float(threshold),
    }


def aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "threshold": 0.5,
        }
    keys = [key for key in metrics_list[0] if key != "threshold"]
    result = {key: sum(item[key] for item in metrics_list) / len(metrics_list) for key in keys}
    result["threshold"] = metrics_list[-1].get("threshold", 0.5)
    return result


@torch.no_grad()
def compute_metrics_from_tensors(
    logits: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    min_defect_pixels: int,
    decision_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    return compute_batch_metrics(
        logits, masks, labels, threshold, min_defect_pixels, decision_logits
    )


@torch.no_grad()
def search_best_threshold(
    logits: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    threshold_min: float = 0.3,
    threshold_max: float = 0.7,
    threshold_step: float = 0.05,
    min_defect_pixels: int = 8,
    decision_logits: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(threshold_min, threshold_max + 1e-6, threshold_step)
    best_threshold = 0.5
    best_metrics = compute_metrics_from_tensors(
        logits, masks, labels, 0.5, min_defect_pixels, decision_logits
    )
    best_f1 = best_metrics["f1"]

    for threshold in thresholds:
        metrics = compute_metrics_from_tensors(
            logits,
            masks,
            labels,
            float(threshold),
            min_defect_pixels,
            decision_logits,
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    best_metrics["threshold"] = best_threshold
    return best_threshold, best_metrics


def search_best_threshold_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold_min: float = 0.3,
    threshold_max: float = 0.7,
    threshold_step: float = 0.05,
    criterion: str = "f1",
    min_recall: float = 0.90,
) -> tuple[float, dict[str, float]]:
    """Image-level threshold search from precomputed scores (memory-safe)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    thresholds = np.arange(threshold_min, threshold_max + 1e-6, threshold_step)

    best_threshold = 0.5
    best_objective = -1.0
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0, "accuracy": 0.0}

    for threshold in thresholds:
        metrics = _metrics_at_threshold(scores, labels, float(threshold))
        objective = _threshold_objective(metrics, criterion, min_recall)
        if objective > best_objective:
            best_objective = objective
            best_threshold = float(threshold)
            best_metrics = metrics

    best_metrics["threshold"] = best_threshold
    best_metrics["threshold_criterion"] = criterion
    return best_threshold, best_metrics


def _to_numpy_1d(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)


def binary_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """AP aligned with sklearn average_precision_score."""
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.sum() <= 0:
        return 0.0

    order = np.argsort(scores)[::-1]
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / labels.sum()

    ap = 0.0
    prev_recall = 0.0
    for prec, rec in zip(precision, recall):
        ap += max(0.0, rec - prev_recall) * prec
        prev_recall = rec
    return float(ap)


def binary_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    max_points: int = 500_000,
    seed: int = 42,
) -> float:
    """Binary AUROC via pairwise comparison (stable for pixel-level)."""
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    pos_scores = scores[labels > 0.5]
    neg_scores = scores[labels <= 0.5]
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    total = pos_scores.size + neg_scores.size
    if total <= max_points:
        pos_pick = pos_scores
        neg_pick = neg_scores
    else:
        n_pos = min(pos_scores.size, max_points // 2)
        n_neg = min(neg_scores.size, max_points - n_pos)
        pos_pick = rng.choice(pos_scores, n_pos, replace=pos_scores.size < n_pos)
        neg_pick = rng.choice(neg_scores, n_neg, replace=neg_scores.size < n_neg)

    n_pairs = min(200_000, pos_pick.size * neg_pick.size)
    pos_sample = rng.choice(pos_pick, n_pairs, replace=True)
    neg_sample = rng.choice(neg_pick, n_pairs, replace=True)
    greater = np.mean(pos_sample > neg_sample)
    equal = np.mean(pos_sample == neg_sample)
    return float(greater + 0.5 * equal)


class ApMetricsAccumulator:
    """Memory-safe streaming accumulator for AP / AUROC metrics."""

    def __init__(
        self,
        max_pixel_samples: int = 500_000,
        seed: int = 42,
        score_mode: str = "decision",
        topk: int = 8,
        target_fpr: float = 0.01,
    ) -> None:
        self.max_pixel_samples = max_pixel_samples
        self.max_pos_pixels = max(1, max_pixel_samples // 2)
        self.max_neg_pixels = max(1, max_pixel_samples - self.max_pos_pixels)
        self.rng = np.random.default_rng(seed)
        self.score_mode = score_mode
        self.topk = topk
        self.target_fpr = target_fpr
        self.labels: list[np.ndarray] = []
        self.det_scores: list[np.ndarray] = []
        self.decision_scores: list[np.ndarray] = []
        self.seg_scores: list[np.ndarray] = []
        self.pos_pixel_scores: list[np.ndarray] = []
        self.neg_pixel_scores: list[np.ndarray] = []
        self._pos_count = 0
        self._neg_count = 0
        self.has_decision_head = False

    def add_batch(
        self,
        seg_logits: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
        decision_logits: torch.Tensor | None = None,
    ) -> None:
        seg_probs = torch.sigmoid(seg_logits.detach().cpu())
        seg_scores = seg_probs.flatten(1).amax(dim=1)
        self.seg_scores.append(_to_numpy_1d(seg_scores))
        self.labels.append(_to_numpy_1d(labels))

        if decision_logits is not None:
            self.has_decision_head = True
            self.decision_scores.append(
                _to_numpy_1d(torch.sigmoid(decision_logits.detach().cpu()))
            )
        det = compute_image_det_scores(
            seg_logits,
            decision_logits,
            score_mode=self.score_mode,
            topk=self.topk,
        )
        self.det_scores.append(_to_numpy_1d(det))

        self._add_pixel_samples(seg_probs, masks.detach().cpu())

    def _sample_indices(
        self,
        indices: np.ndarray,
        remaining: int,
    ) -> np.ndarray:
        count = min(indices.size, remaining)
        if count <= 0:
            return np.array([], dtype=np.int64)
        return self.rng.choice(indices, count, replace=indices.size < count)

    def _add_pixel_samples(self, seg_probs: torch.Tensor, masks: torch.Tensor) -> None:
        scores = seg_probs.reshape(-1).numpy().astype(np.float32)
        pixel_labels = (masks.reshape(-1) > 0.5).numpy().astype(np.float32)

        pos_idx = np.flatnonzero(pixel_labels > 0.5)
        neg_idx = np.flatnonzero(pixel_labels <= 0.5)

        if pos_idx.size and self._pos_count < self.max_pos_pixels:
            pick = self._sample_indices(pos_idx, self.max_pos_pixels - self._pos_count)
            if pick.size:
                self.pos_pixel_scores.append(scores[pick].astype(np.float64))
                self._pos_count += pick.size

        if neg_idx.size and self._neg_count < self.max_neg_pixels:
            pick = self._sample_indices(neg_idx, self.max_neg_pixels - self._neg_count)
            if pick.size:
                self.neg_pixel_scores.append(scores[pick].astype(np.float64))
                self._neg_count += pick.size

    def finalize(self) -> dict[str, float]:
        labels = np.concatenate(self.labels) if self.labels else np.array([])
        det_scores = np.concatenate(self.det_scores) if self.det_scores else np.array([])
        seg_scores = np.concatenate(self.seg_scores) if self.seg_scores else np.array([])

        if labels.size == 0:
            return {
                "ap": 0.0,
                "ap_det": 0.0,
                "seg_ap_det": 0.0,
                "i_auroc": 0.0,
                "ap_loc": 0.0,
                "p_auroc": 0.0,
                "recall_at_fpr": 0.0,
                "recall_at_fpr_threshold": float("nan"),
                "recall_at_fpr_actual_fpr": float("nan"),
            }

        if self.has_decision_head and self.decision_scores:
            ap_scores = np.concatenate(self.decision_scores)
        else:
            ap_scores = det_scores

        det_name = "ap_det" if self.has_decision_head else "ap"
        result = {
            det_name: binary_average_precision(labels, ap_scores),
            "seg_ap_det": binary_average_precision(labels, seg_scores),
            "i_auroc": binary_auroc(labels, ap_scores),
        }
        if self.has_decision_head:
            result["ap"] = result["ap_det"]
            if self.score_mode != "decision":
                result["score_ap_det"] = binary_average_precision(labels, det_scores)

        result.update(recall_at_fpr(labels, det_scores, target_fpr=self.target_fpr))

        pos_scores = (
            np.concatenate(self.pos_pixel_scores).astype(np.float64)
            if self.pos_pixel_scores
            else np.array([], dtype=np.float64)
        )
        neg_scores = (
            np.concatenate(self.neg_pixel_scores).astype(np.float64)
            if self.neg_pixel_scores
            else np.array([], dtype=np.float64)
        )
        if pos_scores.size or neg_scores.size:
            pixel_scores = np.concatenate(
                [pos_scores, neg_scores] if pos_scores.size and neg_scores.size
                else (pos_scores if pos_scores.size else neg_scores)
            )
            pixel_labels = np.concatenate(
                [
                    np.ones(pos_scores.size, dtype=np.float64),
                    np.zeros(neg_scores.size, dtype=np.float64),
                ]
                if pos_scores.size and neg_scores.size
                else (
                    np.ones(pos_scores.size, dtype=np.float64)
                    if pos_scores.size
                    else np.zeros(neg_scores.size, dtype=np.float64)
                )
            )
            result["ap_loc"] = binary_average_precision(pixel_labels, pixel_scores)
            result["p_auroc"] = binary_auroc(pixel_labels, pixel_scores)
        else:
            result["ap_loc"] = 0.0
            result["p_auroc"] = 0.0
        return result


@torch.no_grad()
def compute_batch_iou(seg_logits: torch.Tensor, masks: torch.Tensor, threshold: float) -> float:
    probs = torch.sigmoid(seg_logits)
    preds = (probs >= threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) >= 1).float().sum(dim=(1, 2, 3))
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return float(iou.mean().item())


@torch.no_grad()
def compute_ap_metrics(
    seg_logits: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    decision_logits: torch.Tensor | None = None,
    max_pixel_samples: int = 500_000,
) -> dict[str, float]:
    """Compute AP metrics with streaming / stratified pixel sampling."""
    acc = ApMetricsAccumulator(max_pixel_samples=max_pixel_samples)
    batch_size = 64
    total = seg_logits.size(0)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        acc.add_batch(
            seg_logits[start:end],
            masks[start:end],
            labels[start:end],
            decision_logits[start:end] if decision_logits is not None else None,
        )
    return acc.finalize()


@torch.no_grad()
def compute_full_metrics(
    seg_logits: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
    min_defect_pixels: int = 8,
    decision_logits: torch.Tensor | None = None,
    max_pixel_samples: int = 500_000,
) -> dict[str, float]:
    metrics = compute_metrics_from_tensors(
        seg_logits,
        masks,
        labels,
        threshold,
        min_defect_pixels,
        decision_logits,
    )
    metrics.update(
        compute_ap_metrics(
            seg_logits,
            masks,
            labels,
            decision_logits,
            max_pixel_samples=max_pixel_samples,
        )
    )
    return metrics


@torch.no_grad()
def collect_logits_labels(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        logits_list.append(logits.cpu())
        mask_list.append(masks.cpu())
        label_list.append(labels.cpu())

    return (
        torch.cat(logits_list, dim=0),
        torch.cat(mask_list, dim=0),
        torch.cat(label_list, dim=0),
    )
