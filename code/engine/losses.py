"""Combined segmentation losses for imbalanced defect detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce
        return loss.mean()


class CombinedSegLoss(nn.Module):
    """BCE + Dice + Focal, suited for small sparse defects."""

    def __init__(
        self,
        pos_weight: float = 3.0,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        self.dice = DiceLoss()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self._pos_weight = pos_weight

    def to(self, device: torch.device):
        super().to(device)
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self._pos_weight], device=device)
        )
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        return loss


def focal_loss_bce(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 4.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss used by SuperSimpleNet (DestSeg-style)."""
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class SegDecLoss(nn.Module):
    """Joint segmentation + decision loss for SegDecNet."""

    def __init__(
        self,
        seg_pos_weight: float = 3.0,
        decision_pos_weight: float = 1.0,
        decision_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.decision_weight = decision_weight
        self.seg_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([seg_pos_weight])
        )
        self.decision_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([decision_pos_weight])
        )
        self._seg_pos_weight = seg_pos_weight
        self._decision_pos_weight = decision_pos_weight

    def to(self, device: torch.device):
        super().to(device)
        self.seg_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self._seg_pos_weight], device=device)
        )
        self.decision_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self._decision_pos_weight], device=device)
        )
        return self

    def forward(
        self,
        seg_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        seg_loss = self.seg_loss(seg_logits, masks)
        decision_loss = self.decision_loss(decision_logits, labels)
        return seg_loss + self.decision_weight * decision_loss


class SSNLoss(nn.Module):
    """SuperSimpleNet segmentation + classification loss."""

    def __init__(self, seg_loss_th: float = 0.5) -> None:
        super().__init__()
        self.seg_loss_th = seg_loss_th

    def forward(
        self,
        anomaly_map: torch.Tensor,
        anomaly_score: torch.Tensor,
        mask: torch.Tensor,
        label: torch.Tensor,
        is_segmented: torch.Tensor,
    ) -> torch.Tensor:
        seg_probs = torch.sigmoid(anomaly_map)
        seg_focal = focal_loss_bce(seg_probs, mask, reduction="none")
        seg_l1 = torch.zeros_like(anomaly_map)
        seg_l1[mask == 0] = torch.clamp(anomaly_map[mask == 0] + self.seg_loss_th, min=0)
        seg_l1[mask > 0] = torch.clamp(-anomaly_map[mask > 0] + self.seg_loss_th, min=0)

        is_segmented = torch.cat((is_segmented, is_segmented)).bool()
        mask = mask
        bad_loss = seg_l1[is_segmented][mask[is_segmented] > 0]
        good_loss = seg_l1[is_segmented][mask[is_segmented] == 0]
        focal_val = seg_focal[is_segmented]

        good_mean = good_loss.mean() if len(good_loss) else torch.tensor(0.0, device=anomaly_map.device)
        bad_mean = bad_loss.mean() if len(bad_loss) else torch.tensor(0.0, device=anomaly_map.device)
        focal_mean = focal_val.mean() if len(focal_val) else torch.tensor(0.0, device=anomaly_map.device)
        seg_loss = good_mean + bad_mean + focal_mean
        cls_loss = focal_loss_bce(torch.sigmoid(anomaly_score), label)
        return seg_loss + cls_loss

