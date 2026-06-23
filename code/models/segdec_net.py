"""SegDecNet: segmentation + decision head (JIM 2019, PyTorch port)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_channels: int, out_channels: int, kernel: int, repeats: int = 1) -> nn.Sequential:
    layers: list[nn.Module] = []
    channels = in_channels
    for _ in range(repeats):
        layers.extend(
            [
                nn.Conv2d(channels, out_channels, kernel_size=kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        channels = out_channels
    return nn.Sequential(*layers)


class SegDecNet(nn.Module):
    """
    Two-stage defect detector for KSDD.

    Segmentation head localizes defects; decision head predicts image-level label.
    Architecture follows skokec/segdec-net-jim2019 (DECISION_NET_FULL).
    """

    model_type = "segdec"

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 2

        self.enc1 = _conv_block(in_channels, c1, 5, repeats=2)
        self.enc2 = _conv_block(c1, c2, 5, repeats=3)
        self.enc3 = _conv_block(c2, c3, 5, repeats=4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Conv2d(c3, 1024, kernel_size=15, padding=7, bias=False)
        self.bn_bottleneck = nn.BatchNorm2d(1024)
        self.seg_head = nn.Conv2d(1024, 1, kernel_size=1)

        self.dec_pool1 = nn.MaxPool2d(2)
        self.dec_conv1 = _conv_block(1025, 8, 5)
        self.dec_pool2 = nn.MaxPool2d(2)
        self.dec_conv2 = _conv_block(8, 16, 5)
        self.dec_pool3 = nn.MaxPool2d(2)
        self.dec_conv3 = _conv_block(16, 32, 5)
        self.dec_fc = nn.Linear(32 * 2 + 2, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.enc1(images)
        x = self.enc2(self.pool(x))
        x = self.enc3(self.pool(x))
        features = F.relu(self.bn_bottleneck(self.bottleneck(self.pool(x))))
        seg_logits = self.seg_head(features)

        seg_prob = torch.sigmoid(seg_logits)
        decision_input = torch.cat([features, seg_prob], dim=1)

        d = self.dec_conv1(self.dec_pool1(decision_input))
        d = self.dec_conv2(self.dec_pool2(d))
        d = self.dec_conv3(self.dec_pool3(d))

        d_avg = F.adaptive_avg_pool2d(d, 1).flatten(1)
        d_max = F.adaptive_max_pool2d(d, 1).flatten(1)
        s_avg = F.adaptive_avg_pool2d(seg_logits, 1).flatten(1)
        s_max = F.adaptive_max_pool2d(seg_logits, 1).flatten(1)

        decision_logits = self.dec_fc(torch.cat([d_avg, d_max, s_avg, s_max], dim=1)).squeeze(1)
        seg_logits_up = F.interpolate(seg_logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "seg_logits": seg_logits_up,
            "decision_logits": decision_logits,
        }
