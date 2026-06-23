"""SuperSimpleNet (ICPR 2024) — PyTorch port for KSDD2."""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR
from torchvision.transforms import GaussianBlur

from models.supersimple.feature_extractor import FeatureExtractor
from models.supersimple.perlin_noise import rand_perlin_2d


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(module.weight, 1)


class FeatureAdaptor(nn.Module):
    def __init__(self, projection_dim: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(projection_dim, projection_dim, kernel_size=1, stride=1)
        self.apply(init_weights)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)


def _conv_block(in_channels: int, out_channels: int, kernel_size: int, padding: str = "same") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(
        self,
        projection_dim: int,
        hidden_dim: int,
        feature_w: int,
        feature_h: int,
        config: dict,
    ) -> None:
        super().__init__()
        self.fw = feature_w
        self.fh = feature_h
        self.stop_grad = config.get("stop_grad", False)

        self.seg = nn.Sequential(
            nn.Conv2d(projection_dim, hidden_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, bias=False),
        )
        self.dec_head = _conv_block(projection_dim + 1, 128, kernel_size=5)
        self.map_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.dec_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.fc_score = nn.Linear(128 * 2 + 2, 1)
        self.apply(init_weights)

    def get_params(self) -> tuple:
        seg_params = self.seg.parameters()
        dec_params = list(self.dec_head.parameters()) + list(self.fc_score.parameters())
        return seg_params, dec_params

    def forward(self, seg_features: Tensor, cls_features: Tensor) -> tuple[Tensor, Tensor]:
        anomaly_map = self.seg(seg_features)
        map_dec_copy = anomaly_map.detach() if self.stop_grad else anomaly_map
        dec_out = self.dec_head(torch.cat((cls_features, map_dec_copy), dim=1))

        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)
        map_max = self.map_max_pool(anomaly_map)
        map_avg = self.map_avg_pool(anomaly_map)
        if self.stop_grad:
            map_max = map_max.detach()
            map_avg = map_avg.detach()

        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze(dim=(2, 3))
        score = self.fc_score(dec_cat).squeeze(dim=1)
        return anomaly_map, score


class AnomalyGenerator(nn.Module):
    def __init__(
        self,
        noise_mean: float,
        noise_std: float,
        feature_h: int,
        feature_w: int,
        config: dict,
        perlin_range: tuple[int, int] = (0, 6),
    ) -> None:
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.min_perlin_scale = perlin_range[0]
        self.max_perlin_scale = perlin_range[1]
        self.height = feature_h
        self.width = feature_w
        self.config = config
        self.perlin_height = self.next_power_2(self.height)
        self.perlin_width = self.next_power_2(self.width)

    @staticmethod
    def next_power_2(num: int) -> int:
        return 1 << (num - 1).bit_length()

    def generate_perlin(self, batches: int) -> Tensor:
        perlin = []
        for _ in range(batches):
            perlin_scalex = 2 ** int(torch.randint(self.min_perlin_scale, self.max_perlin_scale, (1,)).item())
            perlin_scaley = 2 ** int(torch.randint(self.min_perlin_scale, self.max_perlin_scale, (1,)).item())
            perlin_noise = rand_perlin_2d(
                (self.perlin_height, self.perlin_width),
                (perlin_scalex, perlin_scaley),
            )
            perlin_noise = F.interpolate(
                perlin_noise.reshape(1, 1, self.perlin_height, self.perlin_width),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            threshold = self.config["perlin_thr"]
            perlin_thr = torch.where(perlin_noise > threshold, 1, 0)

            chance_anomaly = float(torch.rand(1).item())
            if chance_anomaly > 0.5:
                if self.config["no_anomaly"] == "full":
                    perlin_thr = torch.ones_like(perlin_thr)
                elif self.config["no_anomaly"] == "empty":
                    perlin_thr = torch.zeros_like(perlin_thr)
            perlin.append(perlin_thr)
        return torch.cat(perlin)

    def forward(
        self,
        features: Tensor | None,
        adapted: Tensor,
        mask: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
        batch_size = mask.shape[0]
        adapted = torch.cat((adapted, adapted))
        mask = torch.cat((mask, mask))
        labels = torch.cat((labels, labels))
        if features is not None:
            features = torch.cat((features, features))

        noise = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=adapted.shape,
            device=adapted.device,
            requires_grad=False,
        )
        noise_mask = torch.ones(batch_size * 2, 1, self.height, self.width, device=adapted.device)

        if not self.config["bad"]:
            noise_mask = noise_mask * (1 - labels.reshape(batch_size * 2, 1, 1, 1))
        if not self.config["overlap"]:
            noise_mask = noise_mask * (1 - mask)

        if self.config["perlin"]:
            perlin_mask = self.generate_perlin(batch_size * 2).to(adapted.device)
            noise_mask = noise_mask * perlin_mask
        else:
            noise_mask[:batch_size, ...] = 0

        mask = torch.where(mask + noise_mask > 0, 1, 0).float()
        new_anomalous = mask.reshape(batch_size * 2, -1).any(dim=1).float()
        labels = torch.where(labels + new_anomalous > 0, 1, 0).float()

        perturbed_adapt = adapted + noise * noise_mask
        perturbed_feat = features + noise * noise_mask if features is not None else None
        return perturbed_feat, perturbed_adapt, mask, labels


class AnomalyMapGenerator(nn.Module):
    def __init__(self, output_size: tuple[int, int], sigma: float = 4) -> None:
        super().__init__()
        self.size = output_size
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, input_tensor: Tensor) -> Tensor:
        anomaly_map = F.interpolate(input_tensor, size=self.size, mode="bilinear", align_corners=False)
        return self.blur(anomaly_map)


class SuperSimpleNet(nn.Module):
    """SuperSimpleNet for KSDD2 supervised / mixed training."""

    model_type = "ssn"

    def __init__(self, image_size: tuple[int, int], config: dict | None = None) -> None:
        super().__init__()
        self.image_size = image_size
        self.config = config or default_ssn_config(image_size)
        self.feature_extractor = FeatureExtractor(
            backbone=self.config.get("backbone", "wide_resnet50_2"),
            layers=self.config.get("layers", ["layer2", "layer3"]),
            patch_size=self.config.get("patch_size", 3),
            image_size=image_size,
        )
        fc, fh, fw = self.feature_extractor.feature_dim
        self.fh = fh
        self.fw = fw
        self.feature_adaptor = FeatureAdaptor(projection_dim=fc)
        self.adapt_cls_feat = self.config.get("adapt_cls_feat", False)
        self.discriminator = Discriminator(
            projection_dim=fc,
            hidden_dim=1024,
            feature_w=fw,
            feature_h=fh,
            config=self.config,
        )
        self.anomaly_generator = AnomalyGenerator(
            noise_mean=0,
            noise_std=self.config.get("noise_std", 0.015),
            feature_w=fw,
            feature_h=fh,
            config=self.config,
        )
        self.anomaly_map_generator = AnomalyMapGenerator(output_size=image_size, sigma=4)

    def forward(
        self,
        images: Tensor,
        mask: Tensor | None = None,
        label: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        features = self.feature_extractor(images)
        adapted = self.feature_adaptor(features)
        seg_feats = adapted
        cls_feats = adapted if self.adapt_cls_feat else features

        if self.training and mask is not None and label is not None:
            if self.adapt_cls_feat:
                _, noised_adapt, mask, label = self.anomaly_generator(
                    features=None, adapted=adapted, mask=mask, labels=label
                )
                seg_feats = noised_adapt
                cls_feats = noised_adapt
            else:
                noised_feat, noised_adapt, mask, label = self.anomaly_generator(
                    features=features, adapted=adapted, mask=mask, labels=label
                )
                seg_feats = noised_adapt
                cls_feats = noised_feat

            anomaly_map, anomaly_score = self.discriminator(seg_features=seg_feats, cls_features=cls_feats)
            return anomaly_map, anomaly_score, mask, label

        anomaly_map, anomaly_score = self.discriminator(seg_features=seg_feats, cls_features=cls_feats)
        anomaly_map = self.anomaly_map_generator(anomaly_map)
        return anomaly_map, anomaly_score

    def get_optimizers(self, epochs: int | None = None) -> tuple[Optimizer, LRScheduler]:
        total_epochs = epochs or self.config["epochs"]
        seg_params, dec_params = self.discriminator.get_params()
        optim = AdamW(
            [
                {"params": self.feature_adaptor.parameters(), "lr": self.config["adapt_lr"]},
                {"params": seg_params, "lr": self.config["seg_lr"], "weight_decay": 0.00001},
                {"params": dec_params, "lr": self.config["dec_lr"], "weight_decay": 0.00001},
            ]
        )
        sched = MultiStepLR(
            optim,
            milestones=[int(total_epochs * 0.8), int(total_epochs * 0.9)],
            gamma=self.config["gamma"],
        )
        return optim, sched


def default_ssn_config(image_size: tuple[int, int] | None = None) -> dict:
    return {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "patch_size": 3,
        "noise": True,
        "perlin": True,
        "no_anomaly": "empty",
        "bad": True,
        "overlap": False,
        "adapt_cls_feat": False,
        "noise_std": 0.015,
        "perlin_thr": 0.6,
        "seg_lr": 2e-4,
        "dec_lr": 2e-4,
        "adapt_lr": 1e-4,
        "gamma": 0.4,
        "stop_grad": False,
        "clip_grad": True,
        "seg_loss_th": 0.5,
        "epochs": 30,
        "image_size": image_size,
    }
