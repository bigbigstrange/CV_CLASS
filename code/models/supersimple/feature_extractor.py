"""Pretrained backbone feature extractor for SuperSimpleNet."""

from importlib import import_module

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str,
        layers: list[str],
        patch_size: int,
        image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.layers = layers

        try:
            models = import_module("torchvision.models")
            backbone_class = getattr(models, backbone)
            model = backbone_class(weights="IMAGENET1K_V1")
        except AttributeError as exc:
            raise AttributeError(f"Backbone {backbone} not found in torchvision.models.") from exc

        self.feature_extractor = create_feature_extractor(model, return_nodes=layers)
        self.pooler = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)
        self.feature_dim = self.get_feature_dim(image_size)

    def forward(self, images: Tensor) -> Tensor:
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(images)

        feature_layers = list(features.values())
        _, _, h, w = feature_layers[0].shape
        feature_map = []
        for layer in feature_layers:
            resized = F.interpolate(layer, size=(h * 2, w * 2), mode="bilinear", align_corners=True)
            feature_map.append(resized)
        feature_map = torch.cat(feature_map, dim=1)
        return self.pooler(feature_map)

    def get_feature_dim(self, image_shape: tuple[int, int]) -> tuple[int, int, int]:
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(torch.rand(1, 3, *image_shape))
        channels = sum(feature.shape[1] for feature in features.values())
        _, _, h, w = next(iter(features.values())).shape
        return channels, h * 2, w * 2
