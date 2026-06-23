"""Model factory for Tier 1 architectures."""

from __future__ import annotations

from config.train_config import TrainConfig
from models.segdec_net import SegDecNet
from models.supersimple.supersimple_net import SuperSimpleNet, default_ssn_config
from models.unet import UNet


def create_model(config: TrainConfig):
    if config.model_type == "segdec":
        return SegDecNet(in_channels=1)
    if config.model_type == "ssn":
        height, width = config.input_size[1], config.input_size[0]
        ssn_config = default_ssn_config(image_size=(height, width))
        ssn_config.update(config.ssn_config)
        ssn_config["epochs"] = config.epochs
        return SuperSimpleNet(image_size=(height, width), config=ssn_config)
    if config.model_type == "unet":
        return UNet(in_channels=1, out_channels=1)
    raise ValueError(f"Unknown model_type: {config.model_type}")
