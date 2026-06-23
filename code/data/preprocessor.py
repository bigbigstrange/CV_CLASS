"""Unified image/mask preprocessing for KSDD and KSDD2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from config.dataset_config import (
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    KSDD2_INPUT_SIZE,
    KSDD2_MASK_DILATE,
    KSDD_INPUT_SIZE,
    KSDD_MASK_DILATE,
    MASK_THRESHOLD,
    UNIFIED_INPUT_SIZE,
)
from data.schema import DefectRecord


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessor settings shared across datasets."""

    size: tuple[int, int]  # (width, height)
    mean: float = IMAGE_MEAN
    std: float = IMAGE_STD
    mask_threshold: int = MASK_THRESHOLD
    mask_dilate_ksdd: int = KSDD_MASK_DILATE
    mask_dilate_ksdd2: int = KSDD2_MASK_DILATE
    channels_first: bool = False
    rgb: bool = False
    imagenet_norm: bool = False


class Preprocessor:
    """Grayscale -> resize -> normalize pipeline for images and masks."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig(size=UNIFIED_INPUT_SIZE)

    @property
    def size(self) -> tuple[int, int]:
        return self.config.size

    @property
    def output_shape(self) -> tuple[int, int]:
        width, height = self.config.size
        return (height, width)

    def to_grayscale(self, image: Image.Image) -> Image.Image:
        if image.mode == "L":
            return image
        return image.convert("L")

    def resize_image(self, image: Image.Image) -> Image.Image:
        return image.resize(self.config.size, Image.BILINEAR)

    def resize_mask(self, mask: Image.Image) -> Image.Image:
        return mask.resize(self.config.size, Image.NEAREST)

    def dilate_mask(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0 or not mask.any():
            return mask
        kernel = radius * 2 + 1
        mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=kernel))
        return (np.array(mask_img) > 0).astype(np.float32)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image /= 255.0
        if self.config.imagenet_norm:
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            mean = np.array(IMAGENET_MEAN, dtype=np.float32)
            std = np.array(IMAGENET_STD, dtype=np.float32)
            image = (image - mean) / std
            if self.config.channels_first:
                return image.transpose(2, 0, 1)
            return image
        image = (image - self.config.mean) / self.config.std
        return image

    def binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        return (mask > self.config.mask_threshold).astype(np.float32)

    def _format_image(self, image: np.ndarray) -> np.ndarray:
        if self.config.imagenet_norm:
            return image
        if self.config.channels_first:
            return image[np.newaxis, ...]
        return image

    def process_image(self, source: Path | Image.Image) -> np.ndarray:
        image = source if isinstance(source, Image.Image) else Image.open(source)
        image = self.to_grayscale(image)
        if self.config.rgb:
            image = image.convert("RGB")
        image = self.resize_image(image)
        array = np.array(image)
        if self.config.imagenet_norm:
            return self.normalize(array)
        array = self.normalize(array)
        return self._format_image(array)

    def process_mask(
        self,
        source: Path | Image.Image | None,
        dilate: int = 0,
    ) -> np.ndarray:
        width, height = self.config.size
        if source is None:
            return np.zeros((height, width), dtype=np.float32)

        mask = source if isinstance(source, Image.Image) else Image.open(source)
        mask = self.to_grayscale(mask)
        mask = self.resize_mask(mask)
        array = self.binarize_mask(np.array(mask))
        return self.dilate_mask(array, dilate)

    def mask_dilate_for_record(self, record: DefectRecord) -> int:
        if record.dataset == "ksdd":
            return self.config.mask_dilate_ksdd
        return self.config.mask_dilate_ksdd2

    def __call__(self, record: DefectRecord) -> dict[str, object]:
        image = self.process_image(record.image_path)
        dilate = self.mask_dilate_for_record(record)
        if record.has_mask:
            mask = self.process_mask(record.mask_path, dilate=dilate)
        else:
            mask = self.process_mask(None)

        return {
            "image": image,
            "mask": mask,
            "label": float(record.is_defect),
            "has_mask": record.has_mask and record.is_segmented,
            "dataset": record.dataset,
            "split": record.split,
            "sample_id": record.sample_id,
            "group_id": record.group_id,
            "image_path": str(record.image_path),
        }


def create_unified_preprocessor(
    size: tuple[int, int] = UNIFIED_INPUT_SIZE,
    channels_first: bool = False,
    mask_dilate_ksdd: int = KSDD_MASK_DILATE,
    mask_dilate_ksdd2: int = KSDD2_MASK_DILATE,
) -> Preprocessor:
    """Cross-dataset preprocessor for scheme A (pretrain + finetune)."""
    return Preprocessor(
        PreprocessConfig(
            size=size,
            channels_first=channels_first,
            mask_dilate_ksdd=mask_dilate_ksdd,
            mask_dilate_ksdd2=mask_dilate_ksdd2,
        )
    )


def create_ksdd_preprocessor(channels_first: bool = False) -> Preprocessor:
    return Preprocessor(PreprocessConfig(size=KSDD_INPUT_SIZE, channels_first=channels_first))


def create_ksdd2_preprocessor(channels_first: bool = False) -> Preprocessor:
    return Preprocessor(PreprocessConfig(size=KSDD2_INPUT_SIZE, channels_first=channels_first))
