"""PyTorch Dataset and DataLoader builders for all training stages."""

from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from config.train_config import TrainConfig
from data import ksdd, ksdd2
from data.preprocessor import PreprocessConfig, Preprocessor
from data.schema import DefectRecord


class DefectDataset(Dataset):
    def __init__(self, records: list[DefectRecord], preprocessor: Preprocessor) -> None:
        self.records = records
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sample = self.preprocessor(record)
        image = sample["image"]
        if image.ndim == 2:
            image = torch.from_numpy(image).unsqueeze(0).float()
        elif image.ndim == 3 and image.shape[0] not in (1, 3):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).float()

        mask = torch.from_numpy(sample["mask"]).unsqueeze(0).float()
        label = torch.tensor(sample["label"], dtype=torch.float32)
        is_segmented = torch.tensor(float(sample["has_mask"]), dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "is_segmented": is_segmented,
            "has_mask": bool(sample["has_mask"]),
            "dataset": sample["dataset"],
            "sample_id": sample["sample_id"],
        }


def _make_preprocessor(config: TrainConfig) -> Preprocessor:
    use_ssn = config.model_type == "ssn"
    return Preprocessor(
        PreprocessConfig(
            size=config.input_size,
            channels_first=not use_ssn,
            rgb=use_ssn,
            imagenet_norm=use_ssn,
            mask_dilate_ksdd=config.mask_dilate_ksdd,
            mask_dilate_ksdd2=config.mask_dilate_ksdd2,
        )
    )


def _build_loader(
    records: list[DefectRecord],
    preprocessor: Preprocessor,
    config: TrainConfig,
    shuffle: bool,
    sampler=None,
) -> DataLoader:
    dataset = DefectDataset(records, preprocessor)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )


def build_phase1_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_records, test_records = ksdd.load_fold(fold=config.ksdd_fold)
    preprocessor = _make_preprocessor(config)
    train_loader = _build_loader(train_records, preprocessor, config, shuffle=True)
    test_loader = _build_loader(test_records, preprocessor, config, shuffle=False)
    return train_loader, test_loader


def build_pretrain_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_records, test_records = ksdd2.load_train_test(weak_split=config.ksdd2_weak_split)
    preprocessor = _make_preprocessor(config)
    train_loader = _build_loader(train_records, preprocessor, config, shuffle=True)
    test_loader = _build_loader(test_records, preprocessor, config, shuffle=False)
    return train_loader, test_loader


def build_finetune_loaders(
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    ksdd2_train, ksdd2_test = ksdd2.load_train_test(weak_split=config.ksdd2_weak_split)
    ksdd_train, ksdd_test = ksdd.load_fold(fold=config.ksdd_fold)

    preprocessor = _make_preprocessor(config)
    ksdd2_dataset = DefectDataset(ksdd2_train, preprocessor)
    ksdd_dataset = DefectDataset(ksdd_train, preprocessor)
    mixed_dataset = ConcatDataset([ksdd2_dataset, ksdd_dataset])

    weights = [1.0] * len(ksdd2_train) + [float(config.ksdd_oversample)] * len(ksdd_train)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        mixed_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )
    ksdd2_test_loader = _build_loader(ksdd2_test, preprocessor, config, shuffle=False)
    ksdd_test_loader = _build_loader(ksdd_test, preprocessor, config, shuffle=False)
    return train_loader, ksdd2_test_loader, ksdd_test_loader


def build_eval_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    _, ksdd2_test = ksdd2.load_train_test(weak_split=config.ksdd2_weak_split)
    _, ksdd_test = ksdd.load_fold(fold=config.ksdd_fold)
    preprocessor = _make_preprocessor(config)
    return (
        _build_loader(ksdd2_test, preprocessor, config, shuffle=False),
        _build_loader(ksdd_test, preprocessor, config, shuffle=False),
    )
