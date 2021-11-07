import torch
from torch.utils.data import Dataset, DataLoader

# typing
from typing import Tuple, Dict
from omegaconf import OmegaConf


# datasets
from datasets import (
    # 2D Datasets
    RotatedMNIST,
    MNIST6_180,
    MNIST6_M,
    CIFAR10,
    CIFAR100,
    STL10,
    PCam,
)


def construct_dataset(
    cfg: OmegaConf,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create Dataset instances for each partition (train, val, test) of a given dataset.
    :return: Tuple (training_set, validation_set, test_set)
    """

    dataset = {
        "rotMNIST": RotatedMNIST,
        "MNIST6-180": MNIST6_180,
        "MNIST6-M": MNIST6_M,
        "CIFAR10": CIFAR10,
        "CIFAR100": CIFAR100,
        "STL10": STL10,
        "PCam": PCam,
    }[cfg.dataset]

    training_set = dataset(
        partition="train",
        augment=cfg.augment,
    )
    test_set = dataset(
        partition="test",
        augment="None",
    )
    if cfg.dataset in ["PCam"]:
        validation_set = dataset(
            partition="valid",
            augment="None",
        )
    else:
        validation_set = None
    return training_set, validation_set, test_set


def construct_dataloaders(
    cfg: OmegaConf,
) -> Dict[str, DataLoader]:
    """
    Construct DataLoaders for the selected dataset
    :return dict("train": train_loader, "validation": val_loader , "test": test_loader)
    """
    training_set, validation_set, test_set = construct_dataset(cfg)

    num_workers = cfg.no_workers
    num_workers = num_workers * torch.cuda.device_count()
    training_loader = DataLoader(
        training_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    if validation_set is not None:
        val_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    else:
        val_loader = test_loader

    dataloaders = {
        "train": training_loader,
        "validation": val_loader,
        "test": test_loader,
    }

    return dataloaders
