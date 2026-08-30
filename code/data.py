"""MedMNIST 2D loaders: 28x28 -> 224x224, grayscale -> 3-channel, ImageNet norm."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as _mp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Kubeflow notebook containers cap /dev/shm at ~64 MB. The default DataLoader
# sharing strategy ("file_descriptor") passes worker tensors through /dev/shm,
# which overflows mid-epoch and the kernel SIGKILLs a worker ("Killed"). Route
# shared tensors through disk-backed temp files instead so multi-worker loading
# works without the shared-memory cap.
try:
    _mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

import medmnist
from medmnist import INFO


def get_info(subset: str) -> dict:
    info = INFO[subset]
    return {
        "task":         info["task"],            # 'multi-class', 'binary-class', 'multi-label, binary-class', 'ordinal-regression'
        "n_channels":   info["n_channels"],
        "n_classes":    len(info["label"]),
        "label":        info["label"],
        "python_class": info["python_class"],
    }


def is_multilabel(subset: str) -> bool:
    return "multi-label" in INFO[subset]["task"]


def build_transform(image_size: int, mean: list[float], std: list[float], n_channels: int):
    ops = [
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
    ]
    if n_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transforms.Compose(ops)


def get_dataset(subset: str, split: str, cfg: dict, download: bool = True) -> Dataset:
    info = get_info(subset)
    DatasetClass = getattr(medmnist, info["python_class"])
    tfm = build_transform(cfg["image_size"], cfg["imagenet_mean"], cfg["imagenet_std"],
                           info["n_channels"])
    ds = DatasetClass(split=split, transform=tfm, download=download,
                       root=str(cfg["paths"]["data"]),
                       size=28)  # default 28x28 source; we upsample in transform
    return ds


def collate_targets(batch, multilabel: bool):
    """MedMNIST returns labels as int64 column vectors. Squeeze for CE; keep float for BCE."""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.stack([torch.as_tensor(l) for l in labels], dim=0)
    if multilabel:
        labels = labels.float()
    else:
        labels = labels.squeeze(-1).long()
    return imgs, labels


class _CollateFn:
    """Pickle-safe collate for Windows DataLoader workers."""
    def __init__(self, multilabel: bool):
        self.multilabel = multilabel
    def __call__(self, batch):
        return collate_targets(batch, self.multilabel)


def get_loader(subset: str, split: str, cfg: dict, batch_size: int | None = None,
               shuffle: bool | None = None, drop_last: bool = False) -> DataLoader:
    ds = get_dataset(subset, split, cfg)
    multilabel = is_multilabel(subset)
    bs = batch_size if batch_size is not None else cfg["batch_size"]
    sh = shuffle if shuffle is not None else (split == "train")
    return DataLoader(
        ds, batch_size=bs, shuffle=sh,
        num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
        drop_last=drop_last,
        collate_fn=_CollateFn(multilabel),
    )
