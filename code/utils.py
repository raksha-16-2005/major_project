"""Utilities: seeding, logging, paths, VRAM probe, batch-size auto-tuner."""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

DEMO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = None) -> dict:
    path = Path(path) if path else DEMO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = (DEMO_ROOT / val).resolve()
        cfg["paths"][key].mkdir(parents=True, exist_ok=True)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def device_info() -> dict:
    info = {"cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        info["vram_free_gb"] = round(free / (1024**3), 2)
        info["vram_total_gb"] = round(total / (1024**3), 2)
    return info


def auto_tune_batch_size(model_factory, sample_input_shape, candidate_sizes=(64, 32, 16, 8),
                          device: str = "cuda") -> int:
    """Try forward+backward with descending batch sizes; return the first that fits."""
    if device != "cuda" or not torch.cuda.is_available():
        return min(candidate_sizes)
    for bs in candidate_sizes:
        torch.cuda.empty_cache()
        try:
            model = model_factory().to(device)
            x = torch.randn(bs, *sample_input_shape, device=device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            del model, x, y, loss
            torch.cuda.empty_cache()
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    return candidate_sizes[-1]


def per_subset_paths(cfg: dict, subset: str) -> dict:
    p = cfg["paths"]
    out = {
        "cache_dir":       p["cache"] / subset,
        "checkpoint_dir":  p["checkpoints"] / subset,
        "log_dir":         p["logs"] / subset,
        "curve_dir":       p["results"] / "curves",
        "gradcam_dir":     p["results"] / "gradcam",
    }
    for v in out.values():
        v.mkdir(parents=True, exist_ok=True)
    return out
