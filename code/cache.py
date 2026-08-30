"""Pre-compute and persist DINOv2 teacher logits + features per (subset, split)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import get_dataset, get_loader, get_info, is_multilabel
from .teacher import LinearHead, load_dinov2
from .utils import get_logger, per_subset_paths


def cache_paths(cache_dir: Path, split: str) -> tuple[Path, Path, Path]:
    return (cache_dir / f"features_{split}.pt",
            cache_dir / f"labels_{split}.pt",
            cache_dir / f"logits_{split}.pt")


def has_features(cache_dir: Path, split: str) -> bool:
    feat_p, lab_p, _ = cache_paths(cache_dir, split)
    return feat_p.exists() and lab_p.exists()


def has_logits(cache_dir: Path, split: str) -> bool:
    _, _, log_p = cache_paths(cache_dir, split)
    return log_p.exists()


@torch.no_grad()
def cache_features(subset: str, cfg: dict, splits=("train", "val", "test")) -> None:
    """Cache CLS features + raw labels for every split (used by linear probe AND distillation)."""
    log = get_logger(f"cache.{subset}")
    paths = per_subset_paths(cfg, subset)
    cache_dir = paths["cache_dir"]

    if all(has_features(cache_dir, s) for s in splits):
        log.info("features cache already complete; skipping")
        return

    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    teacher = load_dinov2(cfg["teacher"]).to(device)

    for split in splits:
        feat_p, lab_p, _ = cache_paths(cache_dir, split)
        if feat_p.exists() and lab_p.exists():
            log.info(f"{split}: cached, skip")
            continue

        loader = get_loader(subset, split, cfg, batch_size=64, shuffle=False, drop_last=False)
        feats, labs = [], []
        for x, y in tqdm(loader, desc=f"[{subset}] features {split}"):
            x = x.to(device, non_blocking=True)
            f = teacher(x).cpu()
            feats.append(f)
            labs.append(y.cpu())
        feats = torch.cat(feats, 0)
        labs = torch.cat(labs, 0)
        torch.save(feats, feat_p)
        torch.save(labs, lab_p)
        log.info(f"{split}: saved features {tuple(feats.shape)} labels {tuple(labs.shape)}")


@torch.no_grad()
def cache_teacher_logits(subset: str, cfg: dict) -> None:
    """Train a linear probe on cached train features, then dump teacher logits for ALL splits.

    Logits feed the Hinton KD loss as soft targets. Storing logits (not soft probs) lets
    the loss apply temperature at training time.
    """
    log = get_logger(f"teacher_logits.{subset}")
    info = get_info(subset)
    paths = per_subset_paths(cfg, subset)
    cache_dir = paths["cache_dir"]

    cache_features(subset, cfg)

    if all(has_logits(cache_dir, s) for s in ("train", "val", "test")):
        log.info("teacher logits cache already complete; skipping")
        return

    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    head = LinearHead(cfg["teacher_feature_dim"], info["n_classes"]).to(device)
    multilabel = is_multilabel(subset)
    crit = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"])

    train_feats = torch.load(cache_dir / "features_train.pt").to(device)
    train_labs = torch.load(cache_dir / "labels_train.pt")
    if multilabel:
        train_labs = train_labs.float().to(device)
    else:
        train_labs = train_labs.squeeze(-1).long().to(device)

    n = train_feats.size(0)
    bs = 256
    head.train()
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(n, device=device)
        total = 0.0
        with torch.enable_grad():
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                logits = head(train_feats[idx])
                loss = crit(logits, train_labs[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * idx.numel()
        log.info(f"linear-probe epoch {epoch+1}/{cfg['epochs']} loss={total / n:.4f}")

    head.eval()
    for split in ("train", "val", "test"):
        feats = torch.load(cache_dir / f"features_{split}.pt").to(device)
        with torch.no_grad():
            logits = head(feats).cpu()
        torch.save(logits, cache_dir / f"logits_{split}.pt")
        log.info(f"{split}: saved logits {tuple(logits.shape)}")

    torch.save(head.state_dict(), paths["checkpoint_dir"] / "teacher_probe.pt")
    log.info("saved teacher_probe checkpoint")
