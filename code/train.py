"""Unified trainer for student models. Modes: baseline, distill."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import get_info, get_loader, is_multilabel
from .distill import HintonKDLoss
from .student import build_student
from .utils import auto_tune_batch_size, get_logger, per_subset_paths


class _DistillDataset(torch.utils.data.Dataset):
    """Wrap a base dataset so each item also carries its index-aligned teacher logit.

    Lets distillation use a normal multi-worker DataLoader (fast) instead of a
    single-threaded per-sample index pass (which was ~70 s/batch on the cluster).
    """
    def __init__(self, base, teacher_logits):
        self.base = base
        self.t = teacher_logits

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, torch.as_tensor(y), self.t[i]


def _sk_auc(y_true, y_score, multilabel: bool) -> float:
    """Use sklearn for a robust AUC that handles edge cases."""
    from sklearn.metrics import roc_auc_score
    try:
        if multilabel:
            return float(roc_auc_score(y_true, y_score, average="macro"))
        n = y_score.shape[1]
        if n == 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def _eval_loader(model, loader, device, multilabel: bool):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())
    logits = torch.cat(all_logits, 0)
    y = torch.cat(all_y, 0)
    if multilabel:
        scores = torch.sigmoid(logits).numpy()
        preds = (scores >= 0.5).astype(int)
        acc = float((preds == y.numpy().astype(int)).mean())
        auc = _sk_auc(y.numpy(), scores, multilabel=True)
    else:
        scores = F.softmax(logits, dim=-1).numpy()
        preds = scores.argmax(1)
        acc = float((preds == y.numpy()).mean())
        auc = _sk_auc(y.numpy(), scores, multilabel=False)
    return {"auc": auc, "acc": acc}


def train_student(mode: str, subset: str, cfg: dict) -> dict:
    """mode in {'baseline', 'distill'}. Returns dict of best-val metrics + history."""
    assert mode in ("baseline", "distill")
    log = get_logger(f"train.{mode}.{subset}",
                     log_file=per_subset_paths(cfg, subset)["log_dir"] / f"{mode}.log")
    info = get_info(subset)
    multilabel = is_multilabel(subset)
    paths = per_subset_paths(cfg, subset)
    ckpt_path = paths["checkpoint_dir"] / f"{mode}.pt"
    history_path = paths["log_dir"] / f"{mode}_history.json"

    if ckpt_path.exists() and history_path.exists():
        log.info("checkpoint + history already exist; skipping")
        return json.loads(history_path.read_text(encoding="utf-8"))["best"]

    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    # Auto-tune batch size against the student model.
    bs = auto_tune_batch_size(
        lambda: build_student(info["n_classes"]),
        sample_input_shape=(3, cfg["image_size"], cfg["image_size"]),
        candidate_sizes=tuple([cfg["batch_size"]] + [s for s in (32, 16, 8) if s < cfg["batch_size"]]),
        device=device,
    )
    log.info(f"auto-tuned batch size: {bs}")

    train_loader = get_loader(subset, "train", cfg, batch_size=bs, shuffle=True, drop_last=False)
    val_loader = get_loader(subset, "val", cfg, batch_size=bs, shuffle=False)

    model = build_student(info["n_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["lr_milestones"], gamma=cfg["lr_gamma"])

    if mode == "baseline":
        criterion = (nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss())
    else:
        criterion = HintonKDLoss(temperature=cfg["distill_temperature"],
                                  alpha=cfg["distill_alpha"], multilabel=multilabel)
        # Pre-load cached teacher logits in train order. MedMNIST samples are deterministic by index.
        teacher_logits_train = torch.load(paths["cache_dir"] / "logits_train.pt")

    use_amp = bool(cfg.get("mixed_precision", True)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history = {"train_loss": [], "val_auc": [], "val_acc": []}
    best = {"auc": -1.0, "acc": -1.0, "epoch": -1}

    # For distill mode we need per-sample teacher logits paired with each image.
    # Wrap the dataset so a normal multi-worker DataLoader yields (x, y, t_logits).
    if mode == "distill":
        distill_ds = _DistillDataset(train_loader.dataset, teacher_logits_train)
        distill_loader = DataLoader(
            distill_ds, batch_size=bs, shuffle=True,
            num_workers=cfg["num_workers"], pin_memory=cfg["pin_memory"],
            drop_last=False)

    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        n_seen = 0

        if mode == "baseline":
            iterator = tqdm(train_loader, desc=f"[{subset}|{mode}] epoch {epoch+1}/{cfg['epochs']}")
            for x, y in iterator:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits = model(x)
                        loss = criterion(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(x)
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()
                running += loss.item() * x.size(0)
                n_seen += x.size(0)
        else:
            iterator = tqdm(distill_loader,
                            desc=f"[{subset}|{mode}] epoch {epoch+1}/{cfg['epochs']}")
            for x, y, t_logits in iterator:
                x = x.to(device, non_blocking=True)
                if multilabel:
                    y = y.float().to(device)
                else:
                    y = y.squeeze(-1).long().to(device)
                t_logits = t_logits.to(device, non_blocking=True)

                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        s_logits = model(x)
                        loss = criterion(s_logits, t_logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    s_logits = model(x)
                    loss = criterion(s_logits, t_logits, y)
                    loss.backward()
                    optimizer.step()
                running += loss.item() * x.size(0)
                n_seen += x.size(0)

        scheduler.step()
        train_loss = running / max(n_seen, 1)
        val_metrics = _eval_loader(model, val_loader, device, multilabel)
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_metrics["auc"])
        history["val_acc"].append(val_metrics["acc"])
        log.info(f"epoch {epoch+1}: loss={train_loss:.4f} val_auc={val_metrics['auc']:.4f} "
                 f"val_acc={val_metrics['acc']:.4f}")

        target = val_metrics[cfg["val_metric"]]
        if target > best[cfg["val_metric"]]:
            best = {"auc": val_metrics["auc"], "acc": val_metrics["acc"], "epoch": epoch + 1}
            torch.save(model.state_dict(), ckpt_path)

    elapsed = (time.time() - t0) / 60.0
    best["train_time_min"] = round(elapsed, 2)
    payload = {"best": best, "history": history, "config": {
        "mode": mode, "subset": subset, "epochs": cfg["epochs"], "batch_size": bs,
        "lr": cfg["lr"], "T": cfg["distill_temperature"], "alpha": cfg["distill_alpha"]}}
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(f"done. best val_auc={best['auc']:.4f} acc={best['acc']:.4f} "
             f"epoch={best['epoch']} elapsed={elapsed:.1f} min")
    return best
