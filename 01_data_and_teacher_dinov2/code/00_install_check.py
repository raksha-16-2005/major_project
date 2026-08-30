"""Verify CUDA, VRAM, dependencies and print run plan."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import device_info, load_config


def main():
    cfg = load_config()
    info = device_info()
    print("=" * 60)
    print("MedAlmighty Mini — install check")
    print("=" * 60)

    # Imports
    missing = []
    for mod in ("torch", "torchvision", "medmnist", "pytorch_grad_cam",
                "reportlab", "yaml", "tqdm", "sklearn", "matplotlib", "PIL"):
        try:
            __import__(mod if mod != "yaml" else "yaml")
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"[FAIL] missing modules: {missing}")
        print("       Run setup.bat first.")
        sys.exit(1)
    print("[OK] all required modules importable")

    # CUDA
    if not info["cuda_available"]:
        print("[FAIL] CUDA not available. GPU training will not work.")
        sys.exit(1)
    print(f"[OK] CUDA device: {info['device_name']}")
    print(f"     VRAM free: {info['vram_free_gb']} / {info['vram_total_gb']} GB")
    if info["vram_free_gb"] < 3.5:
        print("[WARN] less than ~4 GB VRAM free. Batch size will auto-fall to 16 or 8.")

    # Disk
    free = shutil.disk_usage(cfg["paths"]["cache"]).free / (1024**3)
    print(f"[OK] disk free at {cfg['paths']['cache']}: {free:.1f} GB")
    if free < 10:
        print("[WARN] less than 10 GB free. Caches and checkpoints may not fit.")

    # Plan summary
    print("-" * 60)
    print(f"Subsets to process ({len(cfg['subsets'])}): {', '.join(cfg['subsets'])}")
    print(f"Teacher: {cfg['teacher']}  Student: {cfg['student']}")
    print(f"Epochs: {cfg['epochs']}  batch_size: {cfg['batch_size']}  T={cfg['distill_temperature']}  alpha={cfg['distill_alpha']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
