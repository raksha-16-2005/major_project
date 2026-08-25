"""Train ResNet-50 from scratch (no distillation) on every subset."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train_student
from src.utils import load_config, seed_everything


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    for subset in cfg["subsets"]:
        print(f"\n=== baseline | {subset} ===")
        train_student("baseline", subset, cfg)


if __name__ == "__main__":
    main()
