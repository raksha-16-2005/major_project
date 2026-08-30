"""Cache DINOv2 features + linear-probe teacher logits for every subset."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cache import cache_teacher_logits
from src.utils import load_config, seed_everything


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    for subset in cfg["subsets"]:
        print(f"\n=== {subset} ===")
        cache_teacher_logits(subset, cfg)


if __name__ == "__main__":
    main()
