"""No-op marker: teacher linear probe is trained inside scripts/01_cache_teacher.py.

This script exists so the pipeline numbering matches the README. Re-running it just
verifies that the teacher_probe checkpoint exists per subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, per_subset_paths


def main():
    cfg = load_config()
    missing = []
    for subset in cfg["subsets"]:
        paths = per_subset_paths(cfg, subset)
        ckpt = paths["checkpoint_dir"] / "teacher_probe.pt"
        if not ckpt.exists():
            missing.append(subset)
    if missing:
        print(f"[FAIL] teacher_probe checkpoints missing for: {missing}")
        print("       Run scripts/01_cache_teacher.py first.")
        sys.exit(1)
    print(f"[OK] teacher_probe checkpoints present for all {len(cfg['subsets'])} subsets.")


if __name__ == "__main__":
    main()
