#!/usr/bin/env python3
"""Train and evaluate one (condition, held-out center, seed) on a frozen tile set.

    python scripts/train.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --condition E1 --holdout 0 --seed 42 \\
        --runs /media/abhi/camelyon17/derived/runs_v1

The run directory is <runs>/<condition>/fold<holdout>/seed<seed>/; a finished run is never
overwritten. Run from the repository root (the freeze record's config path is relative to it).
"""

import argparse
import json
from pathlib import Path

from stain_benchmark.training.run import CONDITIONS, train_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True, help="frozen tile set directory (holds freeze.json)")
    ap.add_argument("--runs", required=True, help="parent directory for run outputs")
    ap.add_argument("--condition", choices=CONDITIONS, default="E1")
    ap.add_argument("--holdout", required=True, help="held-out center")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-only", action="store_true",
                    help="score training tiles only (allowed before the analysis plan is committed)")
    args = ap.parse_args()

    run_dir = Path(args.runs) / args.condition / f"fold{args.holdout}" / f"seed{args.seed}"
    roles = ("train",) if args.train_only else ("train", "test_in", "test_cross")
    s = train_run(args.tiles, args.holdout, args.seed, run_dir, condition=args.condition,
                  num_workers=args.workers, roles=roles)
    print(json.dumps({"run_dir": str(run_dir), "roles_scored": s["roles_scored"],
                      "metrics": s["metrics"], "delta_g_auroc": s["delta_g_auroc"],
                      "minutes": round(s["train_seconds"] / 60, 1)}, indent=2))


if __name__ == "__main__":
    main()
