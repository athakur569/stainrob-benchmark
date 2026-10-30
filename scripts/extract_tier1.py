#!/usr/bin/env python3
"""Extract Tier 1a/1b monitor features for a subset of frozen tiles, optionally under a shift.

    python scripts/extract_tier1.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --out /media/abhi/camelyon17/derived/tier1_v1/train_split/unshifted --subset train-split
    python scripts/extract_tier1.py ... --shift blur --dose 3      # dose index into the dose table

Subsets: ``train-split`` = tiles of patients in the training split (allowed now); ``all`` = every
QC-passing tile, which needs the committed analysis plan (DR-012), like scoring test sets.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from stain_benchmark.monitor.extract import extract_tier1
from stain_benchmark.training.run import require_committed_plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", choices=["train-split", "all"], required=True)
    ap.add_argument("--shift", default=None, help="shift family from the dose table")
    ap.add_argument("--dose", type=int, default=None, help="dose index (0-4) in the dose table")
    ap.add_argument("--doses", default="configs/shift_doses_camelyon17.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tiles_dir = Path(args.tiles)
    man = pd.read_csv(tiles_dir / "manifest.csv")
    man = man[man["qc_pass"]]
    if args.subset == "all":
        require_committed_plan()
        table, rule = man, "all QC-passing tiles"
    else:
        ps = pd.read_csv(tiles_dir / "splits" / "patient_splits.csv")
        table = man[man["patient_id"].isin(set(ps.loc[ps["split"] == "train", "patient_id"]))]
        rule = "tiles of training-split patients"

    strength, seed = 0.0, 0
    if args.shift:
        if args.dose is None:
            raise SystemExit("--shift needs --dose")
        table_doses = json.loads(Path(args.doses).read_text())
        strength = table_doses["doses"][args.shift][args.dose]["strength"]
        seed = table_doses["shift_seed"]
    info = extract_tier1(tiles_dir, table, args.out, rule, shift=args.shift, strength=strength,
                         seed=seed, num_workers=args.workers)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
