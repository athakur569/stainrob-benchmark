#!/usr/bin/env python3
"""Tier 1 features for both arms of the analysis plan, for both Tier 1b encoders (DR-018). Resumable.

    real arm       all QC-passing tiles, unshifted                      -> <out>/all/<encoder>/unshifted
    synthetic arm  validation + test patients' tiles (every fold's test_in), unshifted and under
                   every dose of the dose table                         -> <out>/val_test/<encoder>/<family>_<k>
Requires the committed analysis plan (test tiles).

    python scripts/extract_tier1_arms.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --out /media/abhi/camelyon17/derived/tier1_v1
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from stain_benchmark.monitor.extract import extract_tier1
from stain_benchmark.training.run import require_committed_plan

ENCODERS = ("densenet121", "resnet50")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--doses", default="configs/shift_doses_camelyon17.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    require_committed_plan()
    tiles_dir, out = Path(args.tiles), Path(args.out)
    man = pd.read_csv(tiles_dir / "manifest.csv")
    man = man[man["qc_pass"]]
    ps = pd.read_csv(tiles_dir / "splits" / "patient_splits.csv")
    val_test = man[man["patient_id"].isin(set(ps.loc[ps["split"].isin(["val", "test"]), "patient_id"]))]
    doses = json.loads(Path(args.doses).read_text())
    jobs = []
    for enc in ENCODERS:
        jobs.append((out / "all" / enc / "unshifted", man, "all QC-passing tiles", None, 0.0, enc))
        jobs.append((out / "val_test" / enc / "unshifted", val_test, "validation + test patients", None, 0.0, enc))
        for fam, levels in doses["doses"].items():
            for k, d in enumerate(levels):
                jobs.append((out / "val_test" / enc / f"{fam}_{k}", val_test, "validation + test patients",
                             fam, d["strength"], enc))
    todo = [j for j in jobs if not (j[0] / "info.json").exists()]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} extractions done; {len(todo)} to go", flush=True)
    for i, (d, table, rule, fam, strength, enc) in enumerate(todo, 1):
        extract_tier1(tiles_dir, table, d, rule, shift=fam, strength=strength, seed=doses["shift_seed"],
                      num_workers=args.workers, arch=enc, progress=False)
        print(f"[{i}/{len(todo)}] {d.relative_to(out)} ({len(table)} tiles)", flush=True)


if __name__ == "__main__":
    main()
