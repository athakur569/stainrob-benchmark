#!/usr/bin/env python3
"""Fix the synthetic-shift dose table on training-split tiles (pixels only; no labels or models).

For every family and every target colour difference (mean CIELAB ΔE), find the strength that
reaches it on a fixed calibration sample: tiles from patients in the TRAINING split, 10 per
(centre, label), seed 0. SSIM at each dose is recorded as a secondary description (DR-015).

    python scripts/calibrate_shifts.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --out configs/shift_doses_camelyon17.json
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from stain_benchmark.data.shifts import EXTRA_FAMILIES, FAMILIES, calibrate_strength, mean_measure

TARGETS_DELTA_E = [2.0, 4.0, 6.0, 8.0, 10.0]          # common grid: every family, for matched comparisons
# Extensions after the dose-range pilot (DR-020): families with no clear, large response at ΔE 10
# get higher doses, up to what each family can reach (striping ~32, clipped ~44, JPEG ~17, stain ~12).
EXTENDED_DELTA_E = {"striping": [15.0, 20.0, 25.0, 30.0], "clipped": [15.0, 20.0, 30.0, 40.0],
                    "jpeg": [12.0, 15.0], "stain": [12.0]}
PER_CELL = 10          # tiles per (centre, label) in the calibration sample
SEED = 0


def calibration_sample(tiles_dir: Path) -> pd.DataFrame:
    man = pd.read_csv(tiles_dir / "manifest.csv")
    man = man[man["qc_pass"]]
    ps = pd.read_csv(tiles_dir / "splits" / "patient_splits.csv")
    pool = man[man["patient_id"].isin(set(ps.loc[ps["split"] == "train", "patient_id"]))]
    return (pool.groupby(["center", "label"], group_keys=False)
                .apply(lambda g: g.sample(min(PER_CELL, len(g)), random_state=SEED))
                .reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help=f"families to calibrate (post-hoc extra: {', '.join(EXTRA_FAMILIES)})")
    ap.add_argument("--targets", default=None, help="comma-separated ΔE targets (default: common grid + extensions)")
    args = ap.parse_args()
    tiles_dir, out = Path(args.tiles), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: the dose table is fixed; delete it deliberately to recalibrate")

    sample = calibration_sample(tiles_dir)
    imgs = [np.array(Image.open(tiles_dir / p).convert("RGB")) for p in sample["path"]]
    keys = sample["tile_id"].tolist()
    freeze = json.loads((tiles_dir / "freeze.json").read_text())

    doses = {}
    families = args.families.split(",")
    for fam in tqdm(families, desc="families"):
        doses[fam] = []
        targets = ([float(x) for x in args.targets.split(",")] if args.targets
                   else TARGETS_DELTA_E + EXTENDED_DELTA_E.get(fam, []))
        for target in targets:
            s, achieved = calibrate_strength(imgs, fam, target, measure="delta_e", seed=SEED, keys=keys)
            doses[fam].append({
                "target_delta_e": target, "strength": round(s, 4),
                "achieved_delta_e": round(achieved, 3),
                "achieved_ssim": round(mean_measure(imgs, fam, s, "ssim", SEED, keys), 4),
                "reached": bool(abs(achieved - target) <= 0.25),
                "grid": ("common" if target in TARGETS_DELTA_E else
                         "post hoc" if fam in EXTRA_FAMILIES else "extended (pilot, DR-020)"),
            })
            print(f"{fam:9s} ΔE {target:4.1f} → strength {s:7.3f} (ΔE {achieved:5.2f}, "
                  f"SSIM {doses[fam][-1]['achieved_ssim']:.3f})", flush=True)

    record = {
        "dataset": freeze.get("split", {}).get("project_id"),
        "freeze_master_sha256": freeze["master_sha256"],
        "created": date.today().isoformat(),
        "matching_measure": "mean per-pixel CIELAB ΔE76 between tile and shifted tile (DR-015)",
        "calibration_sample": {"rule": f"training-split patients; {PER_CELL} tiles per (centre, label); seed {SEED}",
                               "n_tiles": len(sample), "tile_ids": keys},
        "shift_seed": SEED,
        "doses": doses,
    }
    if any(f in EXTRA_FAMILIES for f in families):
        record["status"] = "POST HOC — not part of the pre-registered dose table (ISS-044)"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    unreached = [(f, d["target_delta_e"]) for f, ds in doses.items() for d in ds if not d["reached"]]
    print(f"\nwrote {out}; unreached targets: {unreached or 'none'}")


if __name__ == "__main__":
    main()
