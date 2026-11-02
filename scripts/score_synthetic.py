#!/usr/bin/env python3
"""Synthetic arm: score each finished run's test_in tiles unshifted and under every dose (analysis plan §7).

For every run under --runs (condition/foldF/seedS with summary.json), writes to
<runs>/<condition>/fold<F>/seed<S>/synthetic/:
    pred_<family>_<k>.csv / features_<family>_<k>.npy     (and pred_unshifted / features_unshifted)
The model never sees shifted images in training; shifts are applied as tiles are read, before the
condition's stain normaliser (refitted exactly as in training: DR-013, fixed reference seed).
Requires the committed analysis plan (test_in is a test role). Resumable: finished runs are skipped.

    python scripts/score_synthetic.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --runs /media/abhi/camelyon17/derived/runs_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stain_benchmark.data.dataset import fold_table
from stain_benchmark.data.freeze import verify_freeze
from stain_benchmark.training.run import build_model, fit_reference, require_committed_plan, score_tiles


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--doses", default="configs/shift_doses_camelyon17.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None, help="restrict to runs whose path contains this text")
    ap.add_argument("--tag", default="", help="output subfolder suffix, e.g. for a post-hoc dose table")
    args = ap.parse_args()
    plan = require_committed_plan()
    tiles_dir, runs = Path(args.tiles), Path(args.runs)
    verify_freeze(tiles_dir)
    doses = json.loads(Path(args.doses).read_text())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = sorted(p.parent for p in runs.glob("*/fold*/seed*/summary.json"))
    if args.only:
        run_dirs = [r for r in run_dirs if args.only in str(r)]
    sub = "synthetic" + (f"_{args.tag}" if args.tag else "")
    todo = [r for r in run_dirs if not (r / sub / "done.json").exists()]
    print(f"{len(run_dirs) - len(todo)} of {len(run_dirs)} runs already scored; {len(todo)} to go", flush=True)
    for i, run in enumerate(todo, 1):
        out = run / sub
        s = json.loads((run / "summary.json").read_text())
        cond, holdout = s["condition"], s["holdout"]
        table = fold_table(tiles_dir, holdout)
        normalizer, ref = fit_reference(cond, tiles_dir, table)
        if normalizer is not None:
            saved = pd.read_csv(run / "reference_tiles.csv")["tile_id"].tolist()
            if saved != ref["tile_id"].tolist():
                raise SystemExit(f"{run}: refitted stain reference differs from the one used in training")
        norm_fn = normalizer.transform if normalizer is not None else None
        model = build_model(pretrained=False).to(dev)
        model.load_state_dict(torch.load(run / "model.pt", map_location=dev))
        test_in = table[table["role"] == "test_in"].reset_index(drop=True)
        out.mkdir(exist_ok=True)
        jobs = ([] if args.tag else [("unshifted", None)]) + [(f"{fam}_{k}", (fam, d["strength"], doses["shift_seed"]))
                                        for fam, levels in doses["doses"].items() for k, d in enumerate(levels)]
        for name, shift in jobs:
            pred, feats = score_tiles(model, tiles_dir, test_in, dev, normalizer=norm_fn, shift=shift,
                                      num_workers=args.workers, return_features=True)
            pred.to_csv(out / f"pred_{name}.csv", index=False)
            np.save(out / f"features_{name}.npy", feats)
        (out / "done.json").write_text(json.dumps({"analysis_plan_commit": plan, "n_tiles": len(test_in),
                                                   "conditions": [j[0] for j in jobs], "doses": args.doses}, indent=2))
        print(f"[{i}/{len(todo)}] {run.relative_to(runs)}: {len(jobs)} conditions × {len(test_in)} tiles", flush=True)


if __name__ == "__main__":
    main()
