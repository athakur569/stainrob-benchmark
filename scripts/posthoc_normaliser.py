#!/usr/bin/env python3
"""POST HOC (P5): why stain normalisation adds fragility to JPEG and striping — ΔG decomposition.

For E2 and E3 at every JPEG and striping dose, per fold, with one joint patient bootstrap (seed-averaged):
    ΔG(a)  deployed: the normaliser estimates its statistics on the shifted tile (pre-registered scoring)
    ΔG(b)  oracle:   statistics estimated on the clean tile, applied to the shifted tile (counterfactual)
    ΔG(a) − ΔG(b)    estimation effect (same model; positive = estimating on the damaged tile adds harm)
    ΔG(b) − ΔG(E1)   transform + model difference (E1's pre-registered ΔG at the same dose; different models)
Specification: docs/posthoc_analyses.md, P5.

    python scripts/posthoc_normaliser.py --runs /media/abhi/camelyon17/derived/runs_v1 \\
        --prereg /media/abhi/camelyon17/derived/analysis_v1 --out /media/abhi/camelyon17/derived/posthoc/p5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stain_benchmark.evaluation.bootstrap import joint_delta_g_seeds
from stain_benchmark.training.run import require_committed_plan

FOLDS, SEEDS, CONDITIONS = ("0", "1", "2", "3", "4"), (42, 99, 123), ("E2", "E3")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    plan = require_committed_plan()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pre = pd.read_csv(Path(args.prereg) / "synthetic_dg.csv")
    pre["fold"] = pre.fold.astype(str)
    e1 = pre[pre.condition == "E1"].set_index(["fold", "dose"])
    rows = []
    for f in FOLDS:
        for c in CONDITIONS:
            rds = [runs / c / f"fold{f}" / f"seed{s}" for s in SEEDS]
            names = sorted(p.name[5:-4] for p in (rds[0] / "synthetic_p5_oracle").glob("pred_*.csv"))
            refs = [pd.read_csv(rd / "synthetic" / "pred_unshifted.csv") for rd in rds]
            shifted = [{**{f"{n}|deployed": pd.read_csv(rd / "synthetic" / f"pred_{n}.csv") for n in names},
                        **{f"{n}|oracle": pd.read_csv(rd / "synthetic_p5_oracle" / f"pred_{n}.csv") for n in names}}
                       for rd in rds]
            pt, boots = joint_delta_g_seeds(refs, shifted, args.boot, seed=0)
            cols = list(shifted[0])
            for n in names:
                ia, ib = cols.index(f"{n}|deployed"), cols.index(f"{n}|oracle")
                diff = boots[:, ia] - boots[:, ib]
                e1_dg = e1.loc[(f, n), "delta_g"]
                rows.append({"fold": f, "condition": c, "dose": n, "family": n.rsplit("_", 1)[0],
                             "delta_e": pre[(pre.dose == n)].delta_e.iloc[0],
                             "dg_deployed": pt[f"{n}|deployed"], "dg_oracle": pt[f"{n}|oracle"], "dg_E1": e1_dg,
                             "estimation_effect": pt[f"{n}|deployed"] - pt[f"{n}|oracle"],
                             "estimation_low": np.quantile(diff, 0.025), "estimation_high": np.quantile(diff, 0.975),
                             "transform_plus_model": pt[f"{n}|oracle"] - e1_dg})
            print(f"fold {f} {c}: {len(names)} doses", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "decomposition.csv", index=False)
    s = d.groupby(["condition", "family", "delta_e"]).agg(
        dg_E1=("dg_E1", "mean"), dg_deployed=("dg_deployed", "mean"), dg_oracle=("dg_oracle", "mean"),
        estimation_effect=("estimation_effect", "mean"), transform_plus_model=("transform_plus_model", "mean"),
        folds_estimation_above_0=("estimation_low", lambda x: int((x > 0).sum())),
        folds_estimation_below_0=("estimation_high", lambda x: int((x < 0).sum()))).reset_index()
    s.to_csv(out / "summary.csv", index=False)
    (out / "provenance.txt").write_text(f"analysis plan {plan}; post-hoc P5; {args.boot} resamples\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
