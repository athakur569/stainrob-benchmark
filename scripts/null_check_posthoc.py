#!/usr/bin/env python3
"""POST HOC (ISS-046): realised false-alarm rate on PATIENT-split nulls, slide vs patient clusters.

The pre-registered null check (analysis plan §9.7, scripts/analyse.py) splits each fold's training
slides in half. CAMELYON17 has five slides per patient, so the same patients sit on both sides of
every split, which is an easier null than deployment, where new patients arrive. This diagnostic
splits the training PATIENTS in half instead and runs each tier twice on the same splits:

    clusters = slide     the pre-registered monitor (D2)
    clusters = patient   the correction proposed after fold 0 (ISS-046)

``--fit-on pooled`` fits the standardisation and principal components on both halves together instead of
the reference half only (ISS-049). ``--fit-on external`` splits the training patients into thirds instead:
one third fits the projection, the other two are reference and deployment (still training tiles only).
``--thirds`` uses that same three-way split for every arm (the first third unused unless external), so arms can be
compared on identical reference and deployment patients. Output columns are the same.

Training tiles only: no test data is read. Not part of the pre-registered analysis; reported as a
labelled post-hoc diagnostic.

    python scripts/null_check_posthoc.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --runs /media/abhi/camelyon17/derived/runs_v1 --tier1 /media/abhi/camelyon17/derived/tier1_v1 \\
        --out /media/abhi/camelyon17/derived/posthoc/null_patient_split.csv
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from stain_benchmark.data.dataset import fold_table
from stain_benchmark.evaluation.analysis import ALPHA, N_COMPONENTS, per_slide_subsample, tier1, tier2

FOLDS = ("0", "1", "2", "3", "4")
TIERS = ("tier1a", "tier1b_densenet121", "tier1b_resnet50", "tier2")


def load_tier1(base: Path, enc: str):
    d = base / "all" / enc / "unshifted"
    tiles = pd.read_csv(d / "tiles.csv")
    pos = pd.Series(np.arange(len(tiles)), index=tiles["tile_id"].astype(str))
    return pos, pd.read_csv(d / "tier1a.csv").drop(columns="tile_id").to_numpy(), np.load(d / "tier1b.npy")


def job(args) -> dict:
    fold, tier, tiles_dir, runs, tier1_dir, n_splits, fit_on, thirds = args
    if tier == "tier2":
        p = pd.read_csv(runs / "E1" / f"fold{fold}" / "seed42" / "predictions_train.csv")
        X = np.load(runs / "E1" / f"fold{fold}" / "seed42" / "features_train.npy").astype(np.float64)
        slides, patients = p["slide_id"].astype(str).to_numpy(), p["patient_id"].astype(str).to_numpy()
    else:
        tab = fold_table(tiles_dir, fold)
        tab = tab[tab["role"] == "train"]
        enc = "densenet121" if tier in ("tier1a", "tier1b_densenet121") else "resnet50"
        pos, a1a, a1b = load_tier1(tier1_dir, enc)
        rows = pos.loc[tab["tile_id"].astype(str)].to_numpy()
        X = (a1a if tier == "tier1a" else a1b)[rows].astype(np.float64)
        slides, patients = tab["slide_id"].astype(str).to_numpy(), tab["patient_id"].astype(str).to_numpy()
    sub = per_slide_subsample(slides)
    X, slides, patients = X[sub], slides[sub], patients[sub]
    ncomp = None if tier == "tier1a" else N_COMPONENTS
    upat = np.unique(patients)
    flags = {"slide": 0, "patient": 0}
    for i in range(n_splits):
        rng = np.random.default_rng([int(fold), i])           # same splits for every tier and cluster type
        order = rng.permutation(upat)
        ext = None
        if thirds or fit_on == "external":                    # thirds: projection / reference / deployment
            k = len(upat) // 3
            fit_rows = np.isin(patients, order[:k])
            keep, m = ~fit_rows, np.isin(patients, order[k:2 * k])
            ext = X[fit_rows] if fit_on == "external" else None
        else:
            keep = np.ones(len(patients), bool)
            m = np.isin(patients, order[: len(upat) // 2])
        for kind, ids in (("slide", slides), ("patient", patients)):
            if tier == "tier2":
                flags[kind] += tier2(X[m], X[~m & keep], ids[m], ids[~m & keep], seed=i)["flagged"]
            else:
                flags[kind] += tier1(X[m], X[~m & keep], ids[m], ids[~m & keep], ncomp, n_permutations=99,
                                     seed=i, fit_on=fit_on, external=ext)["flagged"]
    return {"fold": fold, "tier": tier, "fit_on": fit_on, "split": "thirds" if thirds or fit_on == "external" else "halves",
            "patients": len(upat), "splits": n_splits, "nominal_alpha": ALPHA,
            "false_alarm_slide_clusters": flags["slide"] / n_splits,
            "false_alarm_patient_clusters": flags["patient"] / n_splits}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tier1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--fit-on", default="reference", choices=("reference", "pooled", "external"))
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--thirds", action="store_true", help="same projection/reference/deployment thirds for every arm")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(f, t, Path(args.tiles), Path(args.runs), Path(args.tier1), args.splits, args.fit_on, args.thirds)
            for f in FOLDS for t in args.tiers.split(",")]
    rows = []
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            rows.append(r)
            print(f"[{k}/{len(jobs)}] fold {r['fold']} {r['tier']}: slide clusters {r['false_alarm_slide_clusters']:.3f}, "
                  f"patient clusters {r['false_alarm_patient_clusters']:.3f}", flush=True)
    pd.DataFrame(rows).sort_values(["tier", "fold"]).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
