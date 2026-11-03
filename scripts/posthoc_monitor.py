#!/usr/bin/env python3
"""POST HOC (P3, with P2's stain_batch doses): the corrected monitor, beside the pre-registered one.

Changes from the pre-registered monitor (docs/posthoc_analyses.md, P3):
  1. reference = held-out in-domain tiles (test_in), not the training tiles the classifier has fitted (ISS-045);
  2. clusters = patients, not slides (ISS-046);
  3. Tier 1 standardisation/projection fitted on both batches pooled, or on external data — the fold's training
     tiles — (ISS-049); both are reported, the choice is made by power at a held false-alarm rate.

Real arm: reference = all test_in tiles, deployment = test_cross. Synthetic arm: test_in patients are halved
(stratified by patient label; 10 halvings, seeds 0–9); half A unshifted is the reference, half B under each dose is
deployed, and unshifted B against A is the false-alarm check. ΔG is the pre-registered one (all of test_in), read from
the pre-registered analysis output; for P2's stain_batch doses it is computed here the same way.

    python scripts/posthoc_monitor.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --runs /media/abhi/camelyon17/derived/runs_v1 --tier1 /media/abhi/camelyon17/derived/tier1_v1 \\
        --prereg /media/abhi/camelyon17/derived/analysis_v1 --out /media/abhi/camelyon17/derived/posthoc/p3
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import load_tier1, rows_for  # noqa: E402

from stain_benchmark.data.dataset import fold_table  # noqa: E402
from stain_benchmark.evaluation.analysis import (  # noqa: E402
    N_COMPONENTS, per_slide_subsample, spearman, tier1, tier2)
from stain_benchmark.evaluation.bootstrap import joint_delta_g_seeds  # noqa: E402
from stain_benchmark.training.run import require_committed_plan  # noqa: E402

CONDITIONS, FOLDS, SEEDS = ("E1", "E2", "E3", "E5"), ("0", "1", "2", "3", "4"), (42, 99, 123)
ENCODERS = {"tier1b": "densenet121", "tier1b_resnet50": "resnet50"}
FIT_ON = ("pooled", "external")
N_HALVINGS = 10
G: dict = {}


def init(g):
    G.update(g)


def halves(test_in: pd.DataFrame, fold: str, h: int) -> np.ndarray:
    """Boolean mask over test_in rows: True = half A (reference). Patients halved within patient class."""
    pat = test_in.groupby("patient_id").label.max()
    rng = np.random.default_rng([int(fold), h])
    a = []
    for cls in (0, 1):
        ids = rng.permutation(np.sort(pat[pat == cls].index.to_numpy()))
        a += list(ids[: len(ids) // 2])
    return test_in.patient_id.isin(a).to_numpy()


def batch_dir(name: str) -> str:
    return "synthetic_stain_batch" if name.startswith("stain_batch") else "synthetic"


# ----------------------------------------------------------------------------------------------- Tier 1
def tier1_job(args) -> list[dict]:
    fold, batch = args
    table = G["tables"][fold]
    tr, ti, tc = (table[table.role == r].reset_index(drop=True) for r in ("train", "test_in", "test_cross"))
    out = []
    for key, enc in ENCODERS.items():
        all_tiles, a1a, a1b = G["t1_all"][enc]
        ext_sub = per_slide_subsample(tr.slide_id.to_numpy())
        ext_rows = rows_for(all_tiles, tr.tile_id)[ext_sub]
        tiers = [(key, a1b, N_COMPONENTS)] + ([("tier1a", a1a, None)] if key == "tier1b" else [])
        if batch == "real":
            pairs = [(None, ti, rows_for(all_tiles, ti.tile_id), tc, rows_for(all_tiles, tc.tile_id), a1a, a1b)]
        else:
            vt_tiles, v1a, v1b = G["t1_vt"][(enc, "unshifted")]
            st_tiles, s1a, s1b = G["t1_vt"][(enc, batch)]
            pairs = []
            for h in range(N_HALVINGS):
                m = halves(ti, fold, h)
                pairs.append((h, ti[m], rows_for(vt_tiles, ti[m].tile_id), ti[~m], rows_for(st_tiles, ti[~m].tile_id),
                              (v1a, s1a), (v1b, s1b)))
        for h, rtab, rrows, dtab, drows, src_a, src_b in pairs:
            rsub, dsub = per_slide_subsample(rtab.slide_id.to_numpy()), per_slide_subsample(dtab.slide_id.to_numpy())
            for name, feats_all, ncomp in tiers:
                if batch == "real":
                    R, D = feats_all[rrows][rsub], feats_all[drows][dsub]
                else:
                    pair = src_a if name == "tier1a" else src_b
                    R, D = pair[0][rrows][rsub], pair[1][drows][dsub]
                ext = feats_all[ext_rows]
                for fit_on in FIT_ON:
                    r = tier1(R, D, rtab.patient_id.to_numpy()[rsub], dtab.patient_id.to_numpy()[dsub], ncomp, G["perm"],
                              seed=0 if h is None else h, fit_on=fit_on, external=ext if fit_on == "external" else None)
                    out.append({"fold": fold, "batch": batch, "halving": h, "tier": name, "fit_on": fit_on,
                                "ref_patients": rtab.patient_id.nunique(), "ref_tiles": len(rsub), **r})
    return out


# ----------------------------------------------------------------------------------------------- Tier 2
def tier2_job(args) -> list[dict]:
    c, f, s = args
    rd = G["runs"] / c / f"fold{f}" / f"seed{s}"
    out = []
    pin, fin = pd.read_csv(rd / "predictions_test_in.csv"), np.load(rd / "features_test_in.npy").astype(np.float64)
    pcr, fcr = pd.read_csv(rd / "predictions_test_cross.csv"), np.load(rd / "features_test_cross.npy").astype(np.float64)
    rs, ds = per_slide_subsample(pin.slide_id.to_numpy()), per_slide_subsample(pcr.slide_id.to_numpy())
    lg = ["logit_0", "logit_1"]
    r = tier2(fin[rs], fcr[ds], pin.patient_id.to_numpy()[rs], pcr.patient_id.to_numpy()[ds],
              pin[lg].to_numpy()[rs], pcr[lg].to_numpy()[ds])
    out.append({"condition": c, "fold": f, "seed": s, "batch": "real", "halving": None,
                "ref_patients": pin.patient_id.nunique(), "ref_tiles": len(rs), **r})
    base_p = pd.read_csv(rd / "synthetic" / "pred_unshifted.csv")
    base_f = np.load(rd / "synthetic" / "features_unshifted.npy").astype(np.float64)
    for h in range(N_HALVINGS):
        m = halves(base_p, f, h)
        A, Af = base_p[m].reset_index(drop=True), base_f[m]
        asub = per_slide_subsample(A.slide_id.to_numpy())
        for name in ["unshifted", *G["batches"]]:
            P = pd.read_csv(rd / batch_dir(name) / f"pred_{name}.csv")
            F = np.load(rd / batch_dir(name) / f"features_{name}.npy").astype(np.float64)
            if not np.array_equal(P.tile_id.to_numpy(), base_p.tile_id.to_numpy()):
                raise ValueError(f"{rd}/{name}: tiles differ from the unshifted reference")
            B, Bf = P[~m].reset_index(drop=True), F[~m]
            bsub = per_slide_subsample(B.slide_id.to_numpy())
            r = tier2(Af[asub], Bf[bsub], A.patient_id.to_numpy()[asub], B.patient_id.to_numpy()[bsub],
                      A[lg].to_numpy()[asub], B[lg].to_numpy()[bsub], seed=h)
            out.append({"condition": c, "fold": f, "seed": s, "batch": name, "halving": h,
                        "ref_patients": A.patient_id.nunique(), "ref_tiles": len(asub), **r})
    return out


# ----------------------------------------------------------------------------------------------- P2's ΔG
def stain_batch_dg_job(args) -> list[dict]:
    c, f = args
    names = [n for n in G["batches"] if n.startswith("stain_batch")]
    rds = [G["runs"] / c / f"fold{f}" / f"seed{s}" for s in SEEDS]
    refs = [pd.read_csv(rd / "synthetic" / "pred_unshifted.csv") for rd in rds]
    shifted = [{n: pd.read_csv(rd / "synthetic_stain_batch" / f"pred_{n}.csv") for n in names} for rd in rds]
    pt, boots = joint_delta_g_seeds(refs, shifted, G["boot"], seed=0)
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    return [{"fold": f, "condition": c, "dose": n, "delta_g": pt[n], "dg_low": lo[i], "dg_high": hi[i]}
            for i, n in enumerate(names)]


def harm_label(low: float, est: float, high: float) -> str:
    if low > 0:
        return "harm detected"
    if est <= 0.01 and high <= 0.03:
        return "no harm detected — clean"
    return "no harm detected — inconclusive"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for a in ("--tiles", "--runs", "--tier1", "--prereg", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--doses", default="configs/shift_doses_camelyon17.json")
    ap.add_argument("--extra-doses", default="configs/shift_doses_camelyon17_posthoc_stain_batch.json")
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--max-batches", type=int, default=None, help="dry run: only the first N shifted batches")
    args = ap.parse_args()
    plan = require_committed_plan()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    folds, conds = tuple(args.folds.split(",")), tuple(args.conditions.split(","))
    meta = {}
    for path in (args.doses, args.extra_doses):
        d = json.loads(Path(path).read_text())
        meta.update({f"{fam}_{k}": {"family": fam, "delta_e": x["target_delta_e"]}
                     for fam, levels in d["doses"].items() for k, x in enumerate(levels)})
    batches = list(meta)[: args.max_batches] if args.max_batches else list(meta)
    tiles_dir = Path(args.tiles)
    g = {"runs": Path(args.runs), "tables": {f: fold_table(tiles_dir, f) for f in folds}, "perm": args.perm,
         "boot": args.boot, "batches": batches}
    g["t1_all"] = {enc: load_tier1(Path(args.tier1), enc, "all", "unshifted") for enc in ENCODERS.values()}
    g["t1_vt"] = {(enc, n): load_tier1(Path(args.tier1), enc, "val_test", n)
                  for enc in ENCODERS.values() for n in ["unshifted", *batches]}
    init(g)
    print(f"post-hoc P3 under plan {plan[:7]}: folds {folds}, {len(batches)} shifted batches, "
          f"{N_HALVINGS} halvings", flush=True)
    with ProcessPoolExecutor(args.jobs, initializer=init, initargs=(g,)) as ex:
        t1 = pd.DataFrame([r for rs in ex.map(tier1_job, [(f, b) for f in folds for b in ["real", "unshifted", *batches]])
                           for r in rs])
        t1.to_csv(out / "tier1.csv", index=False)
        print("Tier 1 done", flush=True)
        t2 = pd.DataFrame([r for rs in ex.map(tier2_job, [(c, f, s) for f in folds for c in conds for s in SEEDS])
                           for r in rs])
        t2.to_csv(out / "tier2.csv", index=False)
        print("Tier 2 done", flush=True)
        sb = [n for n in batches if n.startswith("stain_batch")]
        dg_sb = pd.DataFrame([r for rs in ex.map(stain_batch_dg_job, [(c, f) for f in folds for c in conds])
                              for r in rs]) if sb else pd.DataFrame()
    dg = pd.read_csv(Path(args.prereg) / "synthetic_dg.csv")[["fold", "condition", "dose", "delta_g", "dg_low", "dg_high"]]
    dg["fold"] = dg.fold.astype(str)
    dg = pd.concat([dg, dg_sb], ignore_index=True)
    dg["harm"] = [harm_label(lo, e, hi) for lo, e, hi in zip(dg.dg_low, dg.delta_g, dg.dg_high)]
    dg.to_csv(out / "delta_g_with_stain_batch.csv", index=False)
    summarise(t1, t2, dg, meta, out)
    (out / "provenance.json").write_text(json.dumps({"analysis_plan_commit": plan, "posthoc": "P3 (+P2 batches)",
                                                     "halvings": N_HALVINGS, "perm": args.perm, "boot": args.boot,
                                                     "folds": folds, "conditions": conds}, indent=2))
    print(f"wrote {out}", flush=True)


def summarise(t1: pd.DataFrame, t2: pd.DataFrame, dg: pd.DataFrame, meta: dict, out: Path) -> None:
    """False-alarm rate beside power for every arm (P3 reporting rule), ρ with ΔG, and real-arm alarms."""
    t1["fold"], t2["fold"], dg["fold"] = t1.fold.astype(str), t2.fold.astype(str), dg.fold.astype(str)
    syn1 = t1[t1.batch != "real"]
    syn2 = t2[t2.batch != "real"].copy()
    # Tier 2: seed majority (D6) per condition × fold × batch × halving
    maj = (syn2.groupby(["condition", "fold", "batch", "halving"])
           .agg(flagged=("flagged", lambda x: x.sum() >= 2), mahalanobis=("mahalanobis", "mean")).reset_index())
    rows = []
    for c in sorted(dg.condition.unique()):
        h = dg[dg.condition == c][["fold", "dose", "harm", "delta_g"]]
        arms = [(f"{t} ({fo})", syn1[(syn1.tier == t) & (syn1.fit_on == fo)].assign(condition=c))
                for t in ("tier1a", "tier1b", "tier1b_resnet50") for fo in FIT_ON]
        arms.append(("tier2", maj[maj.condition == c]))
        for arm, d in arms:
            if d.empty:
                continue
            null = d[d.batch == "unshifted"].flagged.mean()
            m = d[d.batch != "unshifted"].merge(h, left_on=["fold", "batch"], right_on=["fold", "dose"])
            sig = "mmd2" if arm.startswith("tier1") else "mahalanobis"
            med = m.groupby(["fold", "batch"]).agg(signal=(sig, "median"), delta_g=("delta_g", "first")).reset_index()
            prereg = med[~med.batch.str.startswith("stain_batch")]
            rho = pd.Series({f: spearman(x.signal, x.delta_g) for f, x in prereg.groupby("fold")}, dtype=float)
            rows.append({"condition": c, "arm": arm, "false_alarm_unshifted": null,
                         "power_on_harm": m[m.harm == "harm detected"].flagged.mean(),
                         "alarm_on_clean": m[m.harm == "no harm detected — clean"].flagged.mean(),
                         "rho_with_dg_min": rho.min(), "rho_with_dg_median": rho.median(), "rho_with_dg_max": rho.max(),
                         "n_harm_cells": int((m.harm == "harm detected").sum()),
                         "n_clean_cells": int((m.harm == "no harm detected — clean").sum())})
    pd.DataFrame(rows).to_csv(out / "summary_power_false_alarm.csv", index=False)
    # power by family and dose
    bd = []
    for arm_name, d in [(f"{t} ({fo})", syn1[(syn1.tier == t) & (syn1.fit_on == fo)]) for t in ("tier1a", "tier1b", "tier1b_resnet50")
                        for fo in FIT_ON] + [(f"tier2 {c}", maj[maj.condition == c]) for c in sorted(maj.condition.unique())]:
        r = d[d.batch != "unshifted"].groupby("batch").flagged.mean()
        bd += [{"arm": arm_name, "batch": b, **meta.get(b, {}), "alarm_rate": v} for b, v in r.items()]
    pd.DataFrame(bd).to_csv(out / "alarm_rate_by_dose.csv", index=False)
    # real arm
    r1 = t1[t1.batch == "real"][["fold", "tier", "fit_on", "mmd2", "flagged", "ref_patients", "ref_tiles"]]
    r1.to_csv(out / "real_arm_tier1.csv", index=False)
    r2 = (t2[t2.batch == "real"].groupby(["condition", "fold"])
          .agg(mahalanobis=("mahalanobis", "mean"), tier2_flag_seeds=("flagged", "sum"),
               ref_patients=("ref_patients", "first"), ref_tiles=("ref_tiles", "first")).reset_index())
    r2.to_csv(out / "real_arm_tier2.csv", index=False)


if __name__ == "__main__":
    main()
