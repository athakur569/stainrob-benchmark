#!/usr/bin/env python3
"""The pre-registered analysis (docs/analysis_plan.md with deviations D1–D5).

Reads the finished runs, their synthetic-arm scores and the Tier 1 features, and writes CSV tables
to --out:
    real_arm.csv          per fold × condition: ΔG (AUROC, calibration slope) seed-averaged with
                          patient-clustered intervals; D5 label; monitor signals and alarms on test_cross
    synthetic_dg.csv      per fold × condition × dose: seed-averaged ΔG with intervals
    synthetic_signals.csv per fold × condition × dose: monitor signals and alarms, ΔE, SSIM
    h1_h2.csv             per fold × condition × predictor: Spearman ρ with ΔG and differences vs ΔE / 1−SSIM
    h3.csv                per fold × condition: the three parts of the H3 prediction
    matched_slice.csv     D1: per fold × condition × common-grid ΔE: family orderings by each predictor
    alarms.csv            §9.6: alarm proportions by family and dose
    null_check.csv        §9.7: realised false-alarm rate per fold and tier on null splits
    patients.csv          per fold × condition × seed × tumour patient: mean score, fraction below 0.5
Run with fewer threads per process: OMP_NUM_THREADS=1 python scripts/analyse.py ...
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from stain_benchmark.data.dataset import fold_table
from stain_benchmark.evaluation.analysis import (
    ALPHA,
    N_COMPONENTS,
    ordering_concordance,
    per_slide_subsample,
    spearman,
    tier1,
    tier2,
)
from stain_benchmark.evaluation.bootstrap import (
    compare_predictors,
    delta_g_seeds_interval,
    joint_delta_g_seeds,
)
from stain_benchmark.training.run import require_committed_plan

CONDITIONS = ("E1", "E2", "E3", "E5")
FOLDS = ("0", "1", "2", "3", "4")
SEEDS = (42, 99, 123)
ENCODERS = {"tier1b": "densenet121", "tier1b_resnet50": "resnet50"}
COMMON_GRID = (2.0, 4.0, 6.0, 8.0, 10.0)
G = {}  # globals for worker processes


def dose_meta(doses_path: Path) -> pd.DataFrame:
    d = json.loads(doses_path.read_text())
    rows = [{"dose": f"{fam}_{k}", "family": fam, "delta_e": x["target_delta_e"],
             "achieved_delta_e": x["achieved_delta_e"], "ssim": x["achieved_ssim"]}
            for fam, levels in d["doses"].items() for k, x in enumerate(levels)]
    return pd.DataFrame(rows).set_index("dose")


def label_fold(iv) -> str:
    """D5 label from the seed-averaged ΔG interval."""
    if iv.low > 0:
        return "harm detected"
    if iv.estimate <= 0.01 and iv.high <= 0.03:
        return "no harm detected — clean"
    return "no harm detected — inconclusive"


def run_dir(c, f, s) -> Path:
    return G["runs"] / c / f"fold{f}" / f"seed{s}"


def load_tier1(base: Path, enc: str, subset: str, name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    d = base / subset / enc / name
    tiles = pd.read_csv(d / "tiles.csv")
    f1a = pd.read_csv(d / "tier1a.csv").drop(columns="tile_id").to_numpy()
    f1b = np.load(d / "tier1b.npy").astype(np.float64)
    return tiles, f1a, f1b


def rows_for(tiles: pd.DataFrame, tile_ids: pd.Series) -> np.ndarray:
    pos = pd.Series(np.arange(len(tiles)), index=tiles["tile_id"].astype(str))
    return pos.loc[tile_ids.astype(str)].to_numpy()


# --------------------------------------------------------------------------- Tier 1 (condition-independent)
def tier1_job(args) -> list[dict]:
    """Tier 1a and 1b signals for one fold and one deployment batch (real arm or one dose)."""
    fold, batch = args
    table = G["tables"][fold]
    ref_tab = table[table["role"] == "train"]
    out = []
    for key, enc in ENCODERS.items():
        all_tiles, a1a, a1b = G["t1_all"][enc]
        ref_rows = rows_for(all_tiles, ref_tab["tile_id"])
        ref_sub = per_slide_subsample(ref_tab["slide_id"].to_numpy())
        if batch == "real":
            dep_tab = table[table["role"] == "test_cross"]
            dep_rows = rows_for(all_tiles, dep_tab["tile_id"])
            d1a, d1b = a1a[dep_rows], a1b[dep_rows]
        else:
            vt_tiles, v1a, v1b = G["t1_vt"][(enc, batch)]
            dep_tab = table[table["role"] == "test_in"]
            dep_rows = rows_for(vt_tiles, dep_tab["tile_id"])
            d1a, d1b = v1a[dep_rows], v1b[dep_rows]
        dep_sub = per_slide_subsample(dep_tab["slide_id"].to_numpy())
        rs, ds = ref_tab["slide_id"].to_numpy()[ref_sub], dep_tab["slide_id"].to_numpy()[dep_sub]
        if key == "tier1b":  # Tier 1a is the same for both encoder folders; compute once
            r = tier1(a1a[ref_rows][ref_sub], d1a[dep_sub], rs, ds, None, G["perm"])
            out.append({"fold": fold, "batch": batch, "tier": "tier1a", **r})
        r = tier1(a1b[ref_rows][ref_sub], d1b[dep_sub], rs, ds, N_COMPONENTS, G["perm"])
        out.append({"fold": fold, "batch": batch, "tier": key, **r})
    return out


# --------------------------------------------------------------------------- per run: Tier 2
def tier2_job(args) -> list[dict]:
    c, f, s = args
    rd = run_dir(c, f, s)
    ptr = pd.read_csv(rd / "predictions_train.csv")
    ftr = np.load(rd / "features_train.npy").astype(np.float64)
    rsub = per_slide_subsample(ptr["slide_id"].to_numpy())
    ref, rs, rl = ftr[rsub], ptr["slide_id"].to_numpy()[rsub], ptr[["logit_0", "logit_1"]].to_numpy()[rsub]
    batches = [("real", rd / "predictions_test_cross.csv", rd / "features_test_cross.npy")]
    batches += [(name, rd / "synthetic" / f"pred_{name}.csv", rd / "synthetic" / f"features_{name}.npy")
                for name in ["unshifted", *G["meta"].index]]
    out = []
    for name, pp, fp in batches:
        pred, feat = pd.read_csv(pp), np.load(fp).astype(np.float64)
        dsub = per_slide_subsample(pred["slide_id"].to_numpy())
        r = tier2(ref, feat[dsub], rs, pred["slide_id"].to_numpy()[dsub], rl, pred[["logit_0", "logit_1"]].to_numpy()[dsub])
        out.append({"condition": c, "fold": f, "seed": s, "batch": name, **r})
    return out


# --------------------------------------------------------------------------- per fold × condition: ΔG
def dg_job(args) -> dict:
    c, f = args
    pin = [pd.read_csv(run_dir(c, f, s) / "predictions_test_in.csv") for s in SEEDS]
    pcr = [pd.read_csv(run_dir(c, f, s) / "predictions_test_cross.csv") for s in SEEDS]
    real_auc = delta_g_seeds_interval(pin, pcr, G["boot"], seed=0)
    real_slope = delta_g_seeds_interval(pin, pcr, G["boot"], seed=0, metric="slope")
    names = list(G["meta"].index)
    refs = [pd.read_csv(run_dir(c, f, s) / "synthetic" / "pred_unshifted.csv") for s in SEEDS]
    shifted = [{n: pd.read_csv(run_dir(c, f, s) / "synthetic" / f"pred_{n}.csv") for n in names} for s in SEEDS]
    pt, boots = joint_delta_g_seeds(refs, shifted, G["boot"], seed=0)
    spt, sboots = joint_delta_g_seeds(refs, shifted, G["boot"], seed=0, metric="slope")
    np.savez_compressed(G["out"] / "boots" / f"{c}_fold{f}.npz", auroc=boots, slope=sboots, names=np.array(names))
    patients = []
    for s, a, b in zip(SEEDS, pin, pcr):
        for role, p in (("test_in", a), ("test_cross", b)):
            per = p[p.label == 1].groupby("patient_id")["p_tumour"].agg(
                tiles="size", mean_p="mean", frac_below_half=lambda x: float((x < 0.5).mean()))
            patients += [{"condition": c, "fold": f, "seed": s, "role": role, "patient_id": pid, **row}
                         for pid, row in per.to_dict("index").items()]
    return {"c": c, "f": f, "real_auc": real_auc, "real_slope": real_slope, "syn_point": pt, "syn_boots": boots,
            "syn_slope_point": spt, "syn_slope_boots": sboots, "patients": patients}


def init(g):
    G.update(g)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tier1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--doses", default="configs/shift_doses_camelyon17.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--null-splits", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--folds", default=",".join(FOLDS))
    ap.add_argument("--conditions", default=",".join(CONDITIONS), help="subset, e.g. for an early pipeline check")
    args = ap.parse_args()
    plan = require_committed_plan()
    out = Path(args.out); (out / "boots").mkdir(parents=True, exist_ok=True)
    folds = tuple(args.folds.split(","))
    conds = tuple(args.conditions.split(","))
    meta = dose_meta(Path(args.doses))
    tiles_dir = Path(args.tiles)
    tables = {f: fold_table(tiles_dir, f) for f in folds}
    g = {"runs": Path(args.runs), "tier1": Path(args.tier1), "out": out, "meta": meta, "tables": tables,
         "boot": args.boot, "perm": args.perm}
    g["t1_all"] = {enc: load_tier1(g["tier1"], enc, "all", "unshifted") for enc in ENCODERS.values()}
    g["t1_vt"] = {(enc, n): load_tier1(g["tier1"], enc, "val_test", n) for enc in ENCODERS.values()
                  for n in ["unshifted", *meta.index]}
    init(g)
    print(f"analysis under plan {plan[:7]}: folds {folds}, {len(meta)} doses, {args.boot} bootstrap resamples", flush=True)

    with ProcessPoolExecutor(args.jobs, initializer=init, initargs=(g,)) as ex:
        t1 = pd.DataFrame([r for rs in ex.map(tier1_job, [(f, b) for f in folds for b in ["real", "unshifted", *meta.index]])
                           for r in rs])
        print("Tier 1 done", flush=True)
        t2 = pd.DataFrame([r for rs in ex.map(tier2_job, [(c, f, s) for f in folds for c in conds for s in SEEDS])
                           for r in rs])
        print("Tier 2 done", flush=True)
        dgs = list(ex.map(dg_job, [(c, f) for f in folds for c in conds]))
        print("ΔG done", flush=True)
    t1.to_csv(out / "tier1_all.csv", index=False)
    t2.to_csv(out / "tier2_all.csv", index=False)
    pd.DataFrame([p for d in dgs for p in d["patients"]]).to_csv(out / "patients.csv", index=False)

    # ---- real arm (descriptive) with D3/D5 alarm scoring
    t1w = t1.pivot_table(index=["fold", "batch"], columns="tier", values=["mmd2", "flagged"], aggfunc="first")
    real_rows, syn_dg, syn_sig, h12, h3, sl = [], [], [], [], [], []
    for d in dgs:
        c, f = d["c"], d["f"]
        lab = label_fold(d["real_auc"])
        t2r = t2[(t2.condition == c) & (t2.fold == f) & (t2.batch == "real")]
        f1b, f1a = bool(t1w.loc[(f, "real"), ("flagged", "tier1b")]), bool(t1w.loc[(f, "real"), ("flagged", "tier1a")])
        and_seed = [f1b and fl for fl in t2r.flagged]
        score = {"harm detected": "detection" if np.mean(and_seed) > 0.5 else "miss",
                 "no harm detected — clean": "false alarm" if np.mean(and_seed) > 0.5 else "correct silence",
                 "no harm detected — inconclusive": "ambiguous"}[lab]
        real_rows.append({"fold": f, "condition": c, "delta_g": d["real_auc"].estimate, "dg_low": d["real_auc"].low,
                          "dg_high": d["real_auc"].high, "label_D5": lab,
                          "slope_delta_g": d["real_slope"].estimate, "slope_low": d["real_slope"].low,
                          "slope_high": d["real_slope"].high, "slope_resamples_defined": d["real_slope"].n_boot,
                          "tier1a_mmd2": t1w.loc[(f, "real"), ("mmd2", "tier1a")], "tier1a_flag": f1a,
                          "tier1b_mmd2": t1w.loc[(f, "real"), ("mmd2", "tier1b")], "tier1b_flag": f1b,
                          "tier2_mahalanobis": t2r.mahalanobis.mean(), "tier2_flag_seeds": int(t2r.flagged.sum()),
                          "and_rule_flag_seeds": int(sum(and_seed)), "alarm_scored_D5": score})

        # ---- synthetic arm
        names = list(meta.index)
        lo, hi = np.quantile(d["syn_boots"], [0.025, 0.975], axis=0)
        slo, shi = np.nanquantile(d["syn_slope_boots"], [0.025, 0.975], axis=0)    # D7: undefined slopes dropped
        s_def = np.isfinite(d["syn_slope_boots"]).sum(axis=0)
        t2c = t2[(t2.condition == c) & (t2.fold == f)].groupby("batch").agg(
            mahalanobis=("mahalanobis", "mean"), energy=("energy", "mean"), tier2_flag_seeds=("flagged", "sum"))
        pred = {"tier2": t2c.loc[names, "mahalanobis"].set_axis(names),
                "tier1b": pd.Series([t1w.loc[(f, n), ("mmd2", "tier1b")] for n in names], index=names),
                "tier1a": pd.Series([t1w.loc[(f, n), ("mmd2", "tier1a")] for n in names], index=names),
                "tier1b_resnet50": pd.Series([t1w.loc[(f, n), ("mmd2", "tier1b_resnet50")] for n in names], index=names),
                "delta_e": meta.loc[names, "achieved_delta_e"], "one_minus_ssim": 1 - meta.loc[names, "ssim"]}
        for i, n in enumerate(names):
            syn_dg.append({"fold": f, "condition": c, "dose": n, **meta.loc[n].to_dict(), "delta_g": d["syn_point"][n],
                           "dg_low": lo[i], "dg_high": hi[i], "slope_delta_g": d["syn_slope_point"][n],
                           "slope_low": slo[i], "slope_high": shi[i], "slope_resamples_defined": int(s_def[i])})
            syn_sig.append({"fold": f, "condition": c, "dose": n, **{k: v[n] for k, v in pred.items()},
                            "tier1a_flag": bool(t1w.loc[(f, n), ("flagged", "tier1a")]),
                            "tier1b_flag": bool(t1w.loc[(f, n), ("flagged", "tier1b")]),
                            "tier2_flag_seeds": int(t2c.loc[n, "tier2_flag_seeds"])})
        for subset, keep in (("all 36", names), ("common grid 25", [n for n in names if meta.loc[n, "delta_e"] <= 10])):
            pt, bt = d["syn_point"][keep], d["syn_boots"][:, [names.index(n) for n in keep]]
            for base in ("delta_e", "one_minus_ssim"):
                order = {base: pred[base][keep], **{k: pred[k][keep] for k in ("tier2", "tier1b", "tier1a", "tier1b_resnet50")}}
                cmp = compare_predictors(pt, bt, order)
                for _, row in cmp.iterrows():
                    rho_b = np.array([spearman(order[row.predictor].to_numpy(), r) for r in bt])
                    h12.append({"fold": f, "condition": c, "subset": subset, "baseline": base, **row.to_dict(),
                                "rho_one_sided_low": float(np.nanquantile(rho_b, ALPHA))})
        # ---- H3
        stain = [n for n in names if meta.loc[n, "family"] == "stain"]
        sig_un = t1w.loc[(f, "unshifted"), ("mmd2", "tier1a")]
        h3.append({"fold": f, "condition": c,
                   "tier1a_rho_with_stain_dose": spearman(range(len(stain) + 1), [sig_un, *pred["tier1a"][stain]]),
                   "stain_dg_max": d["syn_point"][stain].max(),
                   **{f"stain_over_blur_dg_at_dE{int(e)}": d["syn_point"][f"stain_{k}"] / d["syn_point"][f"blur_{k}"]
                      for k, e in enumerate(COMMON_GRID)}})
        # ---- D1 matched slice
        for k, e in enumerate(COMMON_GRID):
            fams = [f"{fam}_{k}" for fam in ("blur", "striping", "clipped", "jpeg", "stain")]
            harm = pd.Series({meta.loc[n, "family"]: d["syn_point"][n] for n in fams})
            for pname in ("tier2", "tier1b", "tier1a", "one_minus_ssim", "delta_e"):
                sig = pd.Series({meta.loc[n, "family"]: pred[pname][n] for n in fams})
                sl.append({"fold": f, "condition": c, "delta_e": e, "predictor": pname, **ordering_concordance(harm, sig)})
    pd.DataFrame(real_rows).to_csv(out / "real_arm.csv", index=False)
    sdg = pd.DataFrame(syn_dg); sdg.to_csv(out / "synthetic_dg.csv", index=False)
    ssig = pd.DataFrame(syn_sig); ssig.to_csv(out / "synthetic_signals.csv", index=False)
    pd.DataFrame(h12).to_csv(out / "h1_h2.csv", index=False)
    pd.DataFrame(h3).to_csv(out / "h3.csv", index=False)
    pd.DataFrame(sl).to_csv(out / "matched_slice.csv", index=False)

    # ---- §9.6 alarm proportions (primary AND: Tier 1b AND Tier 2 majority of seeds; secondary: 1a AND 2)
    ssig = ssig.merge(meta[["family", "delta_e"]].rename(columns={"delta_e": "target_delta_e"}),
                      left_on="dose", right_index=True)
    ssig["tier2_flag"] = ssig.tier2_flag_seeds >= 2
    ssig["and_primary"] = ssig.tier1b_flag & ssig.tier2_flag
    ssig["and_secondary"] = ssig.tier1a_flag & ssig.tier2_flag
    ssig.groupby(["family", "target_delta_e"])[["tier1a_flag", "tier1b_flag", "tier2_flag", "and_primary", "and_secondary"]] \
        .mean().reset_index().to_csv(out / "alarms.csv", index=False)

    # ---- §9.7 realised false-alarm rate on null splits of each fold's training tiles
    rng = np.random.default_rng(0)
    nulls = []
    for f in folds:
        tr = tables[f][tables[f]["role"] == "train"]
        all_tiles, a1a, a1b = g["t1_all"]["densenet121"]
        rows = rows_for(all_tiles, tr["tile_id"])
        sub = per_slide_subsample(tr["slide_id"].to_numpy())
        X1a, X1b, sl_ids = a1a[rows][sub], a1b[rows][sub], tr["slide_id"].to_numpy()[sub]
        rd = run_dir("E1", f, 42)
        ptr = pd.read_csv(rd / "predictions_train.csv")
        F2 = np.load(rd / "features_train.npy").astype(np.float64)
        s2 = per_slide_subsample(ptr["slide_id"].to_numpy())
        X2, sl2 = F2[s2], ptr["slide_id"].to_numpy()[s2]
        fl = {"tier1a": 0, "tier1b": 0, "tier2": 0}
        for i in range(args.null_splits):
            for name, X, ids, ncomp in (("tier1a", X1a, sl_ids, None), ("tier1b", X1b, sl_ids, N_COMPONENTS), ("tier2", X2, sl2, None)):
                u = np.unique(ids)
                half = set(rng.permutation(u)[: len(u) // 2])
                m = np.isin(ids, list(half))
                if name == "tier2":
                    fl[name] += tier2(X[m], X[~m], ids[m], ids[~m], seed=i)["flagged"]
                else:
                    fl[name] += tier1(X[m], X[~m], ids[m], ids[~m], ncomp, n_permutations=99, seed=i)["flagged"]
        nulls += [{"fold": f, "tier": k, "realised_false_alarm": v / args.null_splits, "nominal_alpha": ALPHA,
                   "splits": args.null_splits} for k, v in fl.items()]
        print(f"null check fold {f}: {fl}", flush=True)
    pd.DataFrame(nulls).to_csv(out / "null_check.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({"analysis_plan_commit": plan, "boot": args.boot, "perm": args.perm,
                                                     "folds": folds}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
