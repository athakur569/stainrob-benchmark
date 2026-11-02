#!/usr/bin/env python3
"""Dose-range pilot (DR-020): does each synthetic-shift dose do anything at all?

Inside the TRAINING split only: patients are split 80/20 (stratified by whether they have tumour
tiles, seed 0); an E1 model is trained on the 80% and scored on the unseen 20%, unshifted and
under every family x dose of the dose table. No test tile is read. The pilot bounds the dose range;
it does not estimate effects for the paper.

``--rotate K`` (DR-020 amendment): instead of one 80/20 split, K-fold cross-fitting inside the
training split — every training-split patient is held out exactly once, so all tumour patients
contribute (one 80/20 split held out only 2 of the 12). Each tile is scored by the model that did
not see its patient; the pooled predictions are then compared across doses.

    python scripts/pilot_doses.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --out /media/abhi/camelyon17/derived/pilot_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stain_benchmark.evaluation.bootstrap import auroc_interval, joint_delta_g
from stain_benchmark.evaluation.metrics import compute_calibration_slope
from stain_benchmark.training.run import build_model, score_tiles, train_run

SEED_SPLIT, SEED_TRAIN, PILOT_FRACTION = 0, 42, 0.2


def pilot_table(tiles_dir: Path) -> pd.DataFrame:
    man = pd.read_csv(tiles_dir / "manifest.csv")
    man = man[man["qc_pass"]]
    ps = pd.read_csv(tiles_dir / "splits" / "patient_splits.csv")
    train_pats = ps[ps["split"] == "train"]
    pool = man[man["patient_id"].isin(set(train_pats["patient_id"]))].copy()
    tumour = set(pool.loc[pool["label"] == 1, "patient_id"])
    rng = np.random.default_rng(SEED_SPLIT)
    pilot = set()
    for stratum in (sorted(tumour), sorted(set(pool["patient_id"]) - tumour)):
        k = max(1, int(round(PILOT_FRACTION * len(stratum))))
        pilot |= set(rng.choice(stratum, k, replace=False))
    pool["role"] = np.where(pool["patient_id"].isin(pilot), "pilot", "train")
    return pool.reset_index(drop=True)


def rotation_tables(tiles_dir: Path, k: int) -> list[pd.DataFrame]:
    """K tables over training-split tiles; in table f, fold f's patients have role "pilot"."""
    base = pilot_table(tiles_dir).drop(columns="role")
    tumour = set(base.loc[base["label"] == 1, "patient_id"])
    rng = np.random.default_rng(SEED_SPLIT)
    fold_of = {}
    for stratum in (sorted(tumour), sorted(set(base["patient_id"]) - tumour)):
        for i, pid in enumerate(rng.permutation(stratum)):
            fold_of[pid] = i % k
    f = base["patient_id"].map(fold_of)
    return [base.assign(role=np.where(f == j, "pilot", "train")) for j in range(k)]


def score_all(model, tiles_dir, pilot, dev, doses, workers):
    out = {"unshifted": score_tiles(model, tiles_dir, pilot, dev, num_workers=workers)}
    for fam, levels in doses["doses"].items():
        for k, d in enumerate(levels):
            out[f"{fam}|{k}"] = score_tiles(model, tiles_dir, pilot, dev,
                                            shift=(fam, d["strength"], doses["shift_seed"]), num_workers=workers)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--doses", default="configs/shift_doses_camelyon17.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rotate", type=int, default=0, help="K-fold cross-fitted pilot (0 = single 80/20 split)")
    args = ap.parse_args()
    tiles_dir, out = Path(args.tiles), Path(args.out)
    doses = json.loads(Path(args.doses).read_text())
    meta = {f"{fam}|{k}": (fam, d["target_delta_e"], d["strength"], d["achieved_ssim"])
            for fam, levels in doses["doses"].items() for k, d in enumerate(levels)}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.rotate:
        pooled: dict[str, list] = {}
        for j, table in enumerate(rotation_tables(tiles_dir, args.rotate)):
            pilot = table[table["role"] == "pilot"].reset_index(drop=True)
            print(f"rotation {j}: held out {pilot.patient_id.nunique()} patients "
                  f"({pilot.loc[pilot.label == 1, 'patient_id'].nunique()} with tumour), {len(pilot)} tiles", flush=True)
            run_dir = out / f"E1_rot{args.rotate}_fold{j}"
            if not (run_dir / "summary.json").exists():
                train_run(tiles_dir, f"pilot{j}", SEED_TRAIN, run_dir, condition="E1", roles=("pilot",),
                          table=table, num_workers=args.workers)
            model = build_model(pretrained=False).to(dev)
            model.load_state_dict(torch.load(run_dir / "model.pt", map_location=dev))
            for name, pred in score_all(model, tiles_dir, pilot, dev, doses, args.workers).items():
                pooled.setdefault(name, []).append(pred)
        preds = {n: pd.concat(v).sort_values("tile_id").reset_index(drop=True) for n, v in pooled.items()}
        ref = preds.pop("unshifted")
        base = auroc_interval(ref, n_boot=1000, seed=0)
        print(f"cross-fitted AUROC unshifted: {base.estimate:.3f} [{base.low:.3f}, {base.high:.3f}] on "
              f"{ref.patient_id.nunique()} patients ({ref.loc[ref.label == 1, 'patient_id'].nunique()} with tumour)")
        point, boots = joint_delta_g(ref, preds, n_boot=1000, seed=0)
        lo, hi = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
        # Calibration-slope ΔG (DR-006): can the model keep its ranking while its probabilities degrade?
        s_point, s_boots = joint_delta_g(ref, preds, n_boot=300, seed=0, metric="slope")
        s_lo, s_hi = np.quantile(s_boots, 0.025, axis=0), np.quantile(s_boots, 0.975, axis=0)
        ref_slope = compute_calibration_slope(ref["label"], ref["p_tumour"])[0]
        print(f"unshifted calibration slope: {ref_slope:.3f}")
        res = pd.DataFrame([{"family": meta[c][0], "delta_e": meta[c][1], "strength": meta[c][2], "ssim": meta[c][3],
                             "auroc": base.estimate - point[c], "auroc_drop": point[c],
                             "drop_low": lo[i], "drop_high": hi[i],
                             "slope": ref_slope - s_point[c], "slope_drop": s_point[c],
                             "slope_drop_low": s_lo[i], "slope_drop_high": s_hi[i]}
                            for i, c in enumerate(point.index)])
        res.to_csv(out / f"pilot_dose_response_rotated{args.rotate}.csv", index=False)
        print("\nAUROC drop, cross-fitted over all training-split patients (patient-clustered 95% interval):")
        print(res.round(3).to_string(index=False))
        return

    table = pilot_table(tiles_dir)
    pilot = table[table["role"] == "pilot"].reset_index(drop=True)
    print(f"pilot split: train {int((table.role == 'train').sum())} tiles / "
          f"{table.loc[table.role == 'train', 'patient_id'].nunique()} patients; pilot {len(pilot)} tiles / "
          f"{pilot.patient_id.nunique()} patients ({pilot.loc[pilot.label == 1, 'patient_id'].nunique()} with tumour)")
    run_dir = out / "E1_pilot"
    if (run_dir / "summary.json").exists():
        print(f"reusing the pilot model in {run_dir}")
    else:
        summary = train_run(tiles_dir, "pilot", SEED_TRAIN, run_dir, condition="E1", roles=("pilot",),
                            table=table, num_workers=args.workers)
        print(f"trained in {summary['train_seconds'] / 60:.1f} min")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(dev)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=dev))
    ref = score_tiles(model, tiles_dir, pilot, dev, num_workers=args.workers)
    base = auroc_interval(ref, n_boot=1000, seed=0)
    print(f"pilot AUROC unshifted: {base.estimate:.3f} [{base.low:.3f}, {base.high:.3f}]")

    shifted, meta = {}, {}
    for fam, levels in doses["doses"].items():
        for k, d in enumerate(levels):
            name = f"{fam}|{k}"
            shifted[name] = score_tiles(model, tiles_dir, pilot, dev, shift=(fam, d["strength"], doses["shift_seed"]),
                                        num_workers=args.workers)
            meta[name] = (fam, d["target_delta_e"], d["strength"], d["achieved_ssim"])
            print(f"scored {name}", flush=True)
    point, boots = joint_delta_g(ref, shifted, n_boot=1000, seed=0)
    lo, hi = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
    rows = [{"family": meta[c][0], "delta_e": meta[c][1], "strength": meta[c][2], "ssim": meta[c][3],
             "auroc": base.estimate - point[c], "auroc_drop": point[c], "drop_low": lo[i], "drop_high": hi[i]}
            for i, c in enumerate(point.index)]
    res = pd.DataFrame(rows)
    res.to_csv(out / f"pilot_dose_response_{Path(args.doses).stem}.csv", index=False)
    print("\nAUROC drop on unseen training-split patients (patient-clustered 95% interval):")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
