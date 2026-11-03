#!/usr/bin/env python3
"""POST HOC (P5b): does a stain normaliser itself amplify an artefact? No model, no labels, training tiles only.

For each fold's E2 (Macenko) and E3 (Reinhard) normaliser (DR-013 reference), every family and dose of the
pre-registered dose table, and every tile x of the training-split calibration sample (the sample that fixed the
dose table) with shifted version x', three distances, in pixels (RMS over RGB) and in the frozen DenseNet-121
feature space of Tier 1b (Euclidean):

    d_raw    = d(x, x')                        the artefact, no normaliser
    d_oracle = d(N_x(x), N_x(x'))              statistics estimated on the clean tile (oracle, counterfactual)
    d_norm   = d(N_x(x), N_x'(x'))             statistics estimated on each tile as it arrives (as deployed)

d_oracle / d_raw = transform amplification with a correct estimate; d_norm / d_oracle = estimation corruption.
Specification: docs/posthoc_analyses.md, P5b.

    python scripts/normaliser_amplification.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --out /media/abhi/camelyon17/derived/posthoc/normaliser_amplification
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_shifts import calibration_sample  # noqa: E402

from stain_benchmark.data.dataset import fold_table  # noqa: E402
from stain_benchmark.data.shifts import apply_shift  # noqa: E402
from stain_benchmark.monitor.features import Tier1bEncoder  # noqa: E402
from stain_benchmark.training.run import fit_reference  # noqa: E402

FOLDS = ("0", "1", "2", "3", "4")
CONDITIONS = ("E2", "E3")


def rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)))


def job(args) -> pd.DataFrame:
    fold, cond, tiles_dir, doses_path = args
    torch.set_num_threads(1)
    doses = json.loads(Path(doses_path).read_text())
    sample = calibration_sample(tiles_dir)
    imgs = [np.array(Image.open(tiles_dir / p).convert("RGB")) for p in sample["path"]]
    keys = sample["tile_id"].astype(str).tolist()
    norm, _ = fit_reference(cond, tiles_dir, fold_table(tiles_dir, fold))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Tier1bEncoder(pretrained=True, arch="densenet121").to(dev)

    def feats(batch: list[np.ndarray]) -> np.ndarray:
        return enc(torch.from_numpy(np.stack(batch))).cpu().numpy().astype(np.float64)

    clean_n = [norm.transform(x) for x in imgs]
    f_clean, f_clean_n = feats(imgs), feats(clean_n)
    rows = []
    for fam, levels in doses["doses"].items():
        for k, d in enumerate(levels):
            shifted = [apply_shift(x, fam, d["strength"], seed=doses["shift_seed"], key=key) for x, key in zip(imgs, keys)]
            oracle = [norm.transform(xs, source=x) for xs, x in zip(shifted, imgs)]
            deployed = [norm.transform(xs) for xs in shifted]
            f_s, f_o, f_d = feats(shifted), feats(oracle), feats(deployed)
            for i, key in enumerate(keys):
                rows.append({"fold": fold, "condition": cond, "family": fam, "dose": f"{fam}_{k}",
                             "delta_e": d["target_delta_e"], "tile_id": key,
                             "px_raw": rms(imgs[i], shifted[i]), "px_oracle": rms(clean_n[i], oracle[i]),
                             "px_norm": rms(clean_n[i], deployed[i]),
                             "feat_raw": float(np.linalg.norm(f_clean[i] - f_s[i])),
                             "feat_oracle": float(np.linalg.norm(f_clean_n[i] - f_o[i])),
                             "feat_norm": float(np.linalg.norm(f_clean_n[i] - f_d[i]))})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-9
    for sp in ("px", "feat"):
        df[f"{sp}_transform_amp"] = df[f"{sp}_oracle"] / (df[f"{sp}_raw"] + eps)
        df[f"{sp}_estimation"] = df[f"{sp}_norm"] / (df[f"{sp}_oracle"] + eps)
    cols = ["px_transform_amp", "px_estimation", "feat_transform_amp", "feat_estimation"]
    per_fold = df.groupby(["condition", "family", "delta_e", "fold"])[cols].median()
    out = per_fold.groupby(["condition", "family", "delta_e"]).agg(["median", "min", "max"])
    out.columns = [f"{c}_{s}" for c, s in out.columns]
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--doses", default="configs/shift_doses_camelyon17.json")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(f, c, Path(args.tiles), args.doses) for f in FOLDS for c in CONDITIONS]
    parts = []
    ctx = torch.multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(args.jobs, mp_context=ctx) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for n, fu in enumerate(as_completed(futs), 1):
            part = fu.result()
            parts.append(part)
            print(f"[{n}/{len(jobs)}] fold {part.fold.iloc[0]} {part.condition.iloc[0]}: {len(part)} tile × dose rows", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(out / "per_tile.csv", index=False)
    summarise(df).to_csv(out / "summary.csv", index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
