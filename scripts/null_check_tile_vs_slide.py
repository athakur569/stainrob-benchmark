#!/usr/bin/env python3
"""ISS-039: Tier 1 on real CAMELYON17 tiles with no shift, tile-level against slide-level calibration.

The training-split slides are split into two random halves 40 times; no shift exists between the halves, so every
alarm is a false alarm. Up to 8 tiles per slide (seed 0); Tier 1a on the 23 pixel statistics, standardised; Tier 1b on
DenseNet-121 features, standardised and projected on their first 50 principal components (for speed); nominal
false-alarm rate 0.10, 99 permutations. Each split is scored twice: tiles treated as independent, and with slide
groups (whole slides permuted). Reported in wsidrift's README and CHANGELOG (82% and 100% tile-level, 5% and 2% with
slide groups). Training-split tiles only: no held-out tile, model or label is read.

Inputs come from the frozen tile set (`tiles_v1`, its patient splits) via scripts/extract_tier1.py:
    python scripts/extract_tier1.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --out /media/abhi/camelyon17/derived/tier1_v1/train_split/unshifted --subset train-split
    python scripts/extract_tier1.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --out /media/abhi/camelyon17/derived/tier1_v1/train_split/unshifted_densenet121 --subset train-split \\
        --encoder densenet121

    python scripts/null_check_tile_vs_slide.py --tier1 /media/abhi/camelyon17/derived/tier1_v1/train_split
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stain_benchmark.evaluation.analysis import wsidrift_source  # also puts wsidrift on the path
from wsidrift.monitor.divergence import Tier1Monitor

N_SPLITS, ALPHA, N_PERM, PER_SLIDE, N_PC = 40, 0.10, 99, 8, 50


def standardise(X: np.ndarray) -> np.ndarray:
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:N_PC].T if X.shape[1] > N_PC else Xc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier1", required=True, help="the train_split directory written by scripts/extract_tier1.py")
    args = ap.parse_args()
    base = Path(args.tier1)
    tiles = pd.read_csv(base / "unshifted" / "tiles.csv")
    feats = {"1a (23 pixel statistics)": pd.read_csv(base / "unshifted" / "tier1a.csv").drop(columns="tile_id").to_numpy(),
             "1b (DenseNet-121)": np.load(base / "unshifted_densenet121" / "tier1b.npy").astype(np.float64)}
    keep = tiles.groupby("slide_id", group_keys=False).apply(
        lambda g: g.sample(min(PER_SLIDE, len(g)), random_state=0)).index
    tiles = tiles.loc[keep].reset_index(drop=True)
    feats = {k: v[keep] for k, v in feats.items()}
    slides = tiles.slide_id.unique()
    g = tiles.slide_id.to_numpy()
    print(wsidrift_source(), flush=True)
    print(f"{len(tiles)} tiles (≤{PER_SLIDE} per slide) from {len(slides)} training-split slides, "
          f"{tiles.patient_id.nunique()} patients", flush=True)
    rng = np.random.default_rng(0)
    for name, X in feats.items():
        X = standardise(X)
        tile_alarms = slide_alarms = 0
        for s in range(N_SPLITS):
            a = tiles.slide_id.isin(set(rng.permutation(slides)[: len(slides) // 2])).to_numpy()
            tile_alarms += Tier1Monitor(alpha=ALPHA, n_permutations=N_PERM, seed=s).fit(X[a]).score(X[~a]).flagged
            slide_alarms += Tier1Monitor(alpha=ALPHA, n_permutations=N_PERM, seed=s).fit(
                X[a], groups=g[a]).score(X[~a], groups=g[~a]).flagged
        print(f"Tier {name}: false alarms with no shift (nominal {ALPHA:.0%}): tile-level "
              f"{tile_alarms / N_SPLITS:.0%}, slide-level {slide_alarms / N_SPLITS:.0%} ({N_SPLITS} random slide splits)",
              flush=True)


if __name__ == "__main__":
    main()
