"""
Building blocks for the pre-registered analysis (docs/analysis_plan.md, docs/deviations.md D1–D4).

Monitor computations follow D2: features standardised with the reference's mean and SD; Tier 1b and
Tier 2 projected onto the reference's first 50 principal components; at most 8 tiles per slide (seed 0)
for reference and deployment; slide ids as clusters. Primary continuous signals: MMD² (Tier 1a, 1b),
mean Mahalanobis distance (Tier 2). Alarms at α = 0.05 per tier; primary AND rule Tier 1b AND Tier 2 (D4).

The monitors come from the wsidrift package (main branch, slide-level calibration). Set the
environment variable WSIDRIFT_SRC to its ``src`` directory if it is not installed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_WSIDRIFT = os.environ.get("WSIDRIFT_SRC", os.path.expanduser("~/Desktop/code/wsidrift/src"))
if _WSIDRIFT not in sys.path:
    sys.path.insert(0, _WSIDRIFT)
from wsidrift.monitor.divergence import Tier1Monitor  # noqa: E402
from wsidrift.monitor.embedding import Tier2Monitor  # noqa: E402

TILES_PER_SLIDE = 8
N_COMPONENTS = 50
ALPHA = 0.05
SUBSAMPLE_SEED = 0


def per_slide_subsample(slide_ids: np.ndarray, k: int = TILES_PER_SLIDE, seed: int = SUBSAMPLE_SEED) -> np.ndarray:
    """Positions of at most ``k`` tiles per slide, chosen at random (seeded), in original order."""
    slide_ids = np.asarray(slide_ids).astype(str)
    rng = np.random.default_rng(seed)
    keep = []
    for sid in np.unique(slide_ids):
        pos = np.flatnonzero(slide_ids == sid)
        keep.append(pos if len(pos) <= k else rng.choice(pos, k, replace=False))
    return np.sort(np.concatenate(keep))


FIT_ON = ("reference", "pooled", "external")


def prepare(ref: np.ndarray, dep: np.ndarray, n_components: int | None,
            fit_on: str = "reference", external: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Standardise both and optionally project onto the top principal components.

    ``fit_on="reference"`` (pre-registered, D2) fits mean, SD and components on the reference only.
    ``fit_on="pooled"`` (post hoc, ISS-049) fits them on reference and deployment together — label-free,
    and symmetric in the two batches, which a permutation test needs: components fitted on one batch
    capture that batch's own variation, and in high dimension the test then separates any two batches.
    ``fit_on="external"`` (post hoc) fits them on a third set (``external``) disjoint from both batches, e.g.
    the fold's training tiles; it does not depend on the split at all.
    """
    if fit_on not in FIT_ON:
        raise ValueError(f"fit_on must be one of {FIT_ON}")
    ref = np.asarray(ref, dtype=np.float64)
    dep = np.asarray(dep, dtype=np.float64)
    if (fit_on == "external") != (external is not None):
        raise ValueError("pass external data if, and only if, fit_on='external'")
    fit = {"reference": ref, "pooled": np.vstack([ref, dep]) if fit_on == "pooled" else None,
           "external": None if external is None else np.asarray(external, dtype=np.float64)}[fit_on]
    mu, sd = fit.mean(0), fit.std(0) + 1e-8
    r, d = (ref - mu) / sd, (dep - mu) / sd
    if n_components is not None and r.shape[1] > n_components:
        v = np.linalg.svd((fit - mu) / sd, full_matrices=False)[2][:n_components].T
        r, d = r @ v, d @ v
    return r, d


def tier1(ref: np.ndarray, dep: np.ndarray, ref_slides: np.ndarray, dep_slides: np.ndarray,
          n_components: int | None, n_permutations: int = 200, seed: int = 0, fit_on: str = "reference",
          external: np.ndarray | None = None) -> dict:
    """Tier 1 on one deployment batch: MMD² (primary), mean Wasserstein, p-values and the alarm."""
    r, d = prepare(ref, dep, n_components, fit_on, external)
    res = Tier1Monitor(alpha=ALPHA, n_permutations=n_permutations, seed=seed).fit(
        r, groups=np.asarray(ref_slides)).score(d, groups=np.asarray(dep_slides))
    return {"mmd2": res.mmd_squared, "wasserstein": res.wasserstein_mean, "p_mmd": res.p_mmd,
            "p_wass": res.p_wass, "flagged": bool(res.flagged)}


def tier2(ref: np.ndarray, dep: np.ndarray, ref_slides: np.ndarray, dep_slides: np.ndarray,
          ref_logits: np.ndarray | None = None, dep_logits: np.ndarray | None = None, seed: int = 0) -> dict:
    """Tier 2 on one deployment batch: mean Mahalanobis (primary), mean energy and the alarm."""
    r, d = prepare(ref, dep, N_COMPONENTS)
    mon = Tier2Monitor(alpha=ALPHA, seed=seed).fit(r, logits=ref_logits, groups=np.asarray(ref_slides))
    res = mon.score(d, logits=dep_logits, groups=np.asarray(dep_slides))
    return {"mahalanobis": res.mahalanobis_mean, "energy": res.energy_mean, "flagged": bool(res.flagged)}


def ordering_concordance(harm: pd.Series, signal: pd.Series) -> dict:
    """D1 matched-dose slice: how a predictor orders the families against their measured harm.

    Returns the predictor's ordering (most to least), the harm ordering, and the number of family
    pairs the predictor orders the same way as harm (ties in the predictor count as not concordant).
    """
    fams = list(harm.index)
    pairs = [(a, b) for i, a in enumerate(fams) for b in fams[i + 1:]]
    conc = sum(1 for a, b in pairs if np.sign(signal[a] - signal[b]) == np.sign(harm[a] - harm[b]) != 0)
    return {"predictor_order": " > ".join(signal.sort_values(ascending=False).index),
            "harm_order": " > ".join(harm.sort_values(ascending=False).index),
            "concordant_pairs": conc, "pairs": len(pairs)}


def spearman(x, y) -> float:
    return float(spearmanr(x, y).statistic)
