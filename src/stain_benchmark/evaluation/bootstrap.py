"""
Patient-clustered bootstrap for AUROC, calibration slope and ΔG (ISS-031, ISS-039).

Tiles from one patient are not independent, so uncertainty is estimated by resampling PATIENTS
with replacement; every tile of a drawn patient comes along, implemented as per-tile weights
equal to how often its patient was drawn. Resampling is stratified by patient class (patients
with any tumour tile vs normal-only patients) so every resample contains both classes.

A *joint* bootstrap uses one set of patient draws for several conditions that share patients
(e.g. the same test tiles under 25 synthetic shifts), so the dependence between conditions is
kept and differences between them get honest intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from stain_benchmark.evaluation.metrics import compute_calibration_slope


def patient_weights(patient_ids: np.ndarray, tumour_patients: set, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_tiles) tile weights: how many times each tile's patient was drawn.

    Patients are drawn with replacement within two strata — patients with tumour tiles and
    normal-only patients — keeping each stratum's size.
    """
    pids = np.asarray(patient_ids).astype(str)
    uniq = np.unique(pids)
    pos = np.array([p for p in uniq if p in tumour_patients])
    neg = np.array([p for p in uniq if p not in tumour_patients])
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need patients with tumour tiles and normal-only patients")
    col = {p: i for i, p in enumerate(uniq)}
    tile_patient = np.array([col[p] for p in pids])
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, len(uniq)))
    for stratum in (pos, neg):
        idx = np.array([col[p] for p in stratum])
        draws = rng.integers(0, len(stratum), size=(n_boot, len(stratum)))
        for b in range(n_boot):
            counts[b] += np.bincount(idx[draws[b]], minlength=len(uniq))
    return counts[:, tile_patient]


def tumour_patient_set(pred: pd.DataFrame) -> set:
    return set(pred.loc[pred["label"] == 1, "patient_id"].astype(str))


def weighted_auroc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    m = w > 0
    return float(roc_auc_score(y[m], p[m], sample_weight=w[m]))


def weighted_slope(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    m = w > 0
    return compute_calibration_slope(y[m], p[m], sample_weight=w[m])[0]


@dataclass
class Interval:
    estimate: float
    low: float
    high: float
    n_boot: int

    def to_dict(self) -> dict:
        return {"estimate": self.estimate, "low": self.low, "high": self.high, "n_boot": self.n_boot}


def _interval(point: float, boots: np.ndarray, level: float) -> Interval:
    a = (1 - level) / 2
    boots = boots[np.isfinite(boots)]
    return Interval(point, float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)), len(boots))


def auroc_interval(pred: pd.DataFrame, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> Interval:
    """AUROC of ``pred`` (columns label, p_tumour, patient_id) with a patient-clustered interval."""
    y, p = pred["label"].to_numpy(), pred["p_tumour"].to_numpy()
    W = patient_weights(pred["patient_id"].to_numpy(), tumour_patient_set(pred), n_boot, seed)
    boots = np.array([weighted_auroc(y, p, w) for w in W])
    return _interval(float(roc_auc_score(y, p)), boots, level)


def delta_g_interval(pred_in: pd.DataFrame, pred_cross: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
                     level: float = 0.95, metric: str = "auroc") -> Interval:
    """ΔG = metric(test_in) − metric(test_cross), resampling the (disjoint) patients of each set."""
    fn = weighted_auroc if metric == "auroc" else weighted_slope
    sets = []
    for k, pred in enumerate((pred_in, pred_cross)):
        y, p = pred["label"].to_numpy(), pred["p_tumour"].to_numpy()
        W = patient_weights(pred["patient_id"].to_numpy(), tumour_patient_set(pred), n_boot, seed + k)
        sets.append((y, p, W))
    point = fn(sets[0][0], sets[0][1], np.ones(len(sets[0][0]))) - fn(sets[1][0], sets[1][1], np.ones(len(sets[1][0])))
    boots = np.array([fn(sets[0][0], sets[0][1], sets[0][2][b]) - fn(sets[1][0], sets[1][1], sets[1][2][b])
                      for b in range(n_boot)])
    return _interval(float(point), boots, level)


def joint_delta_g(reference: pd.DataFrame, shifted: dict[str, pd.DataFrame], n_boot: int = 2000,
                  seed: int = 0, metric: str = "auroc") -> tuple[pd.Series, np.ndarray]:
    """Paired ΔG for conditions that share the reference's tiles (synthetic shifts of test_in).

    ΔG(c) = metric(reference) − metric(shifted c), metric "auroc" or "slope" (calibration slope).
    One set of patient draws is used for the
    reference and every condition, so the pairing and the dependence between conditions are
    kept. Returns the point estimates (Series over conditions) and the (n_boot, n_conditions)
    bootstrap matrix.
    """
    names = list(shifted)
    for c in names:
        if not np.array_equal(shifted[c]["tile_id"].to_numpy(), reference["tile_id"].to_numpy()):
            raise ValueError(f"condition {c!r} does not have the reference's tiles in the same order")
    fn = weighted_auroc if metric == "auroc" else weighted_slope
    y, p0 = reference["label"].to_numpy(), reference["p_tumour"].to_numpy()
    W = patient_weights(reference["patient_id"].to_numpy(), tumour_patient_set(reference), n_boot, seed)
    ones = np.ones(len(y))
    point = pd.Series({c: fn(y, p0, ones) - fn(y, shifted[c]["p_tumour"].to_numpy(), ones) for c in names})
    boots = np.empty((n_boot, len(names)))
    for b, w in enumerate(W):
        base = fn(y, p0, w)
        boots[b] = [base - fn(y, shifted[c]["p_tumour"].to_numpy(), w) for c in names]
    return point, boots


def compare_predictors(delta_g_point: pd.Series, delta_g_boot: np.ndarray, predictors: dict[str, pd.Series],
                       level: float = 0.95) -> pd.DataFrame:
    """How well does each predictor rank the conditions by harm (Spearman ρ with ΔG)?

    For each predictor: ρ with the point ΔG, and an interval from ρ recomputed on every joint
    bootstrap resample of ΔG (the predictors are condition-level values held fixed). Also the
    difference in ρ between each predictor and the first one, with its interval from the same
    resamples — the comparison the analysis plan pre-specifies (monitor vs ΔE vs SSIM).
    """
    names = list(predictors)
    conds = list(delta_g_point.index)
    X = {n: predictors[n].reindex(conds).to_numpy(dtype=float) for n in names}
    a = (1 - level) / 2
    rho_b = {n: np.array([spearmanr(X[n], row).statistic for row in delta_g_boot]) for n in names}
    rows = []
    for n in names:
        rho = spearmanr(X[n], delta_g_point.to_numpy()).statistic
        d = rho_b[n] - rho_b[names[0]]
        rows.append({"predictor": n, "rho": float(rho),
                     "rho_low": float(np.nanquantile(rho_b[n], a)), "rho_high": float(np.nanquantile(rho_b[n], 1 - a)),
                     f"diff_vs_{names[0]}": float(rho - spearmanr(X[names[0]], delta_g_point.to_numpy()).statistic),
                     "diff_low": float(np.nanquantile(d, a)), "diff_high": float(np.nanquantile(d, 1 - a))})
    return pd.DataFrame(rows)
