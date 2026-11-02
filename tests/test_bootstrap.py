"""Tests for the patient-clustered bootstrap."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from stain_benchmark.evaluation.bootstrap import (
    auroc_interval,
    compare_predictors,
    delta_g_interval,
    joint_delta_g,
    patient_weights,
    weighted_auroc,
)


def make_pred(n_pat=30, tiles=20, sep=1.0, patient_sd=1.0, seed=0, prefix="p"):
    """Tiles clustered in patients: a shared patient effect makes tiles of one patient alike."""
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        tumour = i % 2 == 0
        eff = r.normal(0, patient_sd)
        for t in range(tiles):
            label = int(tumour and t % 2 == 0)       # tumour patients have tumour and normal tiles
            score = sep * label + eff + r.normal(0, 0.5)
            rows.append(dict(tile_id=f"{prefix}{i}_{t}", patient_id=f"{prefix}{i}", label=label,
                             p_tumour=1 / (1 + np.exp(-score))))
    return pd.DataFrame(rows)


def test_weights_are_whole_patients_and_strata_keep_size():
    pred = make_pred(n_pat=10, tiles=3)
    W = patient_weights(pred.patient_id.to_numpy(), {"p0", "p2", "p4", "p6", "p8"}, n_boot=50, seed=1)
    per_patient = pd.DataFrame(W.T).groupby(pred.patient_id.to_numpy()).nunique()
    assert (per_patient == 1).all().all()                       # all tiles of a patient share a weight
    tum = pred.patient_id.isin({"p0", "p2", "p4", "p6", "p8"}).to_numpy()
    assert np.allclose(W[:, tum].sum(1), 5 * 3) and np.allclose(W[:, ~tum].sum(1), 5 * 3)


def test_weighted_auroc_equals_duplicated_tiles():
    y = np.array([0, 1, 0, 1, 1]); p = np.array([.2, .6, .7, .9, .4]); w = np.array([2, 1, 0, 3, 1])
    assert weighted_auroc(y, p, w) == pytest.approx(roc_auc_score(np.repeat(y, w), np.repeat(p, w)))


def test_interval_contains_point_and_is_seeded():
    pred = make_pred()
    a, b = auroc_interval(pred, n_boot=300, seed=4), auroc_interval(pred, n_boot=300, seed=4)
    assert a.low < a.estimate < a.high and (a.low, a.high) == (b.low, b.high)


def test_clustered_interval_is_wider_than_tile_bootstrap():
    # With a strong patient effect, resampling tiles as if independent understates uncertainty.
    pred = make_pred(n_pat=16, tiles=40, patient_sd=1.5, seed=2)
    clustered = auroc_interval(pred, n_boot=400, seed=0)
    r = np.random.default_rng(0); y, p = pred.label.to_numpy(), pred.p_tumour.to_numpy()
    naive = [roc_auc_score(y[i], p[i]) for i in (r.integers(0, len(y), len(y)) for _ in range(400))]
    assert (clustered.high - clustered.low) > 1.5 * (np.quantile(naive, .975) - np.quantile(naive, .025))


def test_delta_g_interval_signs():
    good = make_pred(sep=3.0, seed=1, prefix="a"); bad = make_pred(sep=0.3, seed=2, prefix="b")
    dg = delta_g_interval(good, bad, n_boot=300, seed=0)
    assert dg.estimate > 0 and dg.low > 0


def test_joint_delta_g_pairs_conditions():
    ref = make_pred(sep=2.0)
    same = ref.copy()
    worse = ref.assign(p_tumour=0.5 * ref.p_tumour + 0.25)               # monotone: same ranking
    noisy = ref.assign(p_tumour=np.random.default_rng(0).random(len(ref)))
    point, boots = joint_delta_g(ref, {"same": same, "monotone": worse, "noise": noisy}, n_boot=200)
    assert point["same"] == 0 and np.allclose(boots[:, 0], 0)            # pairing: identical → exactly 0
    assert point["monotone"] == pytest.approx(0)
    assert point["noise"] > 0.2 and boots[:, 2].min() > 0
    with pytest.raises(ValueError, match="same order"):
        joint_delta_g(ref, {"x": ref.iloc[::-1]}, n_boot=5)


def test_compare_predictors_ranks_the_good_predictor_first():
    conds = [f"c{i}" for i in range(8)]
    harm = pd.Series(np.linspace(0, 0.3, 8), index=conds)
    boots = harm.to_numpy() + np.random.default_rng(0).normal(0, 0.01, (300, 8))
    preds = {"monitor": harm * 10 + 1, "magnitude": pd.Series(np.r_[np.linspace(1, 4, 4), np.linspace(1, 4, 4)], index=conds)}
    out = compare_predictors(harm, boots, preds).set_index("predictor")
    assert out.loc["monitor", "rho"] == pytest.approx(1.0)
    assert out.loc["magnitude", "rho"] < out.loc["monitor", "rho"]
    assert out.loc["magnitude", "diff_high"] < 0                          # reliably worse than the monitor


def test_joint_slope_delta_g_detects_flattened_probabilities():
    ref = make_pred(sep=2.0)
    flat = ref.assign(p_tumour=0.5 + 0.1 * (ref.p_tumour - 0.5))          # same ranking, squashed probabilities
    point, boots = joint_delta_g(ref, {"flat": flat}, n_boot=100, metric="slope")
    a_point, _ = joint_delta_g(ref, {"flat": flat}, n_boot=10, metric="auroc")
    assert a_point["flat"] == pytest.approx(0)                       # AUROC cannot see it
    assert point["flat"] < 0 and boots[:, 0].max() < 0               # slope rises (underconfident) -> negative drop


def test_seed_averaged_joint_delta_g():
    from stain_benchmark.evaluation.bootstrap import delta_g_seeds_interval, joint_delta_g_seeds
    refs = [make_pred(sep=2.0, seed=s) for s in (0, 1)]
    refs = [r.assign(tile_id=refs[0].tile_id, patient_id=refs[0].patient_id, label=refs[0].label) for r in refs]
    shifted = [{"same": r.copy(), "noise": r.assign(p_tumour=np.random.default_rng(9).random(len(r)))} for r in refs]
    point, boots = joint_delta_g_seeds(refs, shifted, n_boot=100)
    assert point["same"] == 0 and np.allclose(boots[:, 0], 0) and point["noise"] > 0.2
    single = [joint_delta_g(r, s, n_boot=5)[0]["noise"] for r, s in zip(refs, shifted)]
    assert point["noise"] == pytest.approx(np.mean(single))
    iv = delta_g_seeds_interval(refs, [r.assign(p_tumour=0.5) for r in refs], n_boot=100)
    assert iv.low <= iv.estimate <= iv.high and iv.estimate > 0
