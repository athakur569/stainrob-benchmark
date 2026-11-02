"""Tests for the analysis building blocks."""

import numpy as np
import pandas as pd

from stain_benchmark.evaluation.analysis import (
    ordering_concordance,
    per_slide_subsample,
    prepare,
    tier1,
    tier2,
)


def test_per_slide_subsample_caps_and_is_seeded():
    slides = np.repeat(["a", "b", "c"], [3, 20, 9])
    pos = per_slide_subsample(slides, k=8)
    counts = pd.Series(slides[pos]).value_counts()
    assert counts["a"] == 3 and counts["b"] == 8 and counts["c"] == 8
    assert np.array_equal(pos, per_slide_subsample(slides, k=8)) and (np.diff(pos) > 0).all()


def test_prepare_uses_reference_statistics_only():
    r = np.random.default_rng(0)
    ref, dep = r.normal(5, 2, (200, 10)), r.normal(5, 2, (50, 10)) + 3
    a, b = prepare(ref, dep, None)
    assert np.allclose(a.mean(0), 0, atol=1e-9) and np.allclose(a.std(0), 1, atol=1e-6)
    assert b.mean() > 1                                   # the shift survives standardisation
    a2, b2 = prepare(ref, dep, 4)
    assert a2.shape == (200, 4) and b2.shape == (50, 4)


def _clustered(n_slides, per, dim, seed, shift=0.0):
    r = np.random.default_rng(seed)
    X = np.concatenate([r.normal(0, 1.0, dim) + r.normal(0, 0.3, (per, dim)) for _ in range(n_slides)]) + shift
    return X, np.repeat([f"s{seed}_{i}" for i in range(n_slides)], per)


def test_tier1_and_tier2_flag_a_shift_and_not_the_null():
    ref, rs = _clustered(30, 8, 6, seed=1)
    null, ns = _clustered(20, 8, 6, seed=2)
    shifted, ss = _clustered(20, 8, 6, seed=3, shift=3.0)
    assert not tier1(ref, null, rs, ns, None, n_permutations=99)["flagged"]
    t = tier1(ref, shifted, rs, ss, None, n_permutations=99)
    assert t["flagged"] and t["mmd2"] > tier1(ref, null, rs, ns, None, n_permutations=99)["mmd2"]
    assert tier2(ref, shifted, rs, ss)["flagged"]
    assert tier2(ref, shifted, rs, ss)["mahalanobis"] > tier2(ref, null, rs, ns)["mahalanobis"]


def test_ordering_concordance():
    harm = pd.Series({"blur": 0.28, "jpeg": 0.14, "stain": 0.04, "striping": 0.036, "clipped": 0.035})
    perfect = harm * 10
    flat = pd.Series(1.0, index=harm.index)
    assert ordering_concordance(harm, perfect)["concordant_pairs"] == 10
    assert ordering_concordance(harm, flat)["concordant_pairs"] == 0      # ΔE at a matched dose: all tied
    rev = ordering_concordance(harm, -harm)
    assert rev["concordant_pairs"] == 0 and rev["predictor_order"].startswith("clipped")


def test_prepare_pooled_is_symmetric_in_the_two_batches():
    r = np.random.default_rng(0)
    a, b = r.normal(size=(60, 30)), r.normal(size=(40, 30)) + 1
    ra, rb = prepare(a, b, 5, fit_on="pooled")
    pooled = np.vstack([ra, rb])
    assert np.allclose(pooled.mean(0), 0, atol=1e-9)                     # centred on both batches together
    sb, sa = prepare(b, a, 5, fit_on="pooled")
    assert np.allclose(np.abs(np.vstack([ra, rb]).T @ np.vstack([ra, rb])),
                       np.abs(np.vstack([sa, sb]).T @ np.vstack([sa, sb])))   # same subspace whichever is "reference"


def test_reference_fitted_projection_false_alarms_in_high_dimension():
    """ISS-049 in miniature: two halves of one population, few samples, many features."""
    r = np.random.default_rng(1)
    X = r.normal(size=(120, 400))
    ids = np.repeat(np.arange(30), 4)
    ext = r.normal(size=(120, 400))
    alarms = {"reference": 0, "pooled": 0, "external": 0}
    for i in range(10):
        m = np.isin(ids, np.random.default_rng(i).permutation(30)[:15])
        for fit_on in alarms:
            alarms[fit_on] += tier1(X[m], X[~m], ids[m], ids[~m], 10, n_permutations=49, seed=i, fit_on=fit_on,
                                    external=ext if fit_on == "external" else None)["flagged"]
    assert alarms["reference"] >= 5 and alarms["pooled"] <= 2 and alarms["external"] <= 2


def test_prepare_external_needs_external_data():
    import pytest
    x = np.zeros((5, 3))
    with pytest.raises(ValueError):
        prepare(x, x, None, fit_on="external")
    with pytest.raises(ValueError):
        prepare(x, x, None, fit_on="pooled", external=x)
