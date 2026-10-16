"""Tests for calibration and per-institution metrics."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from stain_benchmark.evaluation.metrics import (
    compute_calibration,
    compute_institution_metrics,
)


class TestCalibration:
    def test_perfectly_calibrated_is_zero(self):
        # In each bin the observed tumour fraction equals the predicted probability.
        y_prob = np.repeat([0.1, 0.5, 0.9], 10)
        y_true = np.concatenate([[1] + [0] * 9, [1] * 5 + [0] * 5, [1] * 9 + [0]])
        assert compute_calibration(y_true, y_prob, n_bins=10).ece == pytest.approx(0.0)

    def test_overconfident_by_hand(self):
        # All predictions 0.9 but only half are tumour: ECE = |0.5 - 0.9| = 0.4.
        y_prob = np.full(10, 0.9)
        y_true = np.array([1, 0] * 5)
        assert compute_calibration(y_true, y_prob).ece == pytest.approx(0.4)

    def test_probability_one_lands_in_last_bin(self):
        result = compute_calibration(np.array([1, 1]), np.array([1.0, 1.0]), n_bins=5)
        assert result.bin_count[-1] == 2
        assert result.ece == pytest.approx(0.0)

    def test_bin_counts_sum_to_n(self):
        rng = np.random.default_rng(0)
        y_prob = rng.random(500)
        result = compute_calibration(rng.integers(0, 2, 500), y_prob)
        assert result.bin_count.sum() == 500

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_calibration(np.array([0, 1]), np.array([0.5]))


class TestInstitutionMetrics:
    def _data(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        p = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 300), 0, 1)
        inst = np.repeat(["A", "B", "C"], 100)
        p[inst == "C"] = rng.random(100)  # site C: model is guessing
        return y, p, inst

    def test_per_site_auc_matches_sklearn(self):
        y, p, inst = self._data()
        result = compute_institution_metrics(y, p, inst)
        for site in "ABC":
            m = inst == site
            assert result.per_institution[site] == pytest.approx(roc_auc_score(y[m], p[m]))

    def test_gap_is_best_minus_worst(self):
        y, p, inst = self._data()
        result = compute_institution_metrics(y, p, inst)
        aucs = result.per_institution.values()
        assert result.fairness_gap == pytest.approx(max(aucs) - min(aucs))
        assert result.fairness_gap > 0.2  # site C is near chance

    def test_small_and_single_class_sites_are_reported(self):
        y, p, inst = self._data()
        y = np.concatenate([y, [0] * 30, [1] * 5])
        p = np.concatenate([p, np.full(35, 0.5)])
        inst = np.concatenate([inst, ["D"] * 30, ["E"] * 5])
        result = compute_institution_metrics(y, p, inst)
        assert result.skipped["D"] == "only one class present"
        assert "5 samples" in result.skipped["E"]
        assert set(result.per_institution) == {"A", "B", "C"}

    def test_single_site_gap_is_nan(self):
        y, p, inst = self._data()
        m = inst == "A"
        assert np.isnan(compute_institution_metrics(y[m], p[m], inst[m]).fairness_gap)
