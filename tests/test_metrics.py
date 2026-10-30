"""Tests for per-institution metrics."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from stain_benchmark.evaluation.metrics import compute_calibration_slope, compute_institution_metrics


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


class TestCalibrationSlope:
    def _simulate(self, true_slope, n=20000, seed=0):
        # Outcome drawn from sigmoid(true_slope * logit(p)): the model's probabilities p are
        # well calibrated only when true_slope == 1.
        rng = np.random.default_rng(seed)
        lp = rng.normal(0, 2, n)
        y = rng.random(n) < 1 / (1 + np.exp(-true_slope * lp))
        return y.astype(int), 1 / (1 + np.exp(-lp))

    @pytest.mark.parametrize("true_slope", [0.5, 1.0, 1.5])
    def test_recovers_known_slope(self, true_slope):
        y, p = self._simulate(true_slope)
        slope, intercept = compute_calibration_slope(y, p)
        assert slope == pytest.approx(true_slope, abs=0.06)
        assert intercept == pytest.approx(0.0, abs=0.06)

    def test_unaffected_by_resampling_prevalence(self):
        # Keeping all positives but only a third of negatives changes prevalence (and so ECE)
        # but leaves the slope unchanged; only the intercept moves.
        y, p = self._simulate(1.0, n=60000)
        keep = (y == 1) | (np.arange(len(y)) % 3 == 0)
        slope, intercept = compute_calibration_slope(y[keep], p[keep])
        assert slope == pytest.approx(1.0, abs=0.06)
        assert intercept == pytest.approx(np.log(3), abs=0.1)

    def test_single_class_raises(self):
        with pytest.raises(ValueError, match="both classes"):
            compute_calibration_slope(np.ones(10), np.full(10, 0.7))
