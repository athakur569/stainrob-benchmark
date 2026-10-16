"""Tests for per-institution metrics."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from stain_benchmark.evaluation.metrics import compute_institution_metrics


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
