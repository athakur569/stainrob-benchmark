"""Tests for the statistical comparison helpers."""

import numpy as np
import pytest
from scipy import stats

from stain_benchmark.evaluation.statistical_tests import (
    benjamini_hochberg_correction,
    bonferroni_correction,
    bootstrap_ci,
    cohens_d_paired,
    compare_methods,
    holm_bonferroni_correction,
    paired_ttest,
    wilcoxon_test,
)

A = np.array([0.80, 0.82, 0.79, 0.81, 0.80, 0.83])
B = np.array([0.85, 0.86, 0.83, 0.86, 0.84, 0.88])


def test_paired_ttest_matches_scipy():
    result = paired_ttest(A, B)
    expected = stats.ttest_rel(A, B)
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.p_value == pytest.approx(expected.pvalue)
    diff = B - A
    assert result.ci_low < diff.mean() < result.ci_high


def test_paired_cohens_d_by_hand():
    diff = B - A
    assert cohens_d_paired(A, B) == pytest.approx(diff.mean() / diff.std(ddof=1))


def test_wilcoxon_matches_scipy():
    assert wilcoxon_test(A, B).p_value == pytest.approx(stats.wilcoxon(A, B).pvalue)


def test_wilcoxon_cannot_reach_005_with_three_pairs():
    # With 3 seeds the smallest two-sided exact p-value is 2 / 2**3 = 0.25, even when
    # every pair moves the same way. See docs/research_log.md ISS-036.
    result = wilcoxon_test([0.80, 0.81, 0.82], [0.90, 0.92, 0.95])
    assert result.p_value == pytest.approx(0.25)
    assert not result.significant


def test_bonferroni():
    corrected, significant = bonferroni_correction([0.01, 0.02, 0.04])
    assert corrected == pytest.approx([0.03, 0.06, 0.12])
    assert significant == [True, False, False]


def test_holm_matches_textbook_example():
    corrected, _ = holm_bonferroni_correction([0.01, 0.04, 0.03, 0.005])
    assert corrected == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_benjamini_hochberg_matches_textbook_example():
    corrected, _ = benjamini_hochberg_correction([0.01, 0.04, 0.03, 0.005])
    assert corrected == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    data = np.random.default_rng(0).normal(0.8, 0.05, 200)
    est, lo, hi = bootstrap_ci(data, seed=1)
    assert lo < est < hi
    assert bootstrap_ci(data, seed=1) == (est, lo, hi)


def test_compare_methods_applies_correction():
    results = {"baseline": A, "m1": B, "m2": B + 0.01}
    out = compare_methods(results, baseline_name="baseline")
    raw = [paired_ttest(A, B).p_value, paired_ttest(A, B + 0.01).p_value]
    assert out["m1"].p_value_corrected == pytest.approx(min(raw[0] * 2, 1.0))
    assert out["m2"].p_value_corrected == pytest.approx(min(raw[1] * 2, 1.0))
