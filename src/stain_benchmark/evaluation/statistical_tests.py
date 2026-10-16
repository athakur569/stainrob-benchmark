"""
Statistical Testing for Benchmark Evaluation

=============================================================================
This module provides statistical tests for comparing methods:

1. Paired t-tests with Bonferroni correction
2. Bootstrap confidence intervals
3. Cohen's d effect size
4. Wilcoxon signed-rank test (non-parametric alternative)
5. Multiple comparison corrections
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    p_value_corrected: float | None = None
    effect_size: float | None = None
    effect_size_interpretation: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    significant: bool = False
    alpha: float = 0.05

    def to_dict(self) -> Dict[str, float | str | bool]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_value_corrected": self.p_value_corrected,
            "effect_size": self.effect_size,
            "effect_size_interpretation": self.effect_size_interpretation,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "significant": self.significant,
            "alpha": self.alpha,
        }

    def __str__(self) -> str:
        sig_str = "✓" if self.significant else "✗"
        return (
            f"{self.test_name}: statistic={self.statistic:.4f}, "
            f"p={self.p_value:.4f} (corrected: {self.p_value_corrected:.4f}), "
            f"effect_size={self.effect_size:.4f} ({self.effect_size_interpretation}), "
            f"significant={sig_str}"
        )


# =============================================================================
# Paired Tests
# =============================================================================


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalResult:
    """
    Perform paired t-test between two sets of scores.

    Use this when you have matched pairs (e.g., same test set,
    different methods).

    Args:
        scores_a: Scores from method A (one per fold/seed).
        scores_b: Scores from method B (one per fold/seed).
        alpha: Significance level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        StatisticalResult with test statistics.

    Example:
        >>> macenko_aucs = [0.85, 0.87, 0.84, 0.86, 0.88]
        >>> dann_aucs = [0.89, 0.91, 0.88, 0.90, 0.92]
        >>> result = paired_ttest(macenko_aucs, dann_aucs)
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Arrays must have same length for paired test")

    if len(scores_a) < 2:
        raise ValueError("Need at least 2 observations for t-test")

    # Perform test
    statistic, p_value = stats.ttest_rel(scores_a, scores_b, alternative=alternative)

    # Compute effect size (Cohen's d for paired samples)
    effect_size = cohens_d_paired(scores_a, scores_b)
    interpretation = interpret_cohens_d(effect_size)

    # Confidence interval for difference
    diff = scores_b - scores_a
    ci_low, ci_high = stats.t.interval(
        1 - alpha,
        len(diff) - 1,
        loc=diff.mean(),
        scale=stats.sem(diff),
    )

    return StatisticalResult(
        test_name="Paired t-test",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_interpretation=interpretation,
        ci_low=ci_low,
        ci_high=ci_high,
        significant=p_value < alpha,
        alpha=alpha,
    )


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalResult:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Use this when normality assumption may be violated or with small samples.

    Args:
        scores_a: Scores from method A.
        scores_b: Scores from method B.
        alpha: Significance level.
        alternative: "two-sided", "less", or "greater".

    Returns:
        StatisticalResult with test statistics.
    """
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Arrays must have same length for paired test")

    # Perform test
    statistic, p_value = stats.wilcoxon(
        scores_a,
        scores_b,
        alternative=alternative,
        zero_method="wilcox",
    )

    # Effect size: r = Z / sqrt(N)
    # Z approximation from statistic
    n = len(scores_a)
    z_score = stats.norm.ppf(1 - p_value / 2) if p_value > 0 else 0
    effect_size = abs(z_score) / np.sqrt(n)
    interpretation = interpret_rank_biserial(effect_size)

    return StatisticalResult(
        test_name="Wilcoxon signed-rank",
        statistic=statistic,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_interpretation=interpretation,
        significant=p_value < alpha,
        alpha=alpha,
    )


# =============================================================================
# Effect Sizes
# =============================================================================


def cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size for independent samples.

    d = (mean_b - mean_a) / pooled_std

    Args:
        group_a: Scores from group A.
        group_b: Scores from group B.

    Returns:
        Cohen's d value.
    """
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0

    return (group_b.mean() - group_a.mean()) / pooled_std


def cohens_d_paired(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d for paired samples.

    d = mean_diff / std_diff

    Args:
        scores_a: Scores from condition A.
        scores_b: Scores from condition B.

    Returns:
        Cohen's d value.
    """
    diff = np.asarray(scores_b) - np.asarray(scores_a)
    std_diff = diff.std(ddof=1)

    if std_diff == 0:
        return 0.0

    return diff.mean() / std_diff


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Based on Cohen (1988) conventions:
    - Small: d = 0.2
    - Medium: d = 0.5
    - Large: d = 0.8
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def interpret_rank_biserial(r: float) -> str:
    """Interpret rank-biserial correlation (Wilcoxon effect size)."""
    r = abs(r)
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Data array.
        statistic_func: Function to compute statistic (default: mean).
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (default: 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Tuple of (point_estimate, ci_low, ci_high).

    Example:
        >>> scores = [0.85, 0.87, 0.84, 0.86, 0.88]
        >>> estimate, ci_low, ci_high = bootstrap_ci(scores)
        >>> print(f"Mean: {estimate:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    n = len(data)

    # Point estimate
    point_estimate = statistic_func(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        bootstrap_stats.append(statistic_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return point_estimate, ci_low, ci_high


def bootstrap_diff_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the difference between two methods.

    Args:
        scores_a: Scores from method A.
        scores_b: Scores from method B.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (mean_diff, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    n = len(scores_a)

    # Point estimate
    mean_diff = scores_b.mean() - scores_a.mean()

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diff = scores_b[idx].mean() - scores_a[idx].mean()
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - confidence_level
    ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return mean_diff, ci_low, ci_high


# =============================================================================
# Multiple Comparison Corrections
# =============================================================================


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Adjusted p-value = original p-value * number of tests

    Args:
        p_values: List of uncorrected p-values.
        alpha: Significance level.

    Returns:
        Tuple of (corrected_p_values, significant_flags).

    Example:
        >>> p_values = [0.01, 0.03, 0.04, 0.08]
        >>> corrected, significant = bonferroni_correction(p_values)
    """
    n_tests = len(p_values)
    corrected = [min(p * n_tests, 1.0) for p in p_values]
    significant = [p < alpha for p in corrected]

    return corrected, significant


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply Holm-Bonferroni step-down correction.

    More powerful than Bonferroni while controlling family-wise error rate.

    Args:
        p_values: List of uncorrected p-values.
        alpha: Significance level.

    Returns:
        Tuple of (corrected_p_values, significant_flags).
    """
    n_tests = len(p_values)
    sorted_indices = np.argsort(p_values)

    corrected = [0.0] * n_tests
    significant = [False] * n_tests

    for rank, idx in enumerate(sorted_indices):
        # Holm correction factor
        factor = n_tests - rank
        corrected[idx] = min(p_values[idx] * factor, 1.0)

        # Ensure monotonicity
        if rank > 0:
            prev_idx = sorted_indices[rank - 1]
            corrected[idx] = max(corrected[idx], corrected[prev_idx])

        significant[idx] = corrected[idx] < alpha

    return corrected, significant


def benjamini_hochberg_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply Benjamini-Hochberg FDR correction.

    Controls False Discovery Rate instead of Family-Wise Error Rate.
    More powerful but controls a different error type.

    Args:
        p_values: List of uncorrected p-values.
        alpha: Target FDR level.

    Returns:
        Tuple of (corrected_p_values, significant_flags).
    """
    n_tests = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # BH correction
    corrected_sorted = np.zeros(n_tests)
    for i in range(n_tests - 1, -1, -1):
        if i == n_tests - 1:
            corrected_sorted[i] = sorted_p[i]
        else:
            corrected_sorted[i] = min(
                sorted_p[i] * n_tests / (i + 1),
                corrected_sorted[i + 1],
            )
    corrected_sorted = np.minimum(corrected_sorted, 1.0)

    # Reorder to original order
    corrected = np.zeros(n_tests)
    corrected[sorted_indices] = corrected_sorted

    significant = [p < alpha for p in corrected]

    return corrected.tolist(), significant


# =============================================================================
# Comparison Utilities
# =============================================================================


def compare_methods(
    results: Dict[str, np.ndarray],
    baseline_name: str,
    alpha: float = 0.05,
    test: str = "paired_ttest",
    correction: str = "bonferroni",
) -> Dict[str, StatisticalResult]:
    """
    Compare multiple methods against a baseline.

    Args:
        results: Dict mapping method name to array of scores.
        baseline_name: Name of baseline method.
        alpha: Significance level.
        test: "paired_ttest" or "wilcoxon".
        correction: "bonferroni", "holm", or "bh".

    Returns:
        Dict mapping method name to StatisticalResult.

    Example:
        >>> results = {
        ...     "baseline": [0.80, 0.82, 0.79, 0.81, 0.80],
        ...     "macenko": [0.85, 0.87, 0.84, 0.86, 0.85],
        ...     "dann": [0.88, 0.90, 0.87, 0.89, 0.88],
        ... }
        >>> comparisons = compare_methods(results, baseline_name="baseline")
    """
    if baseline_name not in results:
        raise ValueError(f"Baseline '{baseline_name}' not in results")

    baseline_scores = np.asarray(results[baseline_name])
    comparisons = {}
    p_values = []
    method_names = []

    # Run tests
    test_func = paired_ttest if test == "paired_ttest" else wilcoxon_test

    for name, scores in results.items():
        if name == baseline_name:
            continue

        scores = np.asarray(scores)
        result = test_func(baseline_scores, scores, alpha=alpha)
        comparisons[name] = result
        p_values.append(result.p_value)
        method_names.append(name)

    # Apply correction
    if correction == "bonferroni":
        corrected, significant = bonferroni_correction(p_values, alpha)
    elif correction == "holm":
        corrected, significant = holm_bonferroni_correction(p_values, alpha)
    elif correction == "bh":
        corrected, significant = benjamini_hochberg_correction(p_values, alpha)
    else:
        raise ValueError(f"Unknown correction: {correction}")

    # Update results with corrected p-values
    for i, name in enumerate(method_names):
        comparisons[name].p_value_corrected = corrected[i]
        comparisons[name].significant = significant[i]

    return comparisons


def print_comparison_table(
    comparisons: Dict[str, StatisticalResult],
    baseline_name: str = "baseline",
) -> None:
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 80)
    print(f"Statistical Comparisons vs. {baseline_name}")
    print("=" * 80)
    print(
        f"{'Method':<15} {'p-value':<12} {'p-corrected':<12} "
        f"{'Effect Size':<12} {'Interpretation':<15} {'Significant':<10}"
    )
    print("-" * 80)

    for name, result in comparisons.items():
        sig_str = "✓ Yes" if result.significant else "✗ No"
        print(
            f"{name:<15} {result.p_value:<12.4f} "
            f"{result.p_value_corrected:<12.4f} "
            f"{result.effect_size:<12.4f} "
            f"{result.effect_size_interpretation:<15} "
            f"{sig_str:<10}"
        )

    print("=" * 80 + "\n")
