"""
Evaluation Metrics for Stain Robustness Benchmark

=============================================================================
This module provides all metrics needed for the benchmark:

1. Classification metrics: AUC-ROC, Accuracy, F1, Sensitivity, Specificity
2. Generalization metrics: ΔG (generalization gap)
3. Per-institution AUC and fairness gap
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes for Results
# =============================================================================


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    auc_roc: float
    accuracy: float
    f1: float
    precision: float
    recall: float  # Same as sensitivity
    specificity: float
    threshold: float = 0.5

    # Optional: ROC curve data
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None
    thresholds: np.ndarray | None = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excluding arrays)."""
        return {
            "auc_roc": self.auc_roc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "sensitivity": self.recall,  # Alias
            "specificity": self.specificity,
            "threshold": self.threshold,
        }


@dataclass
class GeneralizationMetrics:
    """Container for generalization gap metrics."""

    in_domain_score: float
    cross_domain_score: float
    delta_g: float  # ΔG = in_domain - cross_domain
    relative_drop: float  # (in_domain - cross_domain) / in_domain

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "in_domain_score": self.in_domain_score,
            "cross_domain_score": self.cross_domain_score,
            "delta_g": self.delta_g,
            "relative_drop": self.relative_drop,
        }


# =============================================================================
# Classification Metrics
# =============================================================================


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1). If None, computed from y_prob.
        y_prob: Predicted probabilities for positive class.
        threshold: Threshold for converting probabilities to labels.

    Returns:
        ClassificationMetrics object with all metrics.

    Example:
        >>> metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        >>> print(f"AUC: {metrics.auc_roc:.4f}")
    """
    y_true = np.asarray(y_true).flatten()

    # Handle predictions
    if y_pred is None and y_prob is not None:
        y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_pred = np.asarray(y_pred).flatten()

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC-ROC (requires probabilities)
    fpr, tpr, thresholds = None, None, None
    if y_prob is not None:
        y_prob = np.asarray(y_prob).flatten()
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        except ValueError:
            # Only one class present
            auc_roc = 0.5
            logger.warning("Only one class present, AUC set to 0.5")
    else:
        auc_roc = 0.5

    return ClassificationMetrics(
        auc_roc=auc_roc,
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        specificity=specificity,
        threshold=threshold,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
) -> float:
    """
    Find optimal classification threshold.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        method: "youden" (maximize sensitivity + specificity - 1) or
                "f1" (maximize F1 score).

    Returns:
        Optimal threshold value.
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx]

    elif method == "f1":
        precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_prob)
        # Compute F1 for each threshold
        f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Last value is for threshold=1
        return thresholds[optimal_idx]

    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Generalization Gap Metrics
# =============================================================================


def compute_generalization_gap(
    in_domain_score: float,
    cross_domain_score: float,
) -> GeneralizationMetrics:
    """
    Compute generalization gap metrics.

    ΔG = in_domain_score - cross_domain_score

    A smaller (or negative) ΔG indicates better generalization.

    Args:
        in_domain_score: Performance on same-domain test set.
        cross_domain_score: Performance on different-domain test set.

    Returns:
        GeneralizationMetrics object.

    Example:
        >>> gap = compute_generalization_gap(in_domain=0.95, cross_domain=0.80)
        >>> print(f"ΔG = {gap.delta_g:.4f}")  # 0.15
    """
    delta_g = in_domain_score - cross_domain_score
    relative_drop = delta_g / (in_domain_score + 1e-10)

    return GeneralizationMetrics(
        in_domain_score=in_domain_score,
        cross_domain_score=cross_domain_score,
        delta_g=delta_g,
        relative_drop=relative_drop,
    )


def compare_generalization_gaps(
    methods: Dict[str, Tuple[float, float]],
) -> Dict[str, GeneralizationMetrics]:
    """
    Compare generalization gaps across multiple methods.

    Args:
        methods: Dict mapping method name to (in_domain_score, cross_domain_score).

    Returns:
        Dict mapping method name to GeneralizationMetrics.

    Example:
        >>> results = compare_generalization_gaps({
        ...     "baseline": (0.90, 0.70),
        ...     "macenko": (0.88, 0.78),
        ...     "dann": (0.85, 0.82),
        ... })
    """
    return {
        name: compute_generalization_gap(in_score, cross_score)
        for name, (in_score, cross_score) in methods.items()
    }


# =============================================================================
# Per-Institution Metrics (ported from pathology-eval-suite)
# =============================================================================


@dataclass
class InstitutionMetrics:
    """Per-institution AUC and the fairness gap across institutions."""

    per_institution: Dict[str, float]
    skipped: Dict[str, str]  # institution -> reason it has no AUC
    fairness_gap: float  # max − min AUC over scored institutions (nan if < 2)
    macro_mean: float
    macro_std: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (summary values plus one auc_<institution> per site)."""
        d = {
            "fairness_gap": self.fairness_gap,
            "macro_mean_auc": self.macro_mean,
            "macro_std_auc": self.macro_std,
        }
        d.update({f"auc_{k}": v for k, v in self.per_institution.items()})
        return d


def compute_institution_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    institution: np.ndarray,
    min_samples: int = 20,
) -> InstitutionMetrics:
    """
    Compute AUC separately for each institution and the gap between best and worst.

    Institutions with fewer than ``min_samples`` samples or only one class are
    reported in ``skipped`` rather than silently dropped, so a missing site is visible.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        institution: Institution (center / TSS) for each sample.
        min_samples: Minimum samples for an institution to get an AUC.

    Returns:
        InstitutionMetrics object.
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()
    institution = np.asarray(institution).flatten()

    per_inst: Dict[str, float] = {}
    skipped: Dict[str, str] = {}
    for inst in np.unique(institution):
        mask = institution == inst
        if mask.sum() < min_samples:
            skipped[str(inst)] = f"{int(mask.sum())} samples < {min_samples}"
        elif len(np.unique(y_true[mask])) < 2:
            skipped[str(inst)] = "only one class present"
        else:
            per_inst[str(inst)] = float(roc_auc_score(y_true[mask], y_prob[mask]))

    aucs = np.array(list(per_inst.values()))
    return InstitutionMetrics(
        per_institution=per_inst,
        skipped=skipped,
        fairness_gap=float(aucs.max() - aucs.min()) if len(aucs) >= 2 else float("nan"),
        macro_mean=float(aucs.mean()) if len(aucs) else float("nan"),
        macro_std=float(aucs.std()) if len(aucs) else float("nan"),
    )


def compute_calibration_slope(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-6,
    sample_weight: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Calibration slope and intercept: logistic regression of the outcome on logit(p).

    A slope of 1 means predicted probabilities spread as much as they should; below 1 the
    model is overconfident (probabilities too extreme), above 1 underconfident. Unlike
    absolute calibration (ECE), the slope does not depend on the sampled tumour fraction,
    which is fixed 1:1 by design in our tile sets (DR-006).

    Args:
        y_true: Ground truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        eps: Probabilities are clipped to [eps, 1 - eps] before the logit.
        sample_weight: Optional per-sample weights (e.g. patient-bootstrap multiplicities).

    Returns:
        (slope, intercept), or (nan, nan) where the slope does not exist (deviation D7): the predictions
        have no spread (every probability clipped to the same value), or the two classes are perfectly
        separated by the predictions, so the logistic fit has no finite maximum, or Newton's method
        does not converge.
    """
    y_true = np.asarray(y_true).flatten().astype(float)
    p = np.clip(np.asarray(y_prob, dtype=float).flatten(), eps, 1 - eps)
    if len(np.unique(y_true)) < 2:
        raise ValueError("calibration slope needs both classes present")
    w = np.ones_like(y_true) if sample_weight is None else np.asarray(sample_weight, dtype=float).flatten()
    lp = np.log(p / (1 - p))
    used = w > 0
    if np.ptp(lp[used]) < 1e-12:
        return float("nan"), float("nan")                                   # no spread in the predictions
    if lp[used & (y_true == 0)].max() < lp[used & (y_true == 1)].min():
        return float("nan"), float("nan")                                   # perfect separation: no finite MLE
    # Unpenalised weighted logistic regression of y on [1, lp] by Newton's method (two parameters,
    # so each step is a 2x2 solve); same estimate as scikit-learn, far faster inside a bootstrap.
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -35, 35)))
        grad = X.T @ (w * (y_true - mu))
        hess = (X * (w * mu * (1 - mu))[:, None]).T @ X
        try:
            step = np.linalg.solve(hess + 1e-12 * np.eye(2), grad)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta += step
        if np.abs(step).max() < 1e-10:
            return float(beta[1]), float(beta[0])
    return float("nan"), float("nan")                                       # did not converge


# =============================================================================
# Aggregation Functions
# =============================================================================


def aggregate_metrics(
    metrics_list: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple runs/folds.

    Args:
        metrics_list: List of metric dictionaries.

    Returns:
        Dict with mean, std, min, max for each metric.

    Example:
        >>> agg = aggregate_metrics([
        ...     {"auc": 0.90, "f1": 0.85},
        ...     {"auc": 0.92, "f1": 0.87},
        ... ])
        >>> print(agg["auc"])  # {"mean": 0.91, "std": 0.01, ...}
    """
    if not metrics_list:
        return {}

    # Get all metric names
    keys = metrics_list[0].keys()

    result = {}
    for key in keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }

    return result
