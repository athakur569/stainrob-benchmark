"""
Evaluation Metrics for Stain Robustness Benchmark

=============================================================================
This module provides all metrics needed for the benchmark:

1. Classification metrics: AUC-ROC, Accuracy, F1, Sensitivity, Specificity
2. Segmentation metrics: Dice, IoU, Pixel Precision/Recall
3. Normalization metrics: SSIM, MS-SSIM, PSNR, Color EMD
4. Generalization metrics: ΔG (generalization gap)
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.ndimage import distance_transform_edt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.metrics import (
    accuracy_score,
    auc,
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
class SegmentationMetrics:
    """Container for segmentation metrics."""

    dice: float
    iou: float
    precision: float
    recall: float
    hausdorff_95: float | None = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        d = {
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
        }
        if self.hausdorff_95 is not None:
            d["hausdorff_95"] = self.hausdorff_95
        return d


@dataclass
class NormalizationMetrics:
    """Container for stain normalization quality metrics."""

    ssim: float
    ms_ssim: float | None = None
    psnr: float = 0.0
    color_emd: float | None = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        d = {"ssim": self.ssim, "psnr": self.psnr}
        if self.ms_ssim is not None:
            d["ms_ssim"] = self.ms_ssim
        if self.color_emd is not None:
            d["color_emd"] = self.color_emd
        return d


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
# Segmentation Metrics
# =============================================================================


def compute_segmentation_metrics(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    compute_hausdorff: bool = True,
) -> SegmentationMetrics:
    """
    Compute comprehensive segmentation metrics.

    Args:
        mask_true: Ground truth binary mask.
        mask_pred: Predicted binary mask.
        compute_hausdorff: Whether to compute Hausdorff distance (slower).

    Returns:
        SegmentationMetrics object with all metrics.

    Example:
        >>> metrics = compute_segmentation_metrics(mask_true, mask_pred)
        >>> print(f"Dice: {metrics.dice:.4f}")
    """
    mask_true = np.asarray(mask_true).astype(bool)
    mask_pred = np.asarray(mask_pred).astype(bool)

    # Intersection and union
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()

    # Dice coefficient
    dice = 2 * intersection / (mask_true.sum() + mask_pred.sum() + 1e-10)

    # IoU (Jaccard index)
    iou = intersection / (union + 1e-10)

    # Pixel-wise precision and recall
    tp = intersection
    fp = np.logical_and(mask_pred, ~mask_true).sum()
    fn = np.logical_and(~mask_pred, mask_true).sum()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    # Hausdorff distance (95th percentile)
    hausdorff_95 = None
    if compute_hausdorff and mask_true.sum() > 0 and mask_pred.sum() > 0:
        hausdorff_95 = compute_hausdorff_distance(mask_true, mask_pred, percentile=95)

    return SegmentationMetrics(
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        hausdorff_95=hausdorff_95,
    )


def compute_dice(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Compute Dice coefficient.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Args:
        mask_true: Ground truth binary mask.
        mask_pred: Predicted binary mask.

    Returns:
        Dice coefficient (0 to 1).
    """
    mask_true = np.asarray(mask_true).astype(bool)
    mask_pred = np.asarray(mask_pred).astype(bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    return 2 * intersection / (mask_true.sum() + mask_pred.sum() + 1e-10)


def compute_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Compute IoU (Intersection over Union).

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        mask_true: Ground truth binary mask.
        mask_pred: Predicted binary mask.

    Returns:
        IoU score (0 to 1).
    """
    mask_true = np.asarray(mask_true).astype(bool)
    mask_pred = np.asarray(mask_pred).astype(bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()

    return intersection / (union + 1e-10)


def compute_hausdorff_distance(
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    percentile: float = 95,
) -> float:
    """
    Compute Hausdorff distance between two masks.

    Args:
        mask_true: Ground truth binary mask.
        mask_pred: Predicted binary mask.
        percentile: Percentile for robust Hausdorff (default: 95).

    Returns:
        Hausdorff distance in pixels.
    """
    # Get boundaries
    from scipy.ndimage import binary_erosion

    boundary_true = mask_true ^ binary_erosion(mask_true)
    boundary_pred = mask_pred ^ binary_erosion(mask_pred)

    # Distance transforms
    dt_true = distance_transform_edt(~boundary_true)
    dt_pred = distance_transform_edt(~boundary_pred)

    # Distances from pred boundary to true boundary
    dist_pred_to_true = dt_true[boundary_pred]
    # Distances from true boundary to pred boundary
    dist_true_to_pred = dt_pred[boundary_true]

    if len(dist_pred_to_true) == 0 or len(dist_true_to_pred) == 0:
        return float("inf")

    # Percentile Hausdorff
    hd_pred_to_true = np.percentile(dist_pred_to_true, percentile)
    hd_true_to_pred = np.percentile(dist_true_to_pred, percentile)

    return max(hd_pred_to_true, hd_true_to_pred)


# =============================================================================
# Normalization Quality Metrics
# =============================================================================


def compute_normalization_metrics(
    original: np.ndarray,
    normalized: np.ndarray,
    reference: np.ndarray | None = None,
) -> NormalizationMetrics:
    """
    Compute metrics for stain normalization quality.

    Args:
        original: Original image before normalization.
        normalized: Image after normalization.
        reference: Reference image (for color EMD).

    Returns:
        NormalizationMetrics object.

    Example:
        >>> metrics = compute_normalization_metrics(original, normalized)
        >>> print(f"SSIM: {metrics.ssim:.4f}")
    """
    original = np.asarray(original)
    normalized = np.asarray(normalized)

    # SSIM (structure preservation)
    # Compare normalized to original to measure how much structure changed
    ssim = compute_ssim(original, normalized)

    # PSNR
    psnr = compute_psnr(original, normalized)

    # MS-SSIM (if images are large enough)
    ms_ssim = None
    if min(original.shape[:2]) >= 160:
        try:
            ms_ssim = compute_ms_ssim(original, normalized)
        except Exception as e:
            logger.warning("MS-SSIM computation failed: %s", e)

    # Color EMD (if reference provided)
    color_emd = None
    if reference is not None:
        color_emd = compute_color_emd(normalized, reference)

    return NormalizationMetrics(
        ssim=ssim,
        ms_ssim=ms_ssim,
        psnr=psnr,
        color_emd=color_emd,
    )


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    multichannel: bool = True,
) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        image1: First image.
        image2: Second image.
        multichannel: Whether images have multiple channels.

    Returns:
        SSIM score (-1 to 1, higher is better).
    """
    # Ensure same dtype
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    if image1.dtype != image2.dtype:
        image1 = image1.astype(np.float64)
        image2 = image2.astype(np.float64)

    # Determine channel axis
    channel_axis = 2 if multichannel and image1.ndim == 3 else None

    return structural_similarity(
        image1,
        image2,
        channel_axis=channel_axis,
        data_range=255 if image1.dtype == np.uint8 else 1.0,
    )


def compute_ms_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    weights: List[float] | None = None,
) -> float:
    """
    Compute Multi-Scale SSIM.

    Args:
        image1: First image.
        image2: Second image.
        weights: Weights for each scale (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).

    Returns:
        MS-SSIM score (0 to 1, higher is better).
    """
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    levels = len(weights)
    ssim_values = []
    cs_values = []

    for i in range(levels):
        # Compute SSIM at this scale
        ssim_val = compute_ssim(image1, image2)
        ssim_values.append(ssim_val)

        # Downsample for next scale
        if i < levels - 1:
            image1 = image1[::2, ::2]
            image2 = image2[::2, ::2]

            if min(image1.shape[:2]) < 11:
                # Image too small, use remaining scales
                break

    # Combine across scales
    ms_ssim = np.prod([s ** w for s, w in zip(ssim_values, weights[: len(ssim_values)])])

    return float(ms_ssim)


def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        image1: First image.
        image2: Second image.

    Returns:
        PSNR in dB (higher is better).
    """
    data_range = 255 if image1.dtype == np.uint8 else 1.0
    return peak_signal_noise_ratio(image1, image2, data_range=data_range)


def compute_color_emd(
    image1: np.ndarray,
    image2: np.ndarray,
    n_bins: int = 32,
) -> float:
    """
    Compute Earth Mover's Distance between color histograms.

    Measures how different the color distributions are.

    Args:
        image1: First image.
        image2: Second image.
        n_bins: Number of histogram bins per channel.

    Returns:
        EMD value (lower is better, 0 = identical distributions).
    """
    from scipy.stats import wasserstein_distance

    image1 = np.asarray(image1).reshape(-1, 3)
    image2 = np.asarray(image2).reshape(-1, 3)

    # Compute EMD for each channel
    emd_total = 0.0
    for c in range(3):
        # Compute histograms
        hist1, bin_edges = np.histogram(image1[:, c], bins=n_bins, range=(0, 255), density=True)
        hist2, _ = np.histogram(image2[:, c], bins=n_bins, range=(0, 255), density=True)

        # Wasserstein distance (1D EMD)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        emd_total += wasserstein_distance(bin_centers, bin_centers, hist1, hist2)

    return emd_total / 3  # Average across channels


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


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing metrics module...")

    # Test classification metrics
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.random.random(100)
    y_pred = (y_prob > 0.5).astype(int)

    cls_metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    print("\nClassification Metrics:")
    for k, v in cls_metrics.to_dict().items():
        print(f"  {k}: {v:.4f}")

    # Test segmentation metrics
    mask_true = np.random.randint(0, 2, (100, 100))
    mask_pred = np.random.randint(0, 2, (100, 100))

    seg_metrics = compute_segmentation_metrics(mask_true, mask_pred)
    print("\nSegmentation Metrics:")
    for k, v in seg_metrics.to_dict().items():
        print(f"  {k}: {v:.4f}")

    # Test generalization gap
    gap = compute_generalization_gap(0.90, 0.75)
    print("\nGeneralization Gap:")
    for k, v in gap.to_dict().items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ All tests passed!")
