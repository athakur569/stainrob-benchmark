"""
Reinhard Stain Normalization

=============================================================================
Implementation of the Reinhard et al. (2001) color transfer method adapted
for histopathology stain normalization.

This method performs color transfer in LAB color space by matching the
mean and standard deviation of each channel to a reference image.

Reference:
    Reinhard, E., et al. (2001). Color transfer between images.
    IEEE Computer Graphics and Applications, 21(5), 34-41.

Key Steps:
    1. Convert RGB to LAB color space
    2. Compute mean and std for each channel (L, a, b)
    3. Normalize: (x - mean_src) / std_src * std_ref + mean_ref
    4. Convert back to RGB
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .base import StainNormalizer

logger = logging.getLogger(__name__)


def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 CIE Lab with L in [0, 100] and a, b in about [-127, 127].

    OpenCV's uint8 path rescales Lab into 0-255 (L*255/100, a/b offset by 128); the float
    path below keeps true CIE units, which the thresholds and clipping in this module assume."""
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float32 CIE Lab -> uint8 RGB (OpenCV's float path returns RGB in [0, 1])."""
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


@dataclass
class LABStats:
    """Statistics for LAB color channels."""
    l_mean: float
    l_std: float
    a_mean: float
    a_std: float
    b_mean: float
    b_std: float

    def to_array(self) -> np.ndarray:
        """Convert to array [l_mean, l_std, a_mean, a_std, b_mean, b_std]."""
        return np.array([
            self.l_mean, self.l_std,
            self.a_mean, self.a_std,
            self.b_mean, self.b_std,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "LABStats":
        """Create from array."""
        return cls(
            l_mean=arr[0], l_std=arr[1],
            a_mean=arr[2], a_std=arr[3],
            b_mean=arr[4], b_std=arr[5],
        )


class ReinhardNormalizer(StainNormalizer):
    """
    Reinhard color transfer for stain normalization.

    This normalizer transfers color statistics in LAB space from a reference
    image to normalize the stain appearance of histopathology images.

    Attributes:
        luminosity_std_correction: If True, also normalizes L channel std.

    Example:
        >>> normalizer = ReinhardNormalizer()
        >>> normalizer.fit(reference_image)
        >>> normalized = normalizer.transform(target_image)
    """

    # Standard reference statistics (average H&E slide)
    # These can be used if no reference image is available
    STANDARD_REF_STATS = LABStats(
        l_mean=76.0,
        l_std=12.5,
        a_mean=0.0,
        a_std=8.0,
        b_mean=0.0,
        b_std=10.0,
    )

    def __init__(
        self,
        luminosity_std_correction: bool = True,
        mask_background: bool = True,
        background_threshold: float = 0.8,
        use_standard_ref: bool = False,
    ):
        """
        Initialize Reinhard normalizer.

        Args:
            luminosity_std_correction: If True, normalize L channel std too.
                                      If False, only normalize mean.
            mask_background: If True, exclude background pixels from statistics.
            background_threshold: Pixels with L > threshold * 100 are background.
            use_standard_ref: If True, use standard reference statistics.
        """
        super().__init__(name="Reinhard")

        self.luminosity_std_correction = luminosity_std_correction
        self.mask_background = mask_background
        self.background_threshold = background_threshold
        self.use_standard_ref = use_standard_ref

        # Reference statistics (set during fit)
        self._ref_stats: LABStats | None = None

    def fit(self, reference_image: np.ndarray) -> "ReinhardNormalizer":
        """
        Fit normalizer to a reference image.

        Computes the LAB color statistics from the reference image.

        Args:
            reference_image: RGB reference image (H, W, 3), dtype uint8.

        Returns:
            Self, for method chaining.
        """
        self._validate_image(reference_image, "reference_image")

        if self.use_standard_ref:
            self._ref_stats = self.STANDARD_REF_STATS
        else:
            self._ref_stats = self._compute_stats(reference_image)

        # Store for saving/loading
        self._reference_stats = {
            "lab_stats": self._ref_stats.to_array(),
        }

        self._fitted = True
        logger.info(
            "Reinhard normalizer fitted (L: %.1f±%.1f, a: %.1f±%.1f, b: %.1f±%.1f)",
            self._ref_stats.l_mean,
            self._ref_stats.l_std,
            self._ref_stats.a_mean,
            self._ref_stats.a_std,
            self._ref_stats.b_mean,
            self._ref_stats.b_std,
        )

        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to match reference color statistics.

        Args:
            image: RGB image to normalize (H, W, 3), dtype uint8.

        Returns:
            Normalized RGB image (H, W, 3), dtype uint8.
        """
        self._check_fitted()
        self._validate_image(image)

        # Compute source statistics
        src_stats = self._compute_stats(image)

        lab = _rgb_to_lab(image)
        L, a, b = cv2.split(lab)

        # Normalize each channel
        # L channel
        if self.luminosity_std_correction and src_stats.l_std > 0:
            L = (L - src_stats.l_mean) / src_stats.l_std * self._ref_stats.l_std + self._ref_stats.l_mean
        else:
            L = L - src_stats.l_mean + self._ref_stats.l_mean

        # a channel
        if src_stats.a_std > 0:
            a = (a - src_stats.a_mean) / src_stats.a_std * self._ref_stats.a_std + self._ref_stats.a_mean
        else:
            a = a - src_stats.a_mean + self._ref_stats.a_mean

        # b channel
        if src_stats.b_std > 0:
            b = (b - src_stats.b_mean) / src_stats.b_std * self._ref_stats.b_std + self._ref_stats.b_mean
        else:
            b = b - src_stats.b_mean + self._ref_stats.b_mean

        # Clip to valid range
        L = np.clip(L, 0, 100)
        a = np.clip(a, -128, 127)
        b = np.clip(b, -128, 127)

        # Merge and convert back to RGB
        rgb_normalized = _lab_to_rgb(cv2.merge([L, a, b]))

        return rgb_normalized

    def _compute_stats(self, image: np.ndarray) -> LABStats:
        """
        Compute LAB statistics for an image.

        Args:
            image: RGB image (H, W, 3), dtype uint8.

        Returns:
            LABStats object with mean and std for each channel.
        """
        # Convert to LAB
        lab = _rgb_to_lab(image)

        # Split channels
        L, a, b = cv2.split(lab)

        # Create tissue mask if needed
        if self.mask_background:
            # Background is high luminosity (white areas)
            threshold = self.background_threshold * 100  # LAB L is 0-100
            mask = L < threshold

            if mask.sum() < 100:
                # Very few tissue pixels, use all pixels
                logger.warning(
                    "Few tissue pixels found (%d), using all pixels",
                    mask.sum()
                )
                mask = np.ones_like(L, dtype=bool)
        else:
            mask = np.ones_like(L, dtype=bool)

        # Compute statistics on tissue pixels
        return LABStats(
            l_mean=float(L[mask].mean()),
            l_std=float(L[mask].std()),
            a_mean=float(a[mask].mean()),
            a_std=float(a[mask].std()),
            b_mean=float(b[mask].mean()),
            b_std=float(b[mask].std()),
        )

    def load(self, path) -> "ReinhardNormalizer":
        """Load normalizer from file."""
        super().load(path)

        # Reconstruct LABStats
        if "lab_stats" in self._reference_stats:
            self._ref_stats = LABStats.from_array(
                self._reference_stats["lab_stats"]
            )

        return self

    def get_reference_stats(self) -> LABStats | None:
        """Return the reference statistics if fitted."""
        return self._ref_stats


class ReinhardRobustNormalizer(ReinhardNormalizer):
    """
    Reinhard normalizer with robust statistics (median, MAD).

    Uses median instead of mean and MAD instead of std for better
    robustness to outliers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Reinhard-Robust"

    def _compute_stats(self, image: np.ndarray) -> LABStats:
        """Compute robust statistics using median and MAD."""
        # Convert to LAB
        lab = _rgb_to_lab(image)
        L, a, b = cv2.split(lab)

        # Create tissue mask
        if self.mask_background:
            threshold = self.background_threshold * 100
            mask = L < threshold
            if mask.sum() < 100:
                mask = np.ones_like(L, dtype=bool)
        else:
            mask = np.ones_like(L, dtype=bool)

        def mad(x: np.ndarray) -> float:
            """Median Absolute Deviation."""
            return float(np.median(np.abs(x - np.median(x))))

        # Compute robust statistics
        # Scale MAD to approximate std: std ≈ 1.4826 * MAD for normal distribution
        scale = 1.4826

        return LABStats(
            l_mean=float(np.median(L[mask])),
            l_std=scale * mad(L[mask]),
            a_mean=float(np.median(a[mask])),
            a_std=scale * mad(a[mask]),
            b_mean=float(np.median(b[mask])),
            b_std=scale * mad(b[mask]),
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def reinhard_normalize(
    image: np.ndarray,
    reference: np.ndarray,
    luminosity_std_correction: bool = True,
) -> np.ndarray:
    """
    One-shot Reinhard normalization.

    Args:
        image: Image to normalize.
        reference: Reference image.
        luminosity_std_correction: Whether to normalize L channel std.

    Returns:
        Normalized image.

    Example:
        >>> normalized = reinhard_normalize(target, reference)
    """
    normalizer = ReinhardNormalizer(
        luminosity_std_correction=luminosity_std_correction
    )
    return normalizer.fit_transform(reference, image)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Reinhard normalization")
    parser.add_argument("--reference", type=str, required=True, help="Reference image")
    parser.add_argument("--target", type=str, required=True, help="Target image")
    parser.add_argument("--output", type=str, required=True, help="Output image")
    parser.add_argument("--robust", action="store_true", help="Use robust statistics")

    args = parser.parse_args()

    # Load images (BGR -> RGB)
    ref = cv2.imread(args.reference)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    target = cv2.imread(args.target)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    # Normalize
    if args.robust:
        normalizer = ReinhardRobustNormalizer()
    else:
        normalizer = ReinhardNormalizer()

    normalizer.fit(ref)
    normalized = normalizer.transform(target)

    # Save (RGB -> BGR)
    normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, normalized_bgr)

    print(f"Saved normalized image to {args.output}")
