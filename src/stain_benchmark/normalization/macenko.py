"""
Macenko Stain Normalization

=============================================================================
Implementation of the Macenko et al. (2009) stain normalization method.

This method performs stain separation using optical density (OD) space and
Singular Value Decomposition (SVD) to find the principal stain vectors.
Images are normalized by mapping their stain vectors to a reference.

Reference:
    Macenko, M., et al. (2009). A method for normalizing histology slides
    for quantitative analysis. IEEE International Symposium on Biomedical
    Imaging: From Nano to Macro.

Key Steps:
    1. Convert RGB to Optical Density (OD) space
    2. Remove background pixels (low OD values)
    3. Compute principal stain vectors via SVD
    4. Project pixels onto stain vectors
    5. Map to reference stain matrix
    6. Reconstruct normalized RGB image
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import linalg

from .base import StainNormalizer

logger = logging.getLogger(__name__)


class MacenkoNormalizer(StainNormalizer):
    """
    Macenko stain normalization.

    This normalizer separates H&E stains using optical density decomposition
    and SVD, then remaps to a reference stain matrix.

    Attributes:
        alpha: Percentile for robust extreme value computation (default: 1).
        beta: Threshold for optical density background removal (default: 0.15).

    Example:
        >>> normalizer = MacenkoNormalizer()
        >>> normalizer.fit(reference_image)
        >>> normalized = normalizer.transform(target_image)

        # Or with custom parameters
        >>> normalizer = MacenkoNormalizer(alpha=1, beta=0.15)
    """

    # Standard H&E stain matrix (from literature)
    # Rows: Hematoxylin, Eosin
    # Columns: R, G, B (OD space)
    HE_REF = np.array([
        [0.5626, 0.7201, 0.4062],  # Hematoxylin
        [0.2159, 0.8012, 0.5581],  # Eosin
    ])

    # Maximum stain concentrations
    MAX_C_REF = np.array([1.9705, 1.0308])

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.15,
        luminosity_threshold: float = 0.8,
        use_standard_ref: bool = False,
    ):
        """
        Initialize Macenko normalizer.

        Args:
            alpha: Percentile for robust min/max (default: 1).
                  Lower values = more robust to outliers.
            beta: OD threshold for background removal (default: 0.15).
                  Pixels with OD below this are considered background.
            luminosity_threshold: Threshold for light pixels (default: 0.8).
            use_standard_ref: If True, use standard H&E reference matrix
                             instead of computing from reference image.
        """
        super().__init__(name="Macenko")

        self.alpha = alpha
        self.beta = beta
        self.luminosity_threshold = luminosity_threshold
        self.use_standard_ref = use_standard_ref

        # Reference statistics (set during fit)
        self._stain_matrix_ref: np.ndarray | None = None
        self._max_c_ref: np.ndarray | None = None

    def fit(self, reference_image: np.ndarray) -> "MacenkoNormalizer":
        """
        Fit normalizer to a reference image.

        Extracts the stain matrix and maximum concentrations from the
        reference image that will be used to normalize other images.

        Args:
            reference_image: RGB reference image (H, W, 3), dtype uint8.

        Returns:
            Self, for method chaining.
        """
        self._validate_image(reference_image, "reference_image")

        if self.use_standard_ref:
            # Use standard H&E reference
            self._stain_matrix_ref = self.HE_REF.copy()
            self._max_c_ref = self.MAX_C_REF.copy()
        else:
            # Compute from reference image
            self._stain_matrix_ref, self._max_c_ref = self._get_stain_matrix(
                reference_image
            )

        # Store for saving/loading
        self._reference_stats = {
            "stain_matrix": self._stain_matrix_ref,
            "max_c": self._max_c_ref,
        }

        self._fitted = True
        logger.info(
            "Macenko normalizer fitted to reference (stain matrix shape: %s)",
            self._stain_matrix_ref.shape,
        )

        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to match reference stain appearance.

        Args:
            image: RGB image to normalize (H, W, 3), dtype uint8.

        Returns:
            Normalized RGB image (H, W, 3), dtype uint8.
        """
        self._check_fitted()
        self._validate_image(image)

        # Get source stain matrix and concentrations
        stain_matrix_src, max_c_src = self._get_stain_matrix(image)

        # Get stain concentrations for each pixel
        concentrations = self._get_concentrations(image, stain_matrix_src)

        # Normalize concentrations
        concentrations *= (self._max_c_ref / max_c_src).reshape(1, -1)

        # Reconstruct image with reference stain matrix
        normalized = self._reconstruct(concentrations, self._stain_matrix_ref)

        return normalized

    def _get_stain_matrix(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract stain matrix from an image using SVD.

        Args:
            image: RGB image (H, W, 3), dtype uint8.

        Returns:
            Tuple of (stain_matrix, max_concentrations).
            stain_matrix: (2, 3) array of stain vectors.
            max_concentrations: (2,) array of max concentrations.
        """
        # Convert to OD space
        od = self._rgb_to_od(image)

        # Flatten to (N, 3)
        od_flat = od.reshape(-1, 3)

        # Remove background (low OD values)
        od_norm = np.linalg.norm(od_flat, axis=1)
        mask = od_norm > self.beta
        od_tissue = od_flat[mask]

        if len(od_tissue) < 10:
            logger.warning(
                "Very few tissue pixels found (%d). Using standard reference.",
                len(od_tissue),
            )
            return self.HE_REF.copy(), self.MAX_C_REF.copy()

        # Compute SVD
        # Center the data
        od_mean = od_tissue.mean(axis=0)
        od_centered = od_tissue - od_mean

        # SVD to find principal directions
        try:
            _, _, Vt = linalg.svd(od_centered, full_matrices=False)
        except linalg.LinAlgError:
            logger.warning("SVD failed. Using standard reference.")
            return self.HE_REF.copy(), self.MAX_C_REF.copy()

        # Project onto the plane spanned by the two principal directions
        V = Vt[:2, :].T  # (3, 2)
        proj = od_tissue @ V  # (N, 2)

        # Find the angle of each point in the plane
        angles = np.arctan2(proj[:, 1], proj[:, 0])

        # Find the extreme angles (robust percentiles)
        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100 - self.alpha)

        # Get the stain vectors from extreme angles
        v1 = np.array([np.cos(min_angle), np.sin(min_angle)])
        v2 = np.array([np.cos(max_angle), np.sin(max_angle)])

        # Convert back to OD space
        stain1 = V @ v1
        stain2 = V @ v2

        # Ensure correct order: H (more blue) vs E (more pink)
        # Hematoxylin has higher OD in red channel relative to blue
        if stain1[0] < stain2[0]:
            stain_matrix = np.vstack([stain1, stain2])
        else:
            stain_matrix = np.vstack([stain2, stain1])

        # Normalize stain vectors
        stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)

        # Get concentrations for max calculation
        concentrations = self._get_concentrations(image, stain_matrix)

        # Robust max concentrations (percentile)
        max_c = np.percentile(concentrations, 99, axis=0)

        return stain_matrix, max_c

    def _get_concentrations(
        self,
        image: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Get stain concentrations for each pixel.

        Args:
            image: RGB image (H, W, 3).
            stain_matrix: (2, 3) stain matrix.

        Returns:
            Concentrations array (H*W, 2).
        """
        # Convert to OD
        od = self._rgb_to_od(image)
        od_flat = od.reshape(-1, 3)

        # Solve for concentrations: OD = C @ stain_matrix
        # C = OD @ stain_matrix.T @ inv(stain_matrix @ stain_matrix.T)
        # Or use least squares
        try:
            concentrations, _, _, _ = np.linalg.lstsq(
                stain_matrix.T, od_flat.T, rcond=None
            )
            concentrations = concentrations.T
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            pinv = np.linalg.pinv(stain_matrix.T)
            concentrations = (pinv @ od_flat.T).T

        # Clip negative values
        concentrations = np.maximum(concentrations, 0)

        return concentrations

    def _reconstruct(
        self,
        concentrations: np.ndarray,
        stain_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct RGB image from concentrations and stain matrix.

        Args:
            concentrations: (N, 2) concentration values.
            stain_matrix: (2, 3) stain matrix.

        Returns:
            RGB image (H, W, 3), dtype uint8.
        """
        # Reconstruct OD
        od_flat = concentrations @ stain_matrix

        # Convert back to RGB
        rgb_flat = self._od_to_rgb(od_flat)

        # Reshape (need to store original shape)
        # For now, assume square-ish
        n_pixels = len(rgb_flat)
        h = int(np.sqrt(n_pixels))
        w = n_pixels // h

        # Handle non-square case
        if h * w != n_pixels:
            # Find factors
            for i in range(int(np.sqrt(n_pixels)), 0, -1):
                if n_pixels % i == 0:
                    h = i
                    w = n_pixels // i
                    break

        return rgb_flat.reshape(h, w, 3)

    def transform_with_shape(
        self,
        image: np.ndarray,
        output_shape: Tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Transform with explicit output shape handling.

        Args:
            image: Input RGB image.
            output_shape: (H, W) output shape. If None, uses input shape.

        Returns:
            Normalized RGB image with correct shape.
        """
        self._check_fitted()
        self._validate_image(image)

        h, w = image.shape[:2]
        if output_shape is None:
            output_shape = (h, w)

        # Get source stain matrix and concentrations
        stain_matrix_src, max_c_src = self._get_stain_matrix(image)

        # Get stain concentrations for each pixel
        concentrations = self._get_concentrations(image, stain_matrix_src)

        # Normalize concentrations
        concentrations *= (self._max_c_ref / max_c_src).reshape(1, -1)

        # Reconstruct OD
        od_flat = concentrations @ self._stain_matrix_ref

        # Convert back to RGB
        rgb_flat = self._od_to_rgb(od_flat)

        # Reshape to original dimensions
        return rgb_flat.reshape(output_shape[0], output_shape[1], 3)

    @staticmethod
    def _rgb_to_od(image: np.ndarray) -> np.ndarray:
        """
        Convert RGB to Optical Density.

        OD = -log10(I / I0), where I0 = 255 for uint8.

        Args:
            image: RGB image (H, W, 3), dtype uint8.

        Returns:
            OD image (H, W, 3), dtype float64.
        """
        # Avoid log(0)
        image = image.astype(np.float64)
        image = np.maximum(image, 1)

        # OD transform
        od = -np.log10(image / 255.0)

        return od

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        """
        Convert Optical Density back to RGB.

        Args:
            od: OD values (N, 3) or (H, W, 3), dtype float64.

        Returns:
            RGB image, dtype uint8.
        """
        # Inverse transform
        rgb = 255.0 * np.power(10, -od)

        # Clip and convert to uint8
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return rgb

    def get_stain_vectors(self) -> np.ndarray | None:
        """Return the reference stain matrix if fitted."""
        return self._stain_matrix_ref.copy() if self._fitted else None


# =============================================================================
# Convenience Functions
# =============================================================================


def macenko_normalize(
    image: np.ndarray,
    reference: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.15,
) -> np.ndarray:
    """
    One-shot Macenko normalization.

    Args:
        image: Image to normalize.
        reference: Reference image.
        alpha: Percentile for robust computation.
        beta: OD threshold for background.

    Returns:
        Normalized image.

    Example:
        >>> normalized = macenko_normalize(target, reference)
    """
    normalizer = MacenkoNormalizer(alpha=alpha, beta=beta)
    return normalizer.fit_transform(reference, image)


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="Test Macenko normalization")
    parser.add_argument("--reference", type=str, required=True, help="Reference image")
    parser.add_argument("--target", type=str, required=True, help="Target image")
    parser.add_argument("--output", type=str, required=True, help="Output image")

    args = parser.parse_args()

    # Load images (BGR -> RGB)
    ref = cv2.imread(args.reference)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

    target = cv2.imread(args.target)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    # Normalize
    normalizer = MacenkoNormalizer()
    normalizer.fit(ref)
    normalized = normalizer.transform_with_shape(target, target.shape[:2])

    # Save (RGB -> BGR)
    normalized_bgr = cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, normalized_bgr)

    print(f"Saved normalized image to {args.output}")
