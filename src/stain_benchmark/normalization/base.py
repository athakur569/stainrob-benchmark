"""
Base Class for Stain Normalization

=============================================================================
Abstract base class defining the interface for all stain normalization methods.

All normalization methods must implement:
- fit(): Learn parameters from a reference image
- transform(): Apply normalization to a new image
- fit_transform(): Convenience method combining both
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class StainNormalizer(ABC):
    """
    Abstract base class for stain normalization methods.

    All normalization methods should inherit from this class and implement
    the required abstract methods.

    Attributes:
        name: Human-readable name of the normalizer.
        fitted: Whether the normalizer has been fitted to a reference.

    Example:
        >>> normalizer = MacenkoNormalizer()
        >>> normalizer.fit(reference_image)
        >>> normalized = normalizer.transform(target_image)
    """

    def __init__(self, name: str = "BaseNormalizer"):
        """
        Initialize the normalizer.

        Args:
            name: Human-readable name for this normalizer.
        """
        self.name = name
        self._fitted = False
        self._reference_stats: dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        """Whether the normalizer has been fitted to a reference."""
        return self._fitted

    @abstractmethod
    def fit(self, reference_image: np.ndarray) -> "StainNormalizer":
        """
        Fit the normalizer to a reference image.

        The reference image defines the target stain appearance that
        all other images will be normalized to match.

        Args:
            reference_image: RGB image as numpy array (H, W, 3), dtype uint8.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If image format is invalid.
        """
        pass

    @abstractmethod
    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Transform an image to match the reference stain appearance.

        Args:
            image: RGB image as numpy array (H, W, 3), dtype uint8.

        Returns:
            Normalized RGB image as numpy array (H, W, 3), dtype uint8.

        Raises:
            RuntimeError: If normalizer has not been fitted.
            ValueError: If image format is invalid.
        """
        pass

    def fit_transform(
        self,
        reference_image: np.ndarray,
        target_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Fit to reference and transform target (or reference if target is None).

        Args:
            reference_image: Reference image to fit to.
            target_image: Image to transform. If None, transforms reference.

        Returns:
            Normalized image.
        """
        self.fit(reference_image)
        if target_image is None:
            return self.transform(reference_image)
        return self.transform(target_image)

    def transform_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """
        Transform a batch of images.

        Args:
            images: List of RGB images.

        Returns:
            List of normalized RGB images.
        """
        return [self.transform(img) for img in images]

    def save(self, path: str | Path) -> None:
        """
        Save normalizer parameters to file.

        Args:
            path: Path to save parameters (as .npz file).
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted normalizer")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            name=self.name,
            fitted=self._fitted,
            **self._reference_stats,
        )

    def load(self, path: str | Path) -> "StainNormalizer":
        """
        Load normalizer parameters from file.

        Args:
            path: Path to load parameters from.

        Returns:
            Self, for method chaining.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalizer file not found: {path}")

        data = np.load(path, allow_pickle=True)
        self._fitted = bool(data["fitted"])
        self._reference_stats = {
            k: data[k] for k in data.files if k not in ["name", "fitted"]
        }

        return self

    def _validate_image(self, image: np.ndarray, name: str = "image") -> None:
        """
        Validate image format.

        Args:
            image: Image to validate.
            name: Name for error messages.

        Raises:
            ValueError: If image format is invalid.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(f"{name} must be a numpy array, got {type(image)}")

        if image.ndim != 3:
            raise ValueError(f"{name} must be 3D (H, W, C), got shape {image.shape}")

        if image.shape[2] != 3:
            raise ValueError(f"{name} must have 3 channels (RGB), got {image.shape[2]}")

        if image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8, got {image.dtype}")

    def _check_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted. "
                f"Call fit() with a reference image first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"


class IdentityNormalizer(StainNormalizer):
    """
    Identity normalizer that returns images unchanged.

    Use this as a baseline/control in experiments.
    """

    def __init__(self):
        super().__init__(name="Identity")

    def fit(self, reference_image: np.ndarray) -> "IdentityNormalizer":
        """No-op fit (identity doesn't need reference)."""
        self._validate_image(reference_image, "reference_image")
        self._fitted = True
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return image unchanged."""
        self._check_fitted()
        self._validate_image(image)
        return image.copy()
