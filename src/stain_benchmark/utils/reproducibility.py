"""
Reproducibility Utilities

=============================================================================
This module provides utilities to ensure reproducible experiments:

1. set_seed() - Set all random seeds (Python, NumPy, PyTorch)
2. get_reproducibility_info() - Capture environment for logging
3. make_deterministic() - Enable CUDA deterministic mode
4. ReproducibilityContext - Context manager for reproducible blocks
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generator

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility settings."""

    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False  # Set False for reproducibility, True for speed
    warn_only: bool = False  # If True, only warn about non-determinism

    def __post_init__(self) -> None:
        if self.deterministic and self.benchmark:
            logger.warning(
                "deterministic=True and benchmark=True are conflicting. "
                "Setting benchmark=False for reproducibility."
            )
            self.benchmark = False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    This function sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch (CPU and CUDA)
    - CUDA deterministic algorithms (if deterministic=True)

    Args:
        seed: The random seed to use (default: 42).
        deterministic: If True, enable CUDA deterministic mode.
                      This may slow down training but ensures reproducibility.

    Example:
        >>> from stain_benchmark.utils.reproducibility import set_seed
        >>> set_seed(42)  # Call at the start of your script
        >>> # Now all random operations will be reproducible
    """
    # Python random
    random.seed(seed)

    # Environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        if deterministic:
            # Enable deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # PyTorch 1.8+ deterministic flag
            if hasattr(torch, "use_deterministic_algorithms"):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError as e:
                    logger.warning(
                        "Could not enable full deterministic mode: %s. "
                        "Some operations may still be non-deterministic.",
                        str(e),
                    )
        else:
            # Faster but non-deterministic
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    logger.info(
        "Random seed set to %d (deterministic=%s)",
        seed,
        deterministic,
    )


def make_deterministic(warn_only: bool = False) -> None:
    """
    Enable full deterministic mode for PyTorch.

    This enables the strictest deterministic settings. Some operations
    may not have deterministic implementations and will raise errors
    unless warn_only=True.

    Args:
        warn_only: If True, only warn about non-deterministic operations
                  instead of raising errors.

    Note:
        This should be called AFTER set_seed() for complete reproducibility.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

    # Set environment variable for CUDA determinism
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info("Deterministic mode enabled (warn_only=%s)", warn_only)


def get_reproducibility_info() -> dict[str, Any]:
    """
    Capture environment information for reproducibility logging.

    Returns a dictionary containing:
    - Python version
    - PyTorch version
    - CUDA information
    - NumPy version
    - System information
    - Timestamp

    This should be logged at the start of each experiment.

    Returns:
        Dictionary with environment information.

    Example:
        >>> info = get_reproducibility_info()
        >>> import json
        >>> print(json.dumps(info, indent=2))
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "numpy": {
            "version": np.__version__,
        },
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
    }

    # Add CUDA info if available
    if torch.cuda.is_available():
        info["pytorch"]["cuda_version"] = torch.version.cuda
        info["pytorch"]["cudnn_version"] = torch.backends.cudnn.version()
        info["pytorch"]["gpu_count"] = torch.cuda.device_count()
        info["pytorch"]["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        info["pytorch"]["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        info["pytorch"]["cudnn_benchmark"] = torch.backends.cudnn.benchmark

    return info


def compute_experiment_hash(config: dict[str, Any], seed: int) -> str:
    """
    Compute a unique hash for an experiment configuration.

    This hash can be used to:
    - Verify experiment reproducibility
    - Create unique checkpoint names
    - Track experiment versions

    Args:
        config: Experiment configuration dictionary.
        seed: Random seed used.

    Returns:
        16-character hexadecimal hash string.
    """
    import json

    # Add seed to config for hashing
    config_with_seed = {**config, "_seed": seed}

    # Sort keys for determinism
    config_str = json.dumps(config_with_seed, sort_keys=True, default=str)

    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@contextmanager
def reproducible_context(
    seed: int = 42,
    deterministic: bool = True,
) -> Generator[None, None, None]:
    """
    Context manager for reproducible code blocks.

    Saves the current random state, sets the seed, runs the code,
    and optionally restores the state.

    Args:
        seed: Random seed to use within the context.
        deterministic: Whether to enable deterministic mode.

    Example:
        >>> with reproducible_context(seed=123):
        ...     # All random operations here are reproducible
        ...     x = torch.randn(10)
        >>> # Random state is restored here
    """
    # Save current state
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state_all()

    # Set seed
    set_seed(seed, deterministic=deterministic)

    try:
        yield
    finally:
        # Restore state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class WorkerInitFn:
    """
    Worker initialization function for DataLoader reproducibility.

    Each DataLoader worker needs a unique but reproducible seed.
    This class provides a callable that sets worker seeds based on
    the base seed and worker ID.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> worker_init = WorkerInitFn(seed=42)
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=worker_init,
        ... )
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        """Initialize worker with unique seed."""
        worker_seed = self.seed + worker_id

        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        # For debugging
        logger.debug("Worker %d initialized with seed %d", worker_id, worker_seed)


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Create a PyTorch Generator with a specific seed.

    Use this for operations that accept a generator argument,
    like DataLoader's generator parameter.

    Args:
        seed: Random seed for the generator.

    Returns:
        PyTorch Generator object.

    Example:
        >>> g = get_generator(42)
        >>> loader = DataLoader(dataset, generator=g, shuffle=True)
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def log_reproducibility_info(logger: logging.Logger | None = None) -> None:
    """
    Log all reproducibility information.

    Call this at the start of experiments to record the environment.

    Args:
        logger: Logger to use. If None, uses module logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    info = get_reproducibility_info()

    logger.info("=" * 60)
    logger.info("REPRODUCIBILITY INFORMATION")
    logger.info("=" * 60)
    logger.info("Timestamp: %s", info["timestamp"])
    logger.info("Python: %s", info["python"]["version"].split()[0])
    logger.info("PyTorch: %s", info["pytorch"]["version"])
    logger.info("NumPy: %s", info["numpy"]["version"])

    if info["pytorch"]["cuda_available"]:
        logger.info("CUDA: %s", info["pytorch"]["cuda_version"])
        logger.info("cuDNN: %s", info["pytorch"]["cudnn_version"])
        logger.info("GPUs: %s", ", ".join(info["pytorch"]["gpu_names"]))
        logger.info(
            "cuDNN deterministic: %s, benchmark: %s",
            info["pytorch"]["cudnn_deterministic"],
            info["pytorch"]["cudnn_benchmark"],
        )
    else:
        logger.info("CUDA: Not available")

    logger.info("Platform: %s", info["system"]["platform"])
    logger.info("=" * 60)


# =============================================================================
# Module Initialization
# =============================================================================

# Default seed (can be overridden by set_seed())
DEFAULT_SEED = 42


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    print("Setting seed to 42...")
    set_seed(42)

    print("\nGenerating random numbers (should be reproducible):")
    print(f"  Python random: {random.random():.6f}")
    print(f"  NumPy random:  {np.random.random():.6f}")
    print(f"  PyTorch random: {torch.rand(1).item():.6f}")

    print("\nResetting seed and generating again:")
    set_seed(42)
    print(f"  Python random: {random.random():.6f}")
    print(f"  NumPy random:  {np.random.random():.6f}")
    print(f"  PyTorch random: {torch.rand(1).item():.6f}")

    print("\nEnvironment info:")
    log_reproducibility_info()
