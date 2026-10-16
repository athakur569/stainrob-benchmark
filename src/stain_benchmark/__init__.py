"""
Stain Robustness Benchmark for Computational Pathology

A systematic benchmark comparing stain normalization and domain adaptation
methods for cross-institutional generalization in computational pathology.

Author: Abhishek Thakur
Institution: Harrisburg University of Science and Technology
"""

__version__ = "0.1.0"
__author__ = "Abhishek Thakur"
__email__ = "a.thakur5690@gmail.com"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "set_seed",
]


def __getattr__(name):
    # Lazy so data/tiling modules import without PyTorch (the tiling env has none).
    if name == "set_seed":
        from stain_benchmark.utils.reproducibility import set_seed

        return set_seed
    raise AttributeError(f"module 'stain_benchmark' has no attribute {name!r}")
