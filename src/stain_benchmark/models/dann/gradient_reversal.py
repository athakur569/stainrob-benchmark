"""
Gradient Reversal Layer for Domain-Adversarial Training

=============================================================================
The Gradient Reversal Layer (GRL) is the key component that enables
domain-adversarial training. During forward pass, it acts as identity.
During backward pass, it reverses (negates) the gradients.

This causes the feature extractor to learn domain-invariant features:
- Task classifier: wants features that predict labels well
- Domain classifier: wants features that predict domain well
- GRL: reverses domain classifier gradients, so feature extractor
  learns features that CONFUSE the domain classifier

Reference:
    Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by
    backpropagation. ICML.

    Ganin, Y., et al. (2016). Domain-adversarial training of neural networks.
    JMLR.
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Function for autograd.

    Forward pass: identity
    Backward pass: multiply gradient by -lambda

    Usage:
        >>> x = GradientReversalFunction.apply(features, lambda_)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Forward pass - identity function.

        Args:
            ctx: Context for backward pass.
            x: Input tensor.
            lambda_: Gradient reversal strength.

        Returns:
            x unchanged.
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass - reverse gradient.

        Args:
            ctx: Context from forward pass.
            grad_output: Gradient from subsequent layers.

        Returns:
            Tuple of (reversed gradient, None for lambda).
        """
        lambda_ = ctx.lambda_
        grad_input = -lambda_ * grad_output
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) module.

    This layer reverses gradients during backpropagation, enabling
    adversarial training for domain adaptation.

    Args:
        lambda_: Initial reversal strength (default: 1.0).
                Higher values = stronger reversal.
                Usually scheduled from 0 to 1 during training.

    Example:
        >>> grl = GradientReversalLayer(lambda_=1.0)
        >>> reversed_features = grl(features)
    """

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Update the reversal strength."""
        self.lambda_ = lambda_

    def extra_repr(self) -> str:
        return f"lambda={self.lambda_}"


class ScheduledGRL(nn.Module):
    """
    Gradient Reversal Layer with built-in scheduling.

    Lambda is computed based on training progress using the schedule
    from the original DANN paper.

    Schedule: lambda = 2 / (1 + exp(-gamma * p)) - 1
    where p is progress (0 to 1) and gamma controls the schedule speed.

    Args:
        gamma: Schedule speed parameter (default: 10).
        max_lambda: Maximum lambda value (default: 1.0).

    Example:
        >>> grl = ScheduledGRL(gamma=10)
        >>> for epoch in range(num_epochs):
        ...     progress = epoch / num_epochs
        ...     grl.update_lambda(progress)
        ...     reversed_features = grl(features)
    """

    def __init__(self, gamma: float = 10.0, max_lambda: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.max_lambda = max_lambda
        self._lambda = 0.0
        self._progress = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with scheduled gradient reversal."""
        return GradientReversalFunction.apply(x, self._lambda)

    def update_lambda(self, progress: float) -> float:
        """
        Update lambda based on training progress.

        Args:
            progress: Training progress from 0 (start) to 1 (end).

        Returns:
            Current lambda value.
        """
        self._progress = progress
        # Schedule from DANN paper
        self._lambda = self.max_lambda * (
            2.0 / (1.0 + torch.exp(torch.tensor(-self.gamma * progress)).item()) - 1.0
        )
        return self._lambda

    @property
    def lambda_(self) -> float:
        """Current lambda value."""
        return self._lambda

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, max_lambda={self.max_lambda}, current_lambda={self._lambda:.4f}"


# =============================================================================
# Lambda Scheduling Functions
# =============================================================================


def dann_lambda_schedule(
    progress: float,
    gamma: float = 10.0,
    max_lambda: float = 1.0,
) -> float:
    """
    Compute lambda using the DANN schedule.

    lambda = max_lambda * (2 / (1 + exp(-gamma * p)) - 1)

    Args:
        progress: Training progress from 0 to 1.
        gamma: Schedule speed (higher = faster ramp up).
        max_lambda: Maximum lambda value.

    Returns:
        Lambda value.
    """
    import math
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)
