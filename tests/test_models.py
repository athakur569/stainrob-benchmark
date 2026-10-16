"""Smoke tests for the classifier (E1-E3) and DANN (E5) modules, and the gradient reversal layer.

Untrained weights (pretrained=False) on tiny inputs: these check shapes, the 2048-d
feature output the monitor consumes, and the defining property of gradient reversal.
They say nothing about accuracy.
"""

import pytest
import torch

from stain_benchmark.models.classifier.lightning_module import ResNet50Classifier
from stain_benchmark.models.dann.gradient_reversal import (
    GradientReversalLayer,
    ScheduledGRL,
    dann_lambda_schedule,
)
from stain_benchmark.models.dann.lightning_module import DANNModule

X = torch.randn(2, 3, 64, 64)


class TestGradientReversal:
    def test_forward_is_identity(self):
        x = torch.randn(4, 5)
        assert torch.equal(GradientReversalLayer(0.7)(x), x)

    def test_backward_negates_and_scales(self):
        x = torch.randn(4, 5, requires_grad=True)
        GradientReversalLayer(0.7)(x).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, -0.7))

    def test_schedule_runs_from_zero_to_max(self):
        assert dann_lambda_schedule(0.0) == pytest.approx(0.0)
        assert dann_lambda_schedule(1.0) == pytest.approx(1.0, abs=1e-3)
        grl = ScheduledGRL(gamma=10.0, max_lambda=1.0)
        values = [grl.update_lambda(p / 10) for p in range(11)]
        assert values == sorted(values)


class TestClassifier:
    def test_logits_and_features_shapes(self):
        model = ResNet50Classifier(pretrained=False).eval()
        with torch.no_grad():
            assert model(X).shape == (2, 2)
            assert model.get_features(X).shape == (2, 2048)

    def test_probabilities_sum_to_one(self):
        model = ResNet50Classifier(pretrained=False).eval()
        with torch.no_grad():
            p = model.predict_proba(X)
        assert torch.allclose(p.sum(dim=1), torch.ones(2))


class TestDANN:
    def test_logits_and_features_shapes(self):
        model = DANNModule(pretrained=False).eval()
        with torch.no_grad():
            logits, features = model(X)
        assert logits.shape == (2, 2)
        assert features.shape == (2, 2048)
