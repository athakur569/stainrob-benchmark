"""
DANN PyTorch Lightning Module

=============================================================================
Domain-Adversarial Neural Network training module.

This module implements the complete DANN training loop with:
- Feature extractor (ResNet-50 backbone)
- Label classifier (task head)
- Domain classifier with gradient reversal
- Scheduled lambda for gradual domain adaptation

Reference:
    Ganin, Y., et al. (2016). Domain-adversarial training of neural networks.
    JMLR.
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, F1Score
from torchvision import models

from .gradient_reversal import ScheduledGRL, dann_lambda_schedule

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """
    Feature extractor based on ResNet-50.

    Extracts 2048-dimensional features from images using a
    pretrained ResNet-50 backbone (without the final FC layer).
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_layers: int = 0,
    ):
        """
        Initialize feature extractor.

        Args:
            pretrained: Use ImageNet pretrained weights.
            freeze_layers: Number of initial layers to freeze (0-4).
                          0 = train all, 4 = freeze all conv layers.
        """
        super().__init__()

        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Remove final FC layer (keep everything up to avgpool)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = resnet.avgpool

        # Feature dimension
        self.feature_dim = 2048

        # Freeze early layers if requested
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze the first num_layers layer groups."""
        layers = [
            self.features[0:4],  # conv1, bn1, relu, maxpool
            self.features[4],    # layer1
            self.features[5],    # layer2
            self.features[6],    # layer3
        ]

        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False

        logger.info("Froze first %d layer groups", num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Features (B, 2048).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class LabelClassifier(nn.Module):
    """
    Label classifier for the main task (tumor classification).

    Simple MLP head for binary classification.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify features.

        Args:
            x: Features (B, input_dim).

        Returns:
            Logits (B, num_classes).
        """
        return self.classifier(x)


class DomainClassifier(nn.Module):
    """
    Domain classifier for adversarial training.

    Deeper MLP to distinguish source vs. target domain.
    Receives gradient-reversed features during training.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # Binary: source (0) vs target (1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify domain.

        Args:
            x: Features (B, input_dim) - should be gradient-reversed.

        Returns:
            Domain logits (B, 2).
        """
        return self.classifier(x)


class DANNModule(pl.LightningModule):
    """
    PyTorch Lightning module for DANN training.

    Combines:
    - Feature extractor (shared backbone)
    - Label classifier (task-specific head)
    - Domain classifier with gradient reversal

    The gradient reversal layer causes the feature extractor to learn
    domain-invariant representations by adversarially training against
    the domain classifier.
    """

    def __init__(
        self,
        # Architecture
        pretrained: bool = True,
        freeze_layers: int = 0,
        hidden_dim: int = 256,
        domain_hidden_dim: int = 1024,
        dropout: float = 0.5,
        num_classes: int = 2,
        # Training
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        # Lambda scheduling
        lambda_max: float = 1.0,
        lambda_gamma: float = 10.0,
        # Loss weights
        task_weight: float = 1.0,
        domain_weight: float = 1.0,
    ):
        """
        Initialize DANN module.

        Args:
            pretrained: Use ImageNet pretrained weights for backbone.
            freeze_layers: Number of backbone layers to freeze.
            hidden_dim: Hidden dimension for label classifier.
            domain_hidden_dim: Hidden dimension for domain classifier.
            dropout: Dropout rate.
            num_classes: Number of task classes (default: 2 for binary).
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization.
            lambda_max: Maximum value for gradient reversal lambda.
            lambda_gamma: Controls lambda schedule speed.
            task_weight: Weight for task classification loss.
            domain_weight: Weight for domain classification loss.
        """
        super().__init__()

        self.save_hyperparameters()

        # Build networks
        self.feature_extractor = FeatureExtractor(
            pretrained=pretrained,
            freeze_layers=freeze_layers,
        )

        self.label_classifier = LabelClassifier(
            input_dim=self.feature_extractor.feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self.domain_classifier = DomainClassifier(
            input_dim=self.feature_extractor.feature_dim,
            hidden_dim=domain_hidden_dim,
            dropout=dropout,
        )

        # Gradient reversal layer with scheduling
        self.grl = ScheduledGRL(gamma=lambda_gamma, max_lambda=lambda_max)

        # Loss functions
        self.task_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary")

        # Track domain accuracy
        self.train_domain_acc = Accuracy(task="binary")

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Tuple of (task_logits, features).
        """
        features = self.feature_extractor(x)
        task_logits = self.label_classifier(features)
        return task_logits, features

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step with domain adversarial learning.

        Expects batch to contain:
        - 'source_images': Images from source domain
        - 'source_labels': Labels for source images
        - 'target_images': Images from target domain (no labels needed)
        """
        # Update lambda based on training progress
        progress = self.current_epoch / self.trainer.max_epochs
        current_lambda = self.grl.update_lambda(progress)

        # Unpack batch
        source_images = batch["source_images"]
        source_labels = batch["source_labels"]
        target_images = batch["target_images"]

        batch_size_source = source_images.size(0)
        batch_size_target = target_images.size(0)

        # Combine for efficient forward pass
        combined_images = torch.cat([source_images, target_images], dim=0)

        # Extract features
        features = self.feature_extractor(combined_images)

        # Split features
        source_features = features[:batch_size_source]
        target_features = features[batch_size_source:]

        # Task classification (source only - we have labels)
        task_logits = self.label_classifier(source_features)
        task_loss = self.task_criterion(task_logits, source_labels)

        # Domain classification (both domains)
        # Apply gradient reversal
        reversed_features = self.grl(features)
        domain_logits = self.domain_classifier(reversed_features)

        # Domain labels: 0 = source, 1 = target
        domain_labels = torch.cat([
            torch.zeros(batch_size_source, dtype=torch.long, device=self.device),
            torch.ones(batch_size_target, dtype=torch.long, device=self.device),
        ])

        domain_loss = self.domain_criterion(domain_logits, domain_labels)

        # Total loss
        loss = (
            self.hparams.task_weight * task_loss +
            self.hparams.domain_weight * domain_loss
        )

        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/task_loss", task_loss)
        self.log("train/domain_loss", domain_loss)
        self.log("train/lambda", current_lambda)

        # Metrics
        task_preds = task_logits.argmax(dim=1)
        self.train_acc(task_preds, source_labels)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        domain_preds = domain_logits.argmax(dim=1)
        self.train_domain_acc(domain_preds, domain_labels)
        self.log("train/domain_acc", self.train_domain_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """
        Validation step - evaluate on labeled data.

        Expects batch to contain:
        - 'images': Validation images
        - 'labels': Ground truth labels
        """
        images = batch["images"]
        labels = batch["labels"]

        # Forward pass
        task_logits, _ = self(images)
        loss = self.task_criterion(task_logits, labels)

        # Get predictions and probabilities
        probs = F.softmax(task_logits, dim=1)[:, 1]
        preds = task_logits.argmax(dim=1)

        # Update metrics
        self.val_acc(preds, labels)
        self.val_auc(probs, labels)
        self.val_f1(preds, labels)

        # Logging
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Test step - same as validation."""
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Different learning rates for different parts
        params = [
            {
                "params": self.feature_extractor.parameters(),
                "lr": self.hparams.learning_rate * 0.1,  # Lower LR for pretrained
            },
            {
                "params": self.label_classifier.parameters(),
                "lr": self.hparams.learning_rate,
            },
            {
                "params": self.domain_classifier.parameters(),
                "lr": self.hparams.learning_rate,
            },
        ]

        optimizer = torch.optim.AdamW(
            params,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for analysis)."""
        return self.feature_extractor(x)


# =============================================================================
# Simplified DANN for single-domain training
# =============================================================================


class DANNClassifier(pl.LightningModule):
    """
    Simplified DANN module for standard classification.

    Use this when you just want to use the DANN-trained feature extractor
    for downstream classification without domain adaptation.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        learning_rate: float = 1e-4,
        freeze_backbone: bool = True,
    ):
        """
        Initialize classifier using pretrained DANN features.

        Args:
            feature_extractor: Pretrained feature extractor from DANN.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension for classifier.
            dropout: Dropout rate.
            learning_rate: Learning rate.
            freeze_backbone: Whether to freeze the feature extractor.
        """
        super().__init__()

        self.save_hyperparameters(ignore=["feature_extractor"])

        self.feature_extractor = feature_extractor
        self.classifier = LabelClassifier(
            input_dim=feature_extractor.feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        images, labels = batch["images"], batch["labels"]
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["images"], batch["labels"]
        logits = self(images)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        self.val_acc(preds, labels)
        self.val_auc(probs, labels)

        self.log("val/loss", loss)
        self.log("val/acc", self.val_acc, on_epoch=True)
        self.log("val/auc", self.val_auc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
