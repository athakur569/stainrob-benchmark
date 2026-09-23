"""
ResNet-50 Classifier PyTorch Lightning Module

=============================================================================
Standard ResNet-50 classifier for binary tumor detection.

This module is used as the downstream task to evaluate the quality
of different stain normalization/adaptation methods. The same model
architecture is used across all experiments for fair comparison.
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score, Precision, Recall
from torchvision import models

logger = logging.getLogger(__name__)


class ResNet50Classifier(pl.LightningModule):
    """
    ResNet-50 based classifier for binary tumor detection.

    Features:
    - ImageNet pretrained backbone
    - Optional layer freezing for transfer learning
    - Label smoothing for regularization
    - Comprehensive metrics tracking
    """

    def __init__(
        self,
        # Architecture
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,  # 0-4, number of layer groups to freeze
        dropout: float = 0.5,
        # Training
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        label_smoothing: float = 0.0,
        # LR scheduling
        scheduler: str = "cosine",  # "cosine", "step", "none"
        warmup_epochs: int = 5,
        # Class weights for imbalanced data
        class_weights: Optional[List[float]] = None,
    ):
        """
        Initialize ResNet-50 classifier.

        Args:
            num_classes: Number of output classes (default: 2 for binary).
            pretrained: Use ImageNet pretrained weights.
            freeze_backbone: Freeze all backbone layers.
            freeze_layers: Number of layer groups to freeze (0-4).
            dropout: Dropout before final FC layer.
            learning_rate: Initial learning rate.
            weight_decay: L2 regularization weight.
            label_smoothing: Label smoothing factor (0 = none).
            scheduler: LR scheduler type.
            warmup_epochs: Number of warmup epochs.
            class_weights: Optional weights for imbalanced classes.
        """
        super().__init__()

        self.save_hyperparameters()

        # Build model
        self._build_model()

        # Loss function
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float)
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )

        # Metrics
        self._setup_metrics()

    def _build_model(self) -> None:
        """Build the ResNet-50 model."""
        # Load pretrained ResNet-50
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V2
            if self.hparams.pretrained
            else None
        )
        self.backbone = models.resnet50(weights=weights)

        # Get feature dimension
        num_features = self.backbone.fc.in_features

        # Replace final FC layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(self.hparams.dropout),
            nn.Linear(num_features, self.hparams.num_classes),
        )

        # Freeze layers if requested
        if self.hparams.freeze_backbone:
            self._freeze_all_backbone()
        elif self.hparams.freeze_layers > 0:
            self._freeze_layers(self.hparams.freeze_layers)

    def _freeze_all_backbone(self) -> None:
        """Freeze all backbone layers except final FC."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
        logger.info("Froze all backbone layers")

    def _freeze_layers(self, num_layers: int) -> None:
        """Freeze first num_layers layer groups."""
        layer_names = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        layers_to_freeze = layer_names[: num_layers + 2]  # +2 for conv1, bn1

        for name, param in self.backbone.named_parameters():
            for layer_name in layers_to_freeze:
                if name.startswith(layer_name):
                    param.requires_grad = False
                    break

        logger.info("Froze layers: %s", layers_to_freeze)

    def _setup_metrics(self) -> None:
        """Initialize metrics."""
        # Training metrics
        self.train_acc = Accuracy(task="binary")

        # Validation metrics
        self.val_acc = Accuracy(task="binary")
        self.val_auc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")  # Sensitivity

        # Test metrics (same as validation)
        self.test_acc = Accuracy(task="binary")
        self.test_auc = AUROC(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_precision = Precision(task="binary")
        self.test_recall = Recall(task="binary")
        self.test_confusion = ConfusionMatrix(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Logits (B, num_classes).
        """
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self(x)
        return F.softmax(logits, dim=1)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        images = batch["images"]
        labels = batch["labels"]

        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Predictions
        preds = logits.argmax(dim=1)

        # Update metrics
        self.train_acc(preds, labels)

        # Logging
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Validation step."""
        images = batch["images"]
        labels = batch["labels"]

        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Predictions and probabilities
        probs = F.softmax(logits, dim=1)[:, 1]  # Probability of positive class
        preds = logits.argmax(dim=1)

        # Update metrics
        self.val_acc(preds, labels)
        self.val_auc(probs, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)

        # Logging
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Test step."""
        images = batch["images"]
        labels = batch["labels"]

        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Predictions and probabilities
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        # Update metrics
        self.test_acc(preds, labels)
        self.test_auc(probs, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_confusion(preds, labels)

        # Logging
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True)

    def on_test_epoch_end(self) -> None:
        """Log confusion matrix at end of test."""
        confusion = self.test_confusion.compute()
        tn, fp, fn, tp = confusion.flatten().tolist()

        # Compute additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        self.log("test/specificity", specificity)
        self.log("test/tn", tn)
        self.log("test/fp", fp)
        self.log("test/fn", fn)
        self.log("test/tp", tp)

        logger.info(
            "Test confusion matrix: TN=%d, FP=%d, FN=%d, TP=%d",
            tn, fp, fn, tp
        )

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for backbone and classifier
        backbone_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "fc" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        # Different learning rates
        param_groups = [
            {"params": backbone_params, "lr": self.hparams.learning_rate * 0.1},
            {"params": classifier_params, "lr": self.hparams.learning_rate},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.hparams.weight_decay,
        )

        # Scheduler
        if self.hparams.scheduler == "none":
            return optimizer

        max_epochs = self.trainer.max_epochs if self.trainer else 50

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs - self.hparams.warmup_epochs,
                eta_min=1e-7,
            )
        elif self.hparams.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max_epochs // 3,
                gamma=0.1,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

        # Add warmup
        if self.hparams.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.hparams.warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final FC layer.

        Useful for visualization (t-SNE, etc.) and analysis.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Features (B, 2048).
        """
        # Get all layers except final FC
        features = x
        for name, module in self.backbone.named_children():
            if name == "fc":
                break
            features = module(features)

        features = torch.flatten(features, 1)
        return features


# =============================================================================
# Factory Function
# =============================================================================


def create_classifier(
    config: Dict[str, Any],
    num_classes: int = 2,
) -> ResNet50Classifier:
    """
    Create a classifier from configuration.

    Args:
        config: Configuration dictionary.
        num_classes: Number of output classes.

    Returns:
        ResNet50Classifier instance.
    """
    return ResNet50Classifier(
        num_classes=num_classes,
        pretrained=config.get("pretrained", True),
        freeze_backbone=config.get("freeze_backbone", False),
        freeze_layers=config.get("freeze_layers", 0),
        dropout=config.get("dropout", 0.5),
        learning_rate=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
        label_smoothing=config.get("label_smoothing", 0.0),
        scheduler=config.get("scheduler", "cosine"),
        warmup_epochs=config.get("warmup_epochs", 5),
        class_weights=config.get("class_weights", None),
    )
