"""
Tier 1 features — computed from the tiles alone, without the deployed model (DR-002, DR-004).

Tier 1a: fixed pixel statistics, no model at all (23 numbers per tile):
    colour      mean and SD of R, G, B; of CIELAB L*, a*, b*; of the H, E, D stain channels
                (colour deconvolution, Ruifrok & Johnston)                       18
    texture     grey-level mean and SD, log(1 + variance of the Laplacian) (sharpness),
                grey-level entropy (bits), tissue fraction (HSV saturation > 0.07)  5
Tier 1b: the pooled features of a frozen, open ImageNet encoder, never fine-tuned (DR-005):
    ``resnet50``     torchvision IMAGENET1K_V2, 2,048 features (current default)
    ``densenet121``  torchvision IMAGENET1K_V1, 1,024 features
    Open question (ISS-043): the deployed classifier is also a ResNet-50 started from IMAGENET1K_V2,
    so the default 1b encoder equals the classifier at initialisation; DenseNet-121 removes that
    shared starting point. The choice is Abhi's.

Both are computed on the tiles as they arrive (before any stain normalisation), so they do not
depend on which condition E1–E5 is being monitored.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from skimage.color import rgb2hed, rgb2lab
from torch import nn
from torchvision import models

TIER1A_NAMES = (
    [f"{c}_{s}" for c in ("r", "g", "b") for s in ("mean", "sd")]
    + [f"lab_{c}_{s}" for c in ("l", "a", "b") for s in ("mean", "sd")]
    + [f"hed_{c}_{s}" for c in ("h", "e", "d") for s in ("mean", "sd")]
    + ["grey_mean", "grey_sd", "log_laplacian_var", "grey_entropy", "tissue_fraction"]
)
COLOUR_FEATURES = TIER1A_NAMES[:18]
TISSUE_SATURATION = 0.07 * 255


def tier1a(img: np.ndarray) -> np.ndarray:
    """The 23 Tier 1a statistics of one RGB uint8 tile, in the order of ``TIER1A_NAMES``."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an RGB uint8 image (H, W, 3)")
    rgb = img.reshape(-1, 3).astype(np.float64)
    lab = rgb2lab(img).reshape(-1, 3)
    hed = rgb2hed(img).reshape(-1, 3)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hist = np.bincount(grey.ravel(), minlength=256) / grey.size
    nz = hist[hist > 0]
    sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    stats = []
    for arr in (rgb, lab, hed):
        for k in range(3):
            stats += [arr[:, k].mean(), arr[:, k].std()]
    stats += [grey.mean(), grey.std(),
              np.log1p(cv2.Laplacian(grey, cv2.CV_64F).var()),
              float(-(nz * np.log2(nz)).sum()),
              float((sat > TISSUE_SATURATION).mean())]
    return np.asarray(stats, dtype=np.float64)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


TIER1B_ARCHS = {"resnet50": 2048, "densenet121": 1024}


class Tier1bEncoder(nn.Module):
    """Frozen ImageNet encoder returning pooled features per tile (see ``TIER1B_ARCHS``)."""

    def __init__(self, pretrained: bool = True, arch: str = "resnet50"):
        super().__init__()
        if arch == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            net.fc = nn.Identity()
        elif arch == "densenet121":
            net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
            net.classifier = nn.Identity()
        else:
            raise ValueError(f"unknown Tier 1b encoder {arch!r}; choose from {sorted(TIER1B_ARCHS)}")
        self.arch, self.dim = arch, TIER1B_ARCHS[arch]
        self.net = net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "Tier1bEncoder":
        return super().train(False)          # always frozen, always in evaluation mode

    @torch.no_grad()
    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """``tiles``: uint8 (N, H, W, 3) or float (N, 3, H, W) already ImageNet-normalised."""
        if tiles.dtype == torch.uint8:
            dev = next(self.net.parameters()).device
            x = tiles.to(dev).permute(0, 3, 1, 2).float().div_(255)
            tiles = (x - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
        return self.net(tiles)
