"""
Graded synthetic shifts — the controlled arm of the monitor experiment (DR-007).

Each family imitates a real acquisition or preparation problem; ``strength`` grades it from
0 (unchanged) upward. Families follow the artifact list of an FDA-cleared slide-QC device
(out of focus, striping, missing/clipped tissue) plus image compression, with stain colour as
the expected-harmless control:

    blur         out-of-focus: Gaussian blur, sigma = strength (pixels)          primary harmful
    striping     scanner striping: alternate 16-px bands darkened by `strength`  secondary
    clipped      missing/clipped tissue: a `strength` fraction of the tile blanked to glass
    jpeg         compression: JPEG quality = 100 - strength
    stain        stain intensity/colour: haematoxylin and eosin amounts scaled by
                 1 ± strength (random direction PER TILE), in HED space — pre-registered;
                 across a batch the directions cancel, so it is closer to stain noise (ISS-044)
    stain_batch  consistent over-staining: haematoxylin and eosin both scaled by 1 + strength
                 for every tile — how a laboratory batch drifts (post hoc, ISS-044)

Doses are matched ACROSS families by how much the image changes, not by any monitor statistic
(matching on a Tier 1 statistic would make the harmful-versus-harmless test circular). The
primary yardstick is the mean per-pixel CIELAB colour difference ΔE (DR-015): every family can
reach the same ΔE levels. SSIM was the first choice but cannot be used for matching: stain
changes barely move it (mean SSIM ≥ 0.94 even at maximum stain strength on real tiles, while
blur reaches 0.22), so it is reported alongside as a secondary description.
``calibrate_strength`` finds, for each family, the strength that brings the mean of the
chosen measure to a target value on a fixed sample of training-split tiles.

Every shift is deterministic given (tile, family, strength, seed).
"""

from __future__ import annotations

import zlib

import cv2
import numpy as np
from skimage.color import hed2rgb, rgb2hed, rgb2lab
from skimage.metrics import structural_similarity

FAMILIES = ("blur", "striping", "clipped", "jpeg", "stain")      # pre-registered dose table
EXTRA_FAMILIES = ("stain_batch",)                                 # post hoc (ISS-044)
GLASS = 240                 # RGB value used for blanked (missing) tissue, close to slide background
STRIPE_PX = 16              # band height for striping


def _rng(seed: int, key: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(key.encode())])


def apply_shift(img: np.ndarray, family: str, strength: float, seed: int = 0, key: str = "") -> np.ndarray:
    """Return a shifted copy of an RGB uint8 tile. ``key`` (e.g. the tile id) makes per-tile
    random choices (stripe phase, clipped edge, stain direction) reproducible."""
    if family not in FAMILIES + EXTRA_FAMILIES:
        raise ValueError(f"unknown shift family {family!r}; choose from {FAMILIES + EXTRA_FAMILIES}")
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("expected an RGB uint8 image (H, W, 3)")
    if strength <= 0:
        return img.copy()
    rng = _rng(seed, f"{family}|{key}")
    h, w = img.shape[:2]

    if family == "blur":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=float(strength), borderType=cv2.BORDER_REFLECT)

    if family == "striping":
        phase = int(rng.integers(0, 2))
        bands = ((np.arange(h) // STRIPE_PX) + phase) % 2 == 1
        out = img.astype(np.float32)
        out[bands] *= 1.0 - min(float(strength), 1.0)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    if family == "clipped":
        frac = min(float(strength), 1.0)
        out = img.copy()
        edge = int(rng.integers(0, 4))
        n_rows, n_cols = int(round(frac * h)), int(round(frac * w))
        if edge == 0:
            out[:n_rows] = GLASS
        elif edge == 1:
            out[h - n_rows:] = GLASS
        elif edge == 2:
            out[:, :n_cols] = GLASS
        else:
            out[:, w - n_cols:] = GLASS
        return out

    if family == "jpeg":
        quality = int(np.clip(round(100 - strength), 1, 100))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # stain: scale haematoxylin and eosin amounts, each by 1 + strength * u, u in [-1, 1] per tile;
    # stain_batch: the same direction for every tile (u = (1, 1), over-staining)
    u = np.ones(2) if family == "stain_batch" else rng.uniform(-1.0, 1.0, size=2)
    hed = rgb2hed(img)
    hed[..., 0] *= max(1.0 + strength * u[0], 0.0)
    hed[..., 1] *= max(1.0 + strength * u[1], 0.0)
    return np.clip(np.rint(hed2rgb(hed) * 255), 0, 255).astype(np.uint8)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity of two RGB uint8 tiles (1 = identical)."""
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-pixel CIELAB colour difference (ΔE 1976) between two RGB uint8 tiles (0 = identical)."""
    return float(np.linalg.norm(rgb2lab(a) - rgb2lab(b), axis=2).mean())


MEASURES = {"ssim": ssim, "delta_e": delta_e}
INCREASING = {"ssim": False, "delta_e": True}       # does the measure grow with shift strength?


def mean_measure(tiles: list[np.ndarray], family: str, strength: float, measure: str = "delta_e",
                 seed: int = 0, keys: list[str] | None = None) -> float:
    """Mean of ``measure`` between each tile and its shifted copy."""
    keys = keys or [str(i) for i in range(len(tiles))]
    f = MEASURES[measure]
    return float(np.mean([f(t, apply_shift(t, family, strength, seed, k)) for t, k in zip(tiles, keys)]))


def mean_ssim(tiles: list[np.ndarray], family: str, strength: float, seed: int = 0,
              keys: list[str] | None = None) -> float:
    return mean_measure(tiles, family, strength, "ssim", seed, keys)


# Search ranges per family (strength units as above). Upper ends are physical limits.
STRENGTH_RANGE = {"blur": (0.0, 12.0), "striping": (0.0, 0.9), "clipped": (0.0, 0.9),
                  "jpeg": (0.0, 99.0), "stain": (0.0, 0.95), "stain_batch": (0.0, 3.0)}


def calibrate_strength(tiles: list[np.ndarray], family: str, target: float, measure: str = "delta_e",
                       seed: int = 0, keys: list[str] | None = None, iters: int = 20) -> tuple[float, float]:
    """Strength at which the mean ``measure`` over ``tiles`` reaches ``target`` (bisection).

    Returns (strength, achieved mean). If even the maximum strength cannot reach the target,
    returns the maximum and its value, so the caller can see the target is unreachable.
    """
    up = INCREASING[measure]
    reached = (lambda v: v >= target) if up else (lambda v: v <= target)
    lo, hi = STRENGTH_RANGE[family]
    at_hi = mean_measure(tiles, family, hi, measure, seed, keys)
    if not reached(at_hi):
        return hi, at_hi
    s_lo, s_hi = lo, hi
    for _ in range(iters):
        mid = (s_lo + s_hi) / 2
        if reached(mean_measure(tiles, family, mid, measure, seed, keys)):
            s_hi = mid
        else:
            s_lo = mid
    if family == "jpeg":                     # quality is an integer: keep the closer neighbour
        cands = sorted({float(np.floor(s_hi)), float(np.ceil(s_hi))})
        vals = [mean_measure(tiles, family, c, measure, seed, keys) for c in cands]
        best = int(np.argmin([abs(v - target) for v in vals]))
        return cands[best], vals[best]
    return s_hi, mean_measure(tiles, family, s_hi, measure, seed, keys)
