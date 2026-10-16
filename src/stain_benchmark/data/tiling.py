"""Dataset-agnostic WSI tile planning, extraction, QC scores and sampling.

Slides come from the common metadata schema (stain_benchmark.data.metadata). Tile labels
depend only on the slide's role:
    'annotated'   -> from outline polygons (>= tumour_min_frac inside -> tumour; no outline
                     within normal_margin_um -> normal; otherwise discarded)
    'slide_label' -> every tissue tile gets the slide's label

Pass 1 (plan): from a low-resolution overview of each slide, find tissue, rasterize any
outlines, lay a grid of fixed-size tiles in physical units, and label each candidate.
No full-resolution pixels are read.

Pass 2 (extract): read the chosen tiles at the target resolution, resampled so every tile
covers the same physical area regardless of scanner.

Runs in the `wsitile` env (OpenSlide); needs no PyTorch.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TUMOUR, NORMAL = 1, 0


@dataclass(frozen=True)
class TilingConfig:
    target_mpp: float = 0.5
    tile_px: int = 256
    overview_um: float = 8.0
    sat_otsu_floor: int = 15
    sat_otsu_ceil: int = 60
    val_min: int = 80
    min_tissue_frac: float = 0.70
    tumour_min_frac: float = 0.50
    normal_margin_um: float = 128.0

    @property
    def tile_um(self) -> float:
        return self.target_mpp * self.tile_px

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TilingConfig":
        import yaml

        c = yaml.safe_load(Path(path).read_text())
        return cls(
            target_mpp=c["target_mpp"], tile_px=c["tile_px"], overview_um=c["overview_um"],
            sat_otsu_floor=c["tissue"]["sat_otsu_floor"], sat_otsu_ceil=c["tissue"]["sat_otsu_ceil"],
            val_min=c["tissue"]["val_min"], min_tissue_frac=c["tissue"]["min_tissue_frac"],
            tumour_min_frac=c["labels"]["tumour_min_frac"],
            normal_margin_um=c["labels"]["normal_margin_um"],
        )


# --------------------------------------------------------------------------- overview + masks
def read_overview(slide, overview_um: float) -> tuple[np.ndarray, float, float]:
    """RGB overview at the pyramid level nearest overview_um. Returns (rgb, level_downsample, mpp0)."""
    mpp0 = float(slide.properties["openslide.mpp-x"])
    level = slide.get_best_level_for_downsample(overview_um / mpp0)
    ds = float(slide.level_downsamples[level])
    w, h = slide.level_dimensions[level]
    rgba = np.asarray(slide.read_region((0, 0), level, (w, h)))
    rgb = rgba[..., :3].copy()
    rgb[rgba[..., 3] == 0] = 0  # transparent (unscanned) -> treat as black background
    return rgb, ds, mpp0


def tissue_mask(rgb: np.ndarray, cfg: TilingConfig) -> tuple[np.ndarray, int]:
    """Stained, non-black pixels. Saturation threshold = per-slide Otsu, clamped."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    scanned = v >= cfg.val_min
    if scanned.sum() < 100:
        return np.zeros(s.shape, bool), cfg.sat_otsu_floor
    thr, _ = cv2.threshold(s[scanned].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(np.clip(thr, cfg.sat_otsu_floor, cfg.sat_otsu_ceil))
    mask = (s >= thr) & scanned
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask.astype(bool), thr


def parse_annotations(xml_path: str | Path) -> list[tuple[str, np.ndarray]]:
    """[(group, polygon level-0 xy float array)] from an ASAP XML file."""
    out = []
    for a in ET.parse(xml_path).getroot().iter("Annotation"):
        pts = np.array([[float(c.get("X")), float(c.get("Y"))] for c in a.iter("Coordinate")])
        if len(pts) >= 3:
            out.append((a.get("PartOfGroup") or "", pts))
    return out


def rasterize(polys: list[np.ndarray], shape: tuple[int, int], ds: float) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if polys:
        cv2.fillPoly(m, [np.round(p / ds).astype(np.int32) for p in polys], 1)
    return m.astype(bool)


def _box_fraction(sat: np.ndarray, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """Mean of a boolean mask over boxes [x0,x1) x [y0,y1) using its summed-area table."""
    area = np.maximum((x1 - x0) * (y1 - y0), 1)
    tot = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
    return tot / area


def summed_area(mask: np.ndarray) -> np.ndarray:
    sat = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int64)
    sat[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    return sat


# --------------------------------------------------------------------------- pass 1: plan
def plan_tiles(
    tissue: np.ndarray,
    tumour: np.ndarray,
    exclusion: np.ndarray,
    ds: float,
    mpp0: float,
    role: str,
    cfg: TilingConfig,
    slide_label: int = NORMAL,
) -> pd.DataFrame:
    """Label a grid of tiles from overview masks. Coordinates returned in level-0 pixels."""
    step0 = cfg.tile_um / mpp0                      # tile size in level-0 pixels
    H, W = tissue.shape
    nx, ny = int(W * ds // step0), int(H * ds // step0)
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    x0 = (gx.ravel() * step0)
    y0 = (gy.ravel() * step0)
    bx0 = np.floor(x0 / ds).astype(int); by0 = np.floor(y0 / ds).astype(int)
    bx1 = np.minimum(np.ceil((x0 + step0) / ds).astype(int), W)
    by1 = np.minimum(np.ceil((y0 + step0) / ds).astype(int), H)

    tissue_f = _box_fraction(summed_area(tissue), bx0, by0, bx1, by1)
    keep = tissue_f >= cfg.min_tissue_frac
    df = pd.DataFrame({
        "x0": np.round(x0).astype(int), "y0": np.round(y0).astype(int),
        "tissue_frac": tissue_f,
    })[keep]
    b = (bx0[keep], by0[keep], bx1[keep], by1[keep])

    if role == "slide_label":
        df["tumour_frac"] = float(slide_label == TUMOUR)
        df["label"] = int(slide_label)
        return df.reset_index(drop=True)
    if role != "annotated":
        raise ValueError(f"unknown role {role!r}")

    df["tumour_frac"] = _box_fraction(summed_area(tumour), *b)
    excl = _box_fraction(summed_area(exclusion), *b) > 0
    margin_px = int(np.ceil(cfg.normal_margin_um / (mpp0 * ds)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    near = cv2.dilate(tumour.astype(np.uint8), k).astype(bool)
    near_f = _box_fraction(summed_area(near), *b)

    label = np.full(len(df), -1)
    label[df["tumour_frac"].to_numpy() >= cfg.tumour_min_frac] = TUMOUR
    label[near_f == 0] = NORMAL
    label[excl] = -1
    df["label"] = label
    return df[df["label"] >= 0].reset_index(drop=True)


def plan_slide(root: str | Path, row: pd.Series, cfg: TilingConfig,
               annotations_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Plan one slide from a metadata row (needs slide_id, file, role, label)."""
    import openslide

    s = openslide.OpenSlide(str(Path(root) / row["file"]))
    try:
        rgb, ds, mpp0 = read_overview(s, cfg.overview_um)
    finally:
        s.close()
    tissue, thr = tissue_mask(rgb, cfg)
    tumour = np.zeros(tissue.shape, bool); exclusion = np.zeros(tissue.shape, bool)
    if row["role"] == "annotated":
        anns = parse_annotations(Path(annotations_dir) / f"{row['slide_id']}.xml")
        tumour = rasterize([p for g, p in anns if g.lower() == "tumor"], tissue.shape, ds)
        exclusion = rasterize([p for g, p in anns if g.lower() == "exclusion"], tissue.shape, ds)
    df = plan_tiles(tissue, tumour, exclusion, ds, mpp0, row["role"], cfg, slide_label=int(row["label"]))
    for c in ("slide_id", "patient_id", "center", "role"):
        df[c] = row[c]
    info = {"slide_id": row["slide_id"], "mpp0": mpp0, "overview_ds": ds, "sat_threshold": thr,
            "tissue_frac_slide": float(tissue.mean()), "tumour_frac_slide": float(tumour.mean()),
            "n_tumour": int((df.label == TUMOUR).sum()), "n_normal": int((df.label == NORMAL).sum())}
    return df, info


# --------------------------------------------------------------------------- pass 2: extract
def read_tile(slide, x0: int, y0: int, cfg: TilingConfig):
    """Tile at exactly cfg.target_mpp: read the covering region at the best level, resize."""
    from PIL import Image

    mpp0 = float(slide.properties["openslide.mpp-x"])
    level = slide.get_best_level_for_downsample(cfg.target_mpp / mpp0)
    ds = float(slide.level_downsamples[level])
    size = int(round(cfg.tile_um / (mpp0 * ds)))
    region = slide.read_region((int(x0), int(y0)), level, (size, size)).convert("RGB")
    if size != cfg.tile_px:
        region = region.resize((cfg.tile_px, cfg.tile_px), Image.LANCZOS)
    return region


# --------------------------------------------------------------------------- sampling
def _equal_quota(sizes: pd.Series, total: int) -> pd.Series:
    """Split `total` across groups as evenly as possible, capped by each group's size;
    any leftover from small groups is redistributed to the others."""
    quota = pd.Series(0, index=sizes.index)
    remaining, open_ = total, sizes.index.tolist()
    while remaining > 0 and open_:
        share = max(remaining // len(open_), 1)
        for g in list(open_):
            take = min(share, sizes[g] - quota[g], remaining)
            quota[g] += take; remaining -= take
            if quota[g] >= sizes[g]:
                open_.remove(g)
            if remaining == 0:
                break
    return quota


def sample_tiles(cand: pd.DataFrame, tumour_cap: int, normal_ratio: float, seed: int) -> pd.DataFrame:
    """Tumour: up to `tumour_cap` per slide, random within slide.
    Normal: per center, round(normal_ratio * tumour count) tiles split equally across every
    slide in that center that has normal candidates, random within slide."""
    rng = np.random.default_rng(seed)
    picks = []
    for center, cc in cand.groupby("center", sort=True):
        tum = cc[cc["label"] == TUMOUR]
        n_tum = 0
        for _, s in tum.groupby("slide_id", sort=True):
            k = min(tumour_cap, len(s))
            picks.append(s.iloc[np.sort(rng.choice(len(s), k, replace=False))]); n_tum += k
        nor = cc[cc["label"] == NORMAL]
        sizes = nor.groupby("slide_id", sort=True).size()
        quota = _equal_quota(sizes, int(round(normal_ratio * n_tum)))
        for sid, s in nor.groupby("slide_id", sort=True):
            k = int(quota[sid])
            if k:
                picks.append(s.iloc[np.sort(rng.choice(len(s), k, replace=False))])
    out = pd.concat(picks, ignore_index=True)
    out["tile_id"] = out["slide_id"] + "_" + out["x0"].astype(str) + "_" + out["y0"].astype(str)
    return out


# --------------------------------------------------------------------------- QC metrics
def qc_metrics(tile: np.ndarray, cfg: TilingConfig) -> dict:
    """Per-tile quality scores (computed, not applied; thresholds are chosen after review)."""
    gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256) / gray.size
    nz = hist[hist > 0]
    hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
    tissue = (hsv[..., 1] >= cfg.sat_otsu_floor) & (hsv[..., 2] >= cfg.val_min)
    mean = tile.reshape(-1, 3).mean(0); std = tile.reshape(-1, 3).std(0)
    return {
        "qc_blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "qc_entropy": float(-(nz * np.log2(nz)).sum()),
        "qc_tissue_frac": float(tissue.mean()),
        "rgb_mean_r": float(mean[0]), "rgb_mean_g": float(mean[1]), "rgb_mean_b": float(mean[2]),
        "rgb_std_r": float(std[0]), "rgb_std_g": float(std[1]), "rgb_std_b": float(std[2]),
    }
