#!/usr/bin/env python3
"""Pathologist review (P4 part 2), step 1 of 2 — run in the OpenSlide environment (wsitile).

For each patient's outlined slide: a low-power view of the outlined region with the outline polygons, and, for every
selected tile, a context crop (3 × 3 tiles around it, 1 µm/px). Writes one .npz per slide; step 2
(`scripts/review_sheets.py`, main environment) draws the review PDF. Tile selection and scores come from the
selection CSV written by step 2's `--select` mode.

    PYTHONPATH=src python scripts/review_extract.py --selection <dir>/selection.csv --out <dir>
"""

import argparse
from pathlib import Path

import numpy as np
import openslide
import pandas as pd

from stain_benchmark.data.tiling import parse_annotations

ROOT = Path("/media/abhi/camelyon17/raw")
CONTEXT_UM, CONTEXT_MPP, OVERVIEW_MPP = 384.0, 1.0, 8.0
CLUSTER_GAP_UM, CLUSTER_MARGIN_UM, ZOOM_MAX_PX = 1000.0, 200.0, 2000


def clusters(xy: np.ndarray, gap: float) -> np.ndarray:
    """Single-linkage cluster labels: tiles closer than ``gap`` (level-0 px) share a cluster."""
    lab = -np.ones(len(xy), int)
    for i in range(len(xy)):
        if lab[i] >= 0:
            continue
        lab[i] = lab.max() + 1
        stack = [i]
        while stack:
            j = stack.pop()
            near = np.flatnonzero((np.abs(xy - xy[j]).max(1) < gap) & (lab < 0))
            lab[near] = lab[i]
            stack += list(near)
    return lab


def read_scaled(slide, x0: float, y0: float, w0: float, h0: float, mpp_out: float) -> np.ndarray:
    """Region [x0, x0+w0) × [y0, y0+h0) in level-0 pixels, resampled to ``mpp_out`` µm/px."""
    from PIL import Image
    mpp0 = float(slide.properties["openslide.mpp-x"])
    level = slide.get_best_level_for_downsample(mpp_out / mpp0)
    ds = slide.level_downsamples[level]
    w, h = int(np.ceil(w0 / ds)), int(np.ceil(h0 / ds))
    img = slide.read_region((int(x0), int(y0)), level, (w, h)).convert("RGB")
    return np.asarray(img.resize((max(1, int(w0 * mpp0 / mpp_out)), max(1, int(h0 * mpp0 / mpp_out))), Image.LANCZOS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = pd.read_csv(args.selection)
    out = Path(args.out)
    scores = pd.read_csv(out / "all_scores.csv")
    for slide_id, rows in sel.groupby("slide_id"):
        slide = openslide.OpenSlide(str(ROOT / "images" / f"{slide_id}.tif"))
        mpp0 = float(slide.properties["openslide.mpp-x"])
        polys = [p for g, p in parse_annotations(ROOT / "annotations" / f"{slide_id}.xml")]
        allxy = np.vstack(polys)
        margin = 300 / mpp0
        bx0, by0 = allxy.min(0) - margin
        bx1, by1 = allxy.max(0) + margin
        overview = read_scaled(slide, bx0, by0, bx1 - bx0, by1 - by0, OVERVIEW_MPP)
        tile0 = 128.0 / mpp0                                    # tile edge in level-0 px (256 px at 0.5 µm/px)
        ctx0 = CONTEXT_UM / mpp0
        sp = scores[scores.slide_id == slide_id]
        xy = sp[["x0", "y0"]].to_numpy(float)
        lab = clusters(xy, CLUSTER_GAP_UM / mpp0)
        zooms = []
        for k in range(lab.max() + 1):
            c = xy[lab == k]
            m0 = CLUSTER_MARGIN_UM / mpp0
            zx0, zy0 = c.min(0) - m0
            zx1, zy1 = c.max(0) + tile0 + m0
            mpp_z = max(2.0, max(zx1 - zx0, zy1 - zy0) * mpp0 / ZOOM_MAX_PX)
            zooms.append({"image": read_scaled(slide, zx0, zy0, zx1 - zx0, zy1 - zy0, mpp_z),
                          "origin": np.array([zx0, zy0]), "scale": mpp_z / mpp0, "mpp": mpp_z,
                          "n_tiles": int((lab == k).sum())})
        crops = [read_scaled(slide, x + tile0 / 2 - ctx0 / 2, y + tile0 / 2 - ctx0 / 2, ctx0, ctx0, CONTEXT_MPP)
                 for x, y in zip(rows.x0, rows.y0)]
        np.savez_compressed(out / f"{slide_id}.npz", overview=overview, origin=np.array([bx0, by0]),
                            overview_scale=OVERVIEW_MPP / mpp0, tile0=tile0,
                            polys=np.array(polys, dtype=object), crops=np.stack(crops),
                            tile_ids=rows.tile_id.to_numpy(), zooms=np.array(zooms, dtype=object))
        print(f"{slide_id}: overview {overview.shape}, {len(zooms)} tumour clusters, {len(crops)} context crops", flush=True)


if __name__ == "__main__":
    main()
