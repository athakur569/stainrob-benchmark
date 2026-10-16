"""Pass 2: sample tiles from the plan, extract them, record QC scores (any dataset).

Writes <out_dir>/tiles/<center>/<slide_id>/<tile_id>.png, <out_dir>/manifest.csv (with
qc_pass from the config's qc section) and <out_dir>/split_input.csv: one row per slide with
QC-passing tiles (slide_id, patient_id, center, label, tumour_tiles, tiles) for create_splits.py.
Run in the `wsitile` env:
    PYTHONPATH=src python scripts/tiling_extract.py --config configs/tiling_<dataset>.yaml [--reuse-tiles]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from stain_benchmark.data.tiling import TilingConfig, qc_metrics, read_tile, sample_tiles


def _extract_slide(args):
    root, out_dir, file, rows, cfg, reuse = args
    import openslide
    from PIL import Image

    s = openslide.OpenSlide(str(root / file))
    recs = []
    try:
        for r in rows:
            rel = Path("tiles") / str(r["center"]) / r["slide_id"] / f"{r['tile_id']}.png"
            dest = out_dir / rel
            if reuse and dest.exists():
                tile = Image.open(dest).convert("RGB")
            else:
                tile = read_tile(s, r["x0"], r["y0"], cfg)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tile.save(dest, compress_level=3)
            recs.append({**r, "path": str(rel), **qc_metrics(np.asarray(tile), cfg)})
    finally:
        s.close()
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--reuse-tiles", action="store_true", help="Reuse existing tile PNGs at the same path")
    args = ap.parse_args()
    conf = yaml.safe_load(Path(args.config).read_text())
    cfg = TilingConfig.from_yaml(args.config)
    root, out_dir = Path(conf["root"]), Path(conf["out_dir"])
    smp = conf["sampling"]

    cand = pd.read_csv(out_dir / "plan" / "candidates.csv.gz", dtype={"center": str})
    chosen = sample_tiles(cand, smp["tumour_cap_per_slide"], smp["normal_ratio"], smp["seed"])
    meta = pd.read_csv(conf["metadata"], dtype={"center": str}).set_index("slide_id")
    files = meta["file"]
    print("Selected tiles per center:\n" + chosen.groupby(["center", "label"]).size().unstack().to_string())

    jobs = [(root, out_dir, files[sid], g.to_dict("records"), cfg, args.reuse_tiles) for sid, g in chosen.groupby("slide_id")]
    recs = []
    bar = tqdm(total=len(chosen), unit="tile", desc="extracting", mininterval=2, ascii=True, dynamic_ncols=False, ncols=100)
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(_extract_slide, jobs):
            recs.extend(r)
            bar.update(len(r))
    bar.close()
    man = pd.DataFrame(recs).sort_values(["center", "slide_id", "tile_id"]).reset_index(drop=True)
    qc = conf["qc"]
    man["qc_pass"] = (man["qc_blur"] >= qc["blur_min"]) & (man["qc_entropy"] >= qc["entropy_min"])
    man.to_csv(out_dir / "manifest.csv", index=False)
    print(f"Wrote {len(man)} tiles ({int(man.qc_pass.sum())} pass QC); manifest: {out_dir / 'manifest.csv'}")

    # Slide-level table for scripts/create_splits.py (QC-passing tiles only)
    ok = man[man["qc_pass"]]
    slides = ok.groupby("slide_id").agg(patient_id=("patient_id", "first"), center=("center", "first"),
                                        tumour_tiles=("label", "sum"), tiles=("label", "size")).reset_index()
    slides["label"] = (slides["tumour_tiles"] > 0).astype(int)
    slides.to_csv(out_dir / "split_input.csv", index=False)
    print(f"Wrote {len(slides)} slides / {slides.patient_id.nunique()} patients -> {out_dir / 'split_input.csv'}")


if __name__ == "__main__":
    main()
