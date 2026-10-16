"""Pass 1: plan and label candidate tiles for every used slide in the metadata (any dataset).

Writes <out_dir>/plan/candidates.csv.gz (one row per candidate tile) and slides.csv (one row
per slide with counts and the tissue threshold used). Optionally renders previews.

Run in the `wsitile` env:
    PYTHONPATH=src python scripts/tiling_plan.py --config configs/tiling_<dataset>.yaml [--slides id ...] [--preview]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from stain_benchmark.data.metadata import validate_metadata
from stain_benchmark.data.tiling import NORMAL, TUMOUR, TilingConfig, plan_slide, read_overview


def _plan(args):
    root, row, cfg, ann_dir = args
    try:
        return plan_slide(root, row, cfg, ann_dir)
    except Exception as e:  # report and continue; a bad slide must not stop the run
        return None, {"slide_id": row["slide_id"], "error": f"{type(e).__name__}: {e}"}


def preview(root: Path, row: pd.Series, df: pd.DataFrame, cfg: TilingConfig, out: Path) -> None:
    import openslide
    from PIL import Image, ImageDraw

    s = openslide.OpenSlide(str(root / row["file"]))
    rgb, ds, mpp0 = read_overview(s, cfg.overview_um)
    s.close()
    im = Image.fromarray(rgb); d = ImageDraw.Draw(im, "RGBA")
    step = cfg.tile_um / mpp0 / ds
    for x0, y0, lab in df[["x0", "y0", "label"]].itertuples(index=False):
        x, y = x0 / ds, y0 / ds
        color = (0, 220, 0, 110) if lab == TUMOUR else (0, 90, 255, 70)
        d.rectangle([x, y, x + step, y + step], fill=color)
    im.thumbnail((1200, 1200)); im.save(out / f"preview_{row['slide_id']}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--slides", nargs="*", help="Only these slide ids (default: all eligible)")
    ap.add_argument("--preview", action="store_true", help="Render label overlays")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    conf = yaml.safe_load(Path(args.config).read_text())
    cfg = TilingConfig.from_yaml(args.config)
    root = Path(conf["root"]); out = Path(conf["out_dir"]) / "plan"
    ann_dir = conf.get("annotations_dir")
    out.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(conf["metadata"], dtype={"center": str})
    validate_metadata(table)
    table = table[table["role"] != "excluded"]
    if args.slides:
        table = table[table["slide_id"].isin(args.slides)]
    print(f"Planning {len(table)} slides with {args.workers} workers")

    frames, infos = [], []
    jobs = [(root, row, cfg, ann_dir) for _, row in table.iterrows()]
    with ProcessPoolExecutor(args.workers) as pool:
        bar = tqdm(zip(pool.map(_plan, jobs), table.iterrows()), total=len(jobs), unit="slide",
                   desc="planning", mininterval=2, ascii=True, ncols=100)
        for (df, info), (_, row) in bar:
            infos.append(info)
            if df is not None:
                frames.append(df)
                if args.preview:
                    preview(root, row, df, cfg, out)
            if info.get("error"):
                tqdm.write(f"  {info['slide_id']}: {info['error']}")

    tag = "_subset" if args.slides else ""
    pd.DataFrame(infos).to_csv(out / f"slides{tag}.csv", index=False)
    if frames:
        cand = pd.concat(frames, ignore_index=True)
        cand.to_csv(out / f"candidates{tag}.csv.gz", index=False)
        summary = cand.groupby(["center", "label"]).size().unstack(fill_value=0).rename(columns={NORMAL: "normal", TUMOUR: "tumour"})
        print("\nCandidate tiles per center:\n" + summary.to_string())


if __name__ == "__main__":
    main()
