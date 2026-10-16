"""Contact sheets for choosing QC thresholds and seeing per-center stain differences.

For each QC score, shows the lowest-scoring tiles with their scores, so a person can
judge where unusable starts. Also a per-center, per-class sample sheet.

Run in the `wsitile` env:
    PYTHONPATH=src python scripts/tiling_qc_review.py --config configs/tiling_<dataset>.yaml --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from PIL import Image, ImageDraw

THUMB = 160


def sheet(df: pd.DataFrame, root: Path, caption, cols: int = 10) -> Image.Image:
    rows = int(np.ceil(len(df) / cols))
    im = Image.new("RGB", (cols * THUMB, rows * (THUMB + 16)), "white")
    d = ImageDraw.Draw(im)
    for i, r in enumerate(tqdm(df.itertuples(index=False), total=len(df), unit="tile", desc="sheet", ascii=True, ncols=100, leave=False)):
        x, y = (i % cols) * THUMB, (i // cols) * (THUMB + 16)
        im.paste(Image.open(root / r.path).convert("RGB").resize((THUMB, THUMB)), (x, y))
        d.text((x + 3, y + THUMB + 1), caption(r), fill="black")
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    root = Path(yaml.safe_load(Path(args.config).read_text())["out_dir"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(root / "manifest.csv")

    lab = {0: "N", 1: "T"}
    for col, name in (("qc_blur", "blur"), ("qc_entropy", "entropy"), ("qc_tissue_frac", "tissue")):
        worst = man.nsmallest(args.n, col)
        sheet(worst, root, lambda r, c=col: f"c{r.center} {lab[r.label]} {getattr(r, c):.2f}"[:24]).save(out / f"qc_lowest_{name}.png")
        q = man[col].quantile([0.01, 0.05, 0.25, 0.5]).round(2).to_dict()
        print(f"{name:8} quantiles 1/5/25/50%: {q}")

    rng = np.random.default_rng(0)
    picks = [g.iloc[rng.choice(len(g), min(10, len(g)), replace=False)]
             for _, g in man.groupby(["center", "label"], sort=True)]
    sheet(pd.concat(picks), root, lambda r: f"center {r.center} {'tumour' if r.label else 'normal'}").save(out / "samples_by_center.png")
    print("sheets written to", out)


if __name__ == "__main__":
    main()
