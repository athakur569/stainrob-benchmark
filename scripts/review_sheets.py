#!/usr/bin/env python3
"""Pathologist review (P4 part 2), step 2 of 2 — tile selection and the review PDF (main environment).

The question the sheets serve: inside these macrometastasis outlines, are the tiles the model misses tumour, or
non-tumour tissue the outline encloses? For each patient's outlined slide:
  * the 12 most confidently missed tumour tiles, 8 further misses drawn at random (seed 0; so the sheet is not only
    the extremes), and the 6 highest-scoring tumour tiles for contrast;
  * score = E1 tumour probability averaged over the seeds and folds in which the patient is a test patient.

    python scripts/review_sheets.py --select --runs <runs> --tiles <tiles> --out <dir>     # writes selection.csv
    (then, in the OpenSlide environment: scripts/review_extract.py --selection <dir>/selection.csv --out <dir>)
    python scripts/review_sheets.py --draw --tiles <tiles> --out <dir>                      # writes review.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PATIENTS = ("patient_042", "patient_052", "patient_092", "patient_096")
N_WORST, N_RANDOM, N_HITS = 12, 8, 6


def select(runs: Path, tiles: Path, out: Path) -> None:
    man = pd.read_csv(tiles / "manifest.csv").set_index("tile_id")
    preds = []
    for f in range(5):
        for role in ("test_in", "test_cross"):
            for s in (42, 99, 123):
                preds.append(pd.read_csv(runs / "E1" / f"fold{f}" / f"seed{s}" / f"predictions_{role}.csv"))
    p = pd.concat(preds)
    p = p[(p.label == 1) & p.patient_id.isin(PATIENTS)].groupby(["patient_id", "slide_id", "tile_id"]).p_tumour.mean()
    p = p.reset_index()
    p = p.join(man[["x0", "y0", "path", "tumour_frac", "center"]], on="tile_id")
    p.to_csv(out / "all_scores.csv", index=False)
    rows = []
    rng = np.random.default_rng(0)
    for pid, d in p.groupby("patient_id"):
        d = d.sort_values("p_tumour")
        worst = d.head(N_WORST)
        rest = d[(d.p_tumour < 0.5) & ~d.tile_id.isin(worst.tile_id)]
        rand = rest.iloc[np.sort(rng.choice(len(rest), min(N_RANDOM, len(rest)), replace=False))] if len(rest) else rest
        hits = d.tail(N_HITS)
        for grp, g in (("confident miss", worst), ("random miss", rand), ("highest-scoring hit", hits)):
            rows.append(g.assign(group=grp))
    sel = pd.concat(rows).reset_index(drop=True)
    sel["item"] = sel.groupby("patient_id").cumcount() + 1
    sel.to_csv(out / "selection.csv", index=False)
    print(f"selected {len(sel)} tiles from {sel.patient_id.nunique()} patients", flush=True)


def draw(tiles: Path, out: Path, png: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Polygon, Rectangle
    from PIL import Image

    sel = pd.read_csv(out / "selection.csv")
    scores = pd.read_csv(out / "all_scores.csv")
    pages = []
    if png:
        (out / "pages").mkdir(exist_ok=True)

    with PdfPages(out / "review.pdf") as pdf:
        def save(fig):
            pdf.savefig(fig)
            if png:
                fig.savefig(out / "pages" / f"page_{len(pages) + 1:02d}.png", dpi=80)
            pages.append(1)
            plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.06, 0.92, "CAMELYON17 rehearsal — pathologist review of the four most-missed tumour patients",
                 fontsize=15, weight="bold")
        fig.text(0.06, 0.86, "Question: inside these macrometastasis outlines, are the tiles the model misses tumour,\n"
                 "or non-tumour tissue that the outline encloses?", fontsize=12)
        fig.text(0.06, 0.80, "\n".join([
            "How to read the sheets",
            "• One section per patient: an overview of the slide area holding the outlines (cyan boxes = regions shown",
            "  zoomed next), then one zoomed page per region at about 2 µm/px: yellow = the dataset's tumour outline;",
            "  squares = the tiles we labelled tumour, red = model score below 0.5 (missed), green = 0.5 or above.",
            "  Numbers mark the tiles shown in detail.",
            "• Then the tiles themselves, two per row: the tile as the model saw it (256 px at 0.5 µm/px = 128 µm), and",
            "  a 384 µm context view (1 µm/px) with the tile boxed.",
            f"• Per patient: the {N_WORST} most confidently missed tiles, {N_RANDOM} further misses drawn at random (so the",
            f"  sheet is not only the extremes), and the {N_HITS} highest-scoring tiles for contrast.",
            "• p = tumour probability from the baseline model (E1), averaged over the seeds and folds in which the",
            "  patient is a test patient. cov = fraction of the tile inside the outline (tiles need ≥ 0.5 to be labelled tumour).",
            "",
            "Useful answers: per patient — mostly tumour / mostly enclosed non-tumour / mixed; and, if useful, a note",
            "on any numbered tile (e.g. \"042 #3: stroma only\"). Any tissue type, pattern or artefact worth naming helps.",
        ]), fontsize=10.5, va="top", family="sans-serif", linespacing=1.5)
        save(fig)

        for pid, s in sel.groupby("patient_id", sort=False):
            slide_id = s.slide_id.iloc[0]
            z = np.load(out / f"{slide_id}.npz", allow_pickle=True)
            ov, (ox, oy), sc, t0 = z["overview"], z["origin"], float(z["overview_scale"]), float(z["tile0"])
            crops = dict(zip(z["tile_ids"], z["crops"]))
            sp = scores[scores.slide_id == slide_id]
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.imshow(ov)
            for poly in z["polys"]:
                ax.add_patch(Polygon((poly - [ox, oy]) / sc, closed=True, fill=False, ec="gold", lw=1.4))
            for r in sp.itertuples():
                ax.add_patch(Rectangle(((r.x0 - ox) / sc, (r.y0 - oy) / sc), t0 / sc, t0 / sc, fill=False, lw=0.6,
                                       ec="red" if r.p_tumour < 0.5 else "limegreen"))
            for r in s.itertuples():
                ax.text((r.x0 - ox + t0 / 2) / sc, (r.y0 - oy + t0 / 2) / sc, str(r.item), color="white", fontsize=7,
                        ha="center", va="center", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.6, lw=0))
            for k, zm in enumerate(z["zooms"]):
                (zx, zy), w = zm["origin"], zm["image"].shape[1] * zm["scale"]
                h = zm["image"].shape[0] * zm["scale"]
                ax.add_patch(Rectangle(((zx - ox) / sc, (zy - oy) / sc), w / sc, h / sc, fill=False, ec="cyan", lw=1.5))
                ax.text((zx - ox) / sc, (zy - oy) / sc, chr(65 + k), color="cyan", fontsize=12, weight="bold", va="bottom")
            ax.set_axis_off()
            miss = (sp.p_tumour < 0.5).mean()
            ax.set_title(f"{pid.replace('_', ' ')} · {slide_id.split('_', 2)[2].replace('_', ' ')} · hospital "
                         f"{int(sp.center.iloc[0])} · macrometastasis · {len(sp)} tumour tiles, {miss:.0%} missed\n"
                         "yellow: dataset outline · red: missed (p < 0.5) · green: detected · cyan: zoomed regions", fontsize=11)
            save(fig)
            for k, zm in enumerate(z["zooms"]):
                im, (zx, zy), zs = zm["image"], zm["origin"], float(zm["scale"])
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.imshow(im)
                for poly in z["polys"]:
                    ax.add_patch(Polygon((poly - [zx, zy]) / zs, closed=True, fill=False, ec="gold", lw=1.6))
                for r in sp.itertuples():
                    ax.add_patch(Rectangle(((r.x0 - zx) / zs, (r.y0 - zy) / zs), t0 / zs, t0 / zs, fill=False, lw=0.8,
                                           ec="red" if r.p_tumour < 0.5 else "limegreen"))
                for r in s.itertuples():
                    ax.text((r.x0 - zx + t0 / 2) / zs, (r.y0 - zy + t0 / 2) / zs, str(r.item), color="white",
                            fontsize=8, ha="center", va="center", weight="bold",
                            bbox=dict(boxstyle="round,pad=0.12", fc="black", alpha=0.65, lw=0))
                ax.set_xlim(0, im.shape[1]); ax.set_ylim(im.shape[0], 0); ax.set_axis_off()
                ax.set_title(f"{pid.replace('_', ' ')} · region {chr(65 + k)} ({zm['n_tiles']} tumour tiles, "
                             f"{zm['mpp']:.1f} µm/px)\nyellow: outline · red: missed · green: detected · "
                             "numbers: tiles shown in detail", fontsize=11)
                save(fig)

            items = list(s.itertuples())
            for start in range(0, len(items), 6):
                fig, axes = plt.subplots(3, 4, figsize=(11.69, 8.27))
                for k, ax in enumerate(axes.ravel()):
                    ax.set_axis_off()
                    j = start + k // 2
                    if j >= len(items):
                        continue
                    r = items[j]
                    if k % 2 == 0:
                        ax.imshow(Image.open(tiles / r.path))
                        col = "firebrick" if r.p_tumour < 0.5 else "darkgreen"
                        ax.set_title(f"#{r.item} {r.group}\np = {r.p_tumour:.2f} · cov = {r.tumour_frac:.2f}",
                                     fontsize=8.5, color=col)
                    else:
                        c = crops[r.tile_id]
                        ax.imshow(c)
                        e = c.shape[0] / 3
                        ax.add_patch(Rectangle((e, e), e, e, fill=False, ec="yellow", lw=1.2))
                        ax.set_title(f"#{r.item} context, 384 µm", fontsize=8.5)
                fig.suptitle(f"{pid.replace('_', ' ')} — tiles {start + 1}–{min(start + 6, len(items))} of {len(items)}",
                             fontsize=11)
                fig.tight_layout()
                save(fig)
    print(f"wrote {out / 'review.pdf'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--runs", default="/media/abhi/camelyon17/derived/runs_v1")
    ap.add_argument("--tiles", default="/media/abhi/camelyon17/derived/tiles_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", action="store_true", help="also write each page as a PNG (for checking)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.select:
        select(Path(args.runs), Path(args.tiles), out)
    if args.draw:
        draw(Path(args.tiles), out, args.png)


if __name__ == "__main__":
    main()
