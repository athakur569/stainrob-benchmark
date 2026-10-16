"""Tests for dataset-agnostic tile planning, sampling and QC (run in the `wsitile` env)."""

import numpy as np
import pytest

from stain_benchmark.data.tiling import (
    NORMAL, TUMOUR, TilingConfig, parse_annotations, plan_tiles, rasterize, read_tile, tissue_mask,
)

CFG = TilingConfig()


def test_tissue_mask_rejects_black_void_and_white_glass():
    img = np.zeros((60, 90, 3), np.uint8)              # black unscanned void
    img[:, 30:60] = (235, 235, 235)                     # white glass
    img[:, 60:90] = (200, 120, 180)                     # pink-purple stained tissue
    mask, _ = tissue_mask(img, CFG)
    assert mask[:, 5:25].mean() == 0
    assert mask[:, 35:55].mean() == 0
    assert mask[:, 65:85].mean() > 0.95


def _masks(shape=(100, 100)):
    tissue = np.ones(shape, bool)
    tumour = np.zeros(shape, bool)
    excl = np.zeros(shape, bool)
    return tissue, tumour, excl


# Overview at 8 um/px and mpp0 = 0.5 -> ds = 16; one tile = 128 um = 16 overview px.
DS, MPP0 = 16.0, 0.5


def _label_at(df, tx, ty):
    step0 = CFG.tile_um / MPP0
    hit = df[(df.x0 == round(tx * step0)) & (df.y0 == round(ty * step0))]
    return None if hit.empty else int(hit.label.iloc[0])


def test_label_rules():
    tissue, tumour, excl = _masks()
    tumour[0:16, 0:16] = True       # tile (0,0) fully tumour
    tumour[0:16, 16:24] = True      # tile (1,0) 50% tumour
    tumour[32:48, 32:36] = True     # tile (2,2) 25% tumour -> discard
    excl[80:96, 80:96] = True       # tile (5,5) excluded
    df = plan_tiles(tissue, tumour, excl, DS, MPP0, "annotated", CFG)
    assert _label_at(df, 0, 0) == TUMOUR
    assert _label_at(df, 1, 0) == TUMOUR
    assert _label_at(df, 2, 2) is None          # 1-49% inside: discarded
    assert _label_at(df, 2, 0) is None          # within 128 um of an outline: discarded
    assert _label_at(df, 3, 0) == NORMAL        # 192 um from the outline: normal
    assert _label_at(df, 5, 0) == NORMAL        # far from any outline
    assert _label_at(df, 5, 5) is None          # exclusion region


def test_low_tissue_tiles_dropped():
    tissue, tumour, excl = _masks()
    tissue[:, :] = False
    tissue[0:16, 0:10] = True                    # tile (0,0) ~62% tissue < 70%
    tissue[0:16, 16:32] = True                   # tile (1,0) 100% tissue
    df = plan_tiles(tissue, tumour, excl, DS, MPP0, "slide_label", CFG, slide_label=NORMAL)
    assert _label_at(df, 0, 0) is None
    assert _label_at(df, 1, 0) == NORMAL


def test_negative_slide_all_normal():
    tissue, tumour, excl = _masks()
    df = plan_tiles(tissue, tumour, excl, DS, MPP0, "slide_label", CFG, slide_label=NORMAL)
    assert len(df) == 36 and (df.label == NORMAL).all()


def test_rasterize_and_parse(tmp_path):
    xml = tmp_path / "s.xml"
    xml.write_text(
        '<ASAP_Annotations><Annotations>'
        '<Annotation Type="Polygon" PartOfGroup="Tumor"><Coordinates>'
        '<Coordinate Order="0" X="0" Y="0"/><Coordinate Order="1" X="320" Y="0"/>'
        '<Coordinate Order="2" X="320" Y="320"/><Coordinate Order="3" X="0" Y="320"/>'
        '</Coordinates></Annotation></Annotations></ASAP_Annotations>'
    )
    anns = parse_annotations(xml)
    assert anns[0][0] == "Tumor"
    m = rasterize([anns[0][1]], (40, 40), ds=16.0)
    assert 0.24 < m.mean() < 0.29                # 20x20 px (+ boundary) of 40x40


class FakeSlide:
    def __init__(self, mpp0):
        self.properties = {"openslide.mpp-x": str(mpp0)}
        self.level_downsamples = (1.0, 2.0, 4.0)
        self.requested = None

    def get_best_level_for_downsample(self, d):
        return max(i for i, x in enumerate(self.level_downsamples) if x <= d + 1e-9)

    def read_region(self, loc, level, size):
        from PIL import Image
        self.requested = (level, size)
        return Image.new("RGBA", size, (200, 120, 180, 255))


@pytest.mark.parametrize("mpp0,expect_px", [(0.2431, 263), (0.2269, 282), (0.25, 256)])
def test_read_tile_same_physical_area(mpp0, expect_px):
    s = FakeSlide(mpp0)
    tile = read_tile(s, 0, 0, CFG)
    level, size = s.requested
    assert level == 1 and size == (expect_px, expect_px)
    assert tile.size == (256, 256)


def _cand():
    rows = []
    for center in (0, 1):
        rows += [dict(center=center, slide_id=f"c{center}_big", x0=i, y0=0, label=TUMOUR) for i in range(900)]
        rows += [dict(center=center, slide_id=f"c{center}_small", x0=i, y0=0, label=TUMOUR) for i in range(40)]
        for k in range(10):
            n = 3 if k == 0 else 500        # one slide with only 3 normal candidates
            rows += [dict(center=center, slide_id=f"c{center}_neg{k}", x0=i, y0=1, label=NORMAL) for i in range(n)]
    import pandas as pd
    return pd.DataFrame(rows)


def test_sampling_caps_balance_and_spread():
    from stain_benchmark.data.tiling import sample_tiles
    out = sample_tiles(_cand(), tumour_cap=500, normal_ratio=1.0, seed=42)
    for center, c in out.groupby("center"):
        per = c[c.label == TUMOUR].groupby("slide_id").size()
        assert per.max() == 500 and per.min() == 40               # cap respected, small slide kept whole
        n_t, n_n = (c.label == TUMOUR).sum(), (c.label == NORMAL).sum()
        assert n_n == n_t == 540                                    # 1:1 per center
        nper = c[c.label == NORMAL].groupby("slide_id").size()
        assert nper.min() == 3 and nper.drop(f"c{center}_neg0").max() - nper.drop(f"c{center}_neg0").min() <= 1
    assert out.tile_id.is_unique


def test_sampling_is_deterministic():
    from stain_benchmark.data.tiling import sample_tiles
    a = sample_tiles(_cand(), 500, 1.0, seed=42); b = sample_tiles(_cand(), 500, 1.0, seed=42)
    c = sample_tiles(_cand(), 500, 1.0, seed=7)
    assert a.tile_id.tolist() == b.tile_id.tolist() and a.tile_id.tolist() != c.tile_id.tolist()


def test_slide_label_role_tumour_slide():
    tissue, tumour, excl = _masks()
    df = plan_tiles(tissue, tumour, excl, DS, MPP0, "slide_label", CFG, slide_label=TUMOUR)
    assert len(df) == 36 and (df.label == TUMOUR).all()     # TCGA-style: every tissue tile = slide label


def test_unknown_role_rejected():
    tissue, tumour, excl = _masks()
    with pytest.raises(ValueError):
        plan_tiles(tissue, tumour, excl, DS, MPP0, "negative", CFG)
