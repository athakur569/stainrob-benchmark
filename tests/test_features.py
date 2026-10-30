"""Tests for the Tier 1 features."""

import numpy as np
import pytest
import torch

from stain_benchmark.data.shifts import apply_shift
from stain_benchmark.monitor.features import (
    COLOUR_FEATURES,
    TIER1A_NAMES,
    Tier1bEncoder,
    tier1a,
)
from tests.test_shifts import he_like

IDX = {n: i for i, n in enumerate(TIER1A_NAMES)}
TILE = he_like(size=128)


def test_tier1a_shape_names_and_determinism():
    f = tier1a(TILE)
    assert f.shape == (23,) == (len(TIER1A_NAMES),) and len(COLOUR_FEATURES) == 18
    assert np.array_equal(f, tier1a(TILE)) and np.isfinite(f).all()


def test_flat_grey_tile_has_no_texture_and_no_tissue():
    f = tier1a(np.full((64, 64, 3), 200, np.uint8))
    assert f[IDX["grey_sd"]] == 0 and f[IDX["log_laplacian_var"]] == 0
    assert f[IDX["grey_entropy"]] == 0 and f[IDX["tissue_fraction"]] == 0
    assert f[IDX["r_mean"]] == 200


def test_blur_lowers_sharpness_more_than_colour():
    g = tier1a(apply_shift(TILE, "blur", 3.0))
    f = tier1a(TILE)
    assert g[IDX["log_laplacian_var"]] < f[IDX["log_laplacian_var"]] - 1
    assert abs(g[IDX["lab_l_mean"]] - f[IDX["lab_l_mean"]]) < 1      # blur keeps mean colour


def test_stain_shift_moves_stain_channels():
    g = tier1a(apply_shift(TILE, "stain", 0.8, key="k"))
    f = tier1a(TILE)
    moved = np.abs(g - f)[[IDX["hed_h_mean"], IDX["hed_e_mean"]]]
    assert moved.max() > 0.005


def test_bad_input_raises():
    with pytest.raises(ValueError, match="RGB uint8"):
        tier1a(TILE.astype(np.float32))


def test_tier1b_frozen_shape_and_determinism():
    enc = Tier1bEncoder(pretrained=False)
    enc.train()                                            # stays in eval mode regardless
    assert not enc.net.training
    assert not any(p.requires_grad for p in enc.parameters())
    x = torch.from_numpy(np.stack([TILE, TILE[::-1].copy()]))
    a, b = enc(x), enc(x)
    assert a.shape == (2, 2048) and torch.equal(a, b)


def test_extract_writes_aligned_outputs(frozen, tmp_path):
    import json

    import pandas as pd

    from stain_benchmark.data.dataset import fold_table
    from stain_benchmark.monitor.extract import extract_tier1
    table = fold_table(frozen, holdout=0)
    info = extract_tier1(frozen, table, tmp_path / "x", "test", num_workers=0, device="cpu",
                         pretrained=False, progress=False)
    tiles = pd.read_csv(tmp_path / "x" / "tiles.csv")
    f1a = pd.read_csv(tmp_path / "x" / "tier1a.csv")
    f1b = np.load(tmp_path / "x" / "tier1b.npy")
    assert len(tiles) == len(f1a) == len(f1b) == len(table) == info["n_tiles"]
    assert (tiles.tile_id == f1a.tile_id).all() and f1b.shape[1] == 2048
    assert json.loads((tmp_path / "x" / "info.json").read_text())["shift"] is None
    shifted = extract_tier1(frozen, table, tmp_path / "y", "test", shift="blur", strength=2.0,
                            num_workers=0, device="cpu", pretrained=False, progress=False)
    g1a = pd.read_csv(tmp_path / "y" / "tier1a.csv")
    assert shifted["strength"] == 2.0
    assert (g1a.log_laplacian_var < f1a.log_laplacian_var).all()
    with pytest.raises(FileExistsError):
        extract_tier1(frozen, table, tmp_path / "x", "test", num_workers=0, device="cpu", pretrained=False)
