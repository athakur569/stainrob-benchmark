"""Tests for the graded synthetic shifts."""

import numpy as np
import pytest

from stain_benchmark.data.shifts import (
    FAMILIES,
    GLASS,
    apply_shift,
    calibrate_strength,
    delta_e,
    mean_measure,
    mean_ssim,
    ssim,
)


def he_like(seed=0, size=64):
    """Textured pink/purple tile (cells on stroma) so blur and JPEG have structure to act on."""
    r = np.random.default_rng(seed)
    base = np.full((size, size, 3), (225, 160, 200), np.float32)
    for _ in range(40):
        y, x = r.integers(4, size - 4, 2)
        rad = int(r.integers(2, 5))
        yy, xx = np.ogrid[:size, :size]
        base[(yy - y) ** 2 + (xx - x) ** 2 <= rad ** 2] = (90, 60, 150)
    return np.clip(base + r.normal(0, 8, base.shape), 0, 255).astype(np.uint8)


TILE = he_like()


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_strength_is_identity(family):
    assert np.array_equal(apply_shift(TILE, family, 0.0), TILE)


@pytest.mark.parametrize("family", FAMILIES)
def test_shape_dtype_and_determinism(family):
    a = apply_shift(TILE, family, 0.5 if family != "jpeg" else 60, seed=3, key="t1")
    b = apply_shift(TILE, family, 0.5 if family != "jpeg" else 60, seed=3, key="t1")
    assert a.shape == TILE.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)


@pytest.mark.parametrize("family,strengths", [
    ("blur", [0.5, 1.5, 4.0]), ("striping", [0.1, 0.3, 0.6]), ("clipped", [0.1, 0.3, 0.6]),
    ("jpeg", [20, 70, 95]),
])
def test_ssim_falls_as_strength_rises(family, strengths):
    tiles = [he_like(s) for s in range(4)]
    values = [mean_ssim(tiles, family, s) for s in strengths]
    assert values == sorted(values, reverse=True)
    assert values[-1] < 0.95


def test_stain_shift_mostly_keeps_structure():
    # A strong stain change alters colour but keeps structure far better than a strong blur.
    tiles = [he_like(s) for s in range(4)]
    assert mean_ssim(tiles, "stain", 0.6) > mean_ssim(tiles, "blur", 4.0)


def test_clipped_blanks_the_requested_fraction():
    out = apply_shift(TILE, "clipped", 0.25, key="x")
    assert np.isclose((out == GLASS).all(axis=2).mean(), 0.25, atol=0.03)


def test_striping_darkens_alternate_bands_only():
    flat = np.full((64, 64, 3), 200, np.uint8)
    out = apply_shift(flat, "striping", 0.5, key="x")
    rows = out[:, 0, 0]
    assert set(np.unique(rows)) == {100, 200}
    assert (rows[:16] == rows[0]).all() and rows[0] != rows[16]


def test_calibration_hits_ssim_target():
    tiles = [he_like(s) for s in range(4)]
    s, achieved = calibrate_strength(tiles, "blur", 0.80, measure="ssim")
    assert abs(achieved - 0.80) < 0.01 and s > 0


@pytest.mark.parametrize("family", ["blur", "stain", "clipped"])
def test_calibration_hits_delta_e_target(family):
    # 256-px tiles as in the real data: "clipped" blanks whole rows, so it moves in steps of
    # 1/height of the tile (~0.2 ΔE at 256 px, but ~0.9 ΔE at 64 px).
    tiles = [he_like(s, size=256) for s in range(4)]
    s, achieved = calibrate_strength(tiles, family, 4.0, measure="delta_e")
    assert abs(achieved - 4.0) < 0.1 and s > 0


def test_delta_e_rises_with_strength_for_every_family():
    tiles = [he_like(s) for s in range(3)]
    for fam, (a, b) in {"blur": (0.5, 3), "striping": (0.1, 0.5), "clipped": (0.1, 0.5),
                        "jpeg": (30, 95), "stain": (0.2, 0.8)}.items():
        assert mean_measure(tiles, fam, a) < mean_measure(tiles, fam, b)


def test_unreachable_target_reported():
    flat = [np.full((32, 32, 3), 200, np.uint8)]
    s, achieved = calibrate_strength(flat, "blur", 0.5, measure="ssim")   # blurring a flat tile changes nothing
    assert achieved > 0.5 and s == 12.0


def test_unknown_family_and_bad_input_raise():
    with pytest.raises(ValueError, match="unknown shift family"):
        apply_shift(TILE, "noise", 1.0)
    with pytest.raises(ValueError, match="RGB uint8"):
        apply_shift(TILE.astype(np.float32), "blur", 1.0)


def test_ssim_and_delta_e_identity():
    assert ssim(TILE, TILE) == pytest.approx(1.0)
    assert delta_e(TILE, TILE) == pytest.approx(0.0)


def test_jpeg_calibration_picks_the_closer_integer_quality():
    tiles = [he_like(s, size=128) for s in range(3)]
    s, achieved = calibrate_strength(tiles, "jpeg", 6.0, measure="delta_e")
    assert s == int(s)
    for neighbour in (s - 1, s + 1):
        assert abs(achieved - 6.0) <= abs(mean_measure(tiles, "jpeg", neighbour) - 6.0) + 1e-9
