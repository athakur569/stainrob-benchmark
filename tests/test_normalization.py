"""Tests for stain normalizers (E1 identity, E2 Macenko, E3 Reinhard) on synthetic H&E tiles."""

import logging

import numpy as np
import pytest

from stain_benchmark.normalization.base import IdentityNormalizer
from stain_benchmark.normalization.macenko import MacenkoNormalizer
from stain_benchmark.normalization.reinhard import (
    ReinhardNormalizer, ReinhardRobustNormalizer, _lab_to_rgb, _rgb_to_lab,
)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


H, E = _unit([0.65, 0.70, 0.29]), _unit([0.07, 0.99, 0.11])   # Ruifrok & Johnston


def he_tile(seed, h=H, e=E, scale=1.0, size=128):
    """Synthetic H&E tile following the Beer-Lambert two-stain model."""
    r = np.random.default_rng(seed); n = size * size
    C = np.column_stack([r.gamma(2, 0.25, n) * (r.random(n) < 0.4), r.gamma(3, 0.15, n)]) * scale
    return np.clip(255 * np.exp(-(C @ np.vstack([h, e]))), 1, 255).reshape(size, size, 3).astype(np.uint8)


REF = he_tile(1)
OTHER = he_tile(2, _unit([0.55, 0.80, 0.25]), _unit([0.15, 0.95, 0.25]), scale=1.3)


def _gap(a, b, stat=np.mean):
    """Distance between Lab colour statistics of the TISSUE pixels (L* < 80) of two images --
    the population Reinhard matches; background is transformed with tissue statistics."""
    la, lb = _rgb_to_lab(a).reshape(-1, 3), _rgb_to_lab(b).reshape(-1, 3)
    la, lb = la[la[:, 0] < 80], lb[lb[:, 0] < 80]
    return np.abs(stat(la, axis=0) - stat(lb, axis=0)).sum()


def test_lab_helpers_round_trip():
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    lab = _rgb_to_lab(img)
    assert 0 <= lab[..., 0].min() and lab[..., 0].max() <= 100            # true CIE L*
    assert np.abs(_lab_to_rgb(lab).astype(int) - img).max() <= 1


@pytest.mark.parametrize("cls", [ReinhardNormalizer, ReinhardRobustNormalizer, MacenkoNormalizer])
def test_self_normalisation_near_identity(cls):
    out = cls().fit(REF).transform(REF)
    assert out.dtype == np.uint8 and out.shape == REF.shape
    assert np.abs(out.astype(int) - REF).mean() < 3.0


@pytest.mark.parametrize("cls", [ReinhardNormalizer, ReinhardRobustNormalizer, MacenkoNormalizer])
def test_moves_other_stain_toward_reference(cls):
    out = cls().fit(REF).transform(OTHER)
    assert out.mean() > 60                                                 # not the old near-black output
    stat = np.median if cls is ReinhardRobustNormalizer else np.mean       # robust variant matches medians
    assert _gap(out, REF, stat) < 0.5 * _gap(OTHER, REF, stat)


def test_reinhard_uses_tissue_mask_without_fallback(caplog):
    with caplog.at_level(logging.WARNING):
        ReinhardNormalizer().fit(REF)
    assert "Few tissue pixels" not in caplog.text


def test_macenko_recovers_stain_vectors():
    sv = np.asarray(MacenkoNormalizer().fit(REF).get_stain_vectors())
    sv = sv if sv.shape[0] == 2 else sv.T
    assert max(abs(sv[0] @ H), abs(sv[1] @ H)) > 0.98
    assert max(abs(sv[0] @ E), abs(sv[1] @ E)) > 0.98


def test_identity_is_unchanged():
    assert np.array_equal(IdentityNormalizer().fit(REF).transform(OTHER), OTHER)


def test_macenko_stain_rows_are_h_then_e_and_non_negative():
    for img in (REF, OTHER, np.ascontiguousarray(REF[:, ::-1])):
        sv, _ = MacenkoNormalizer()._get_stain_matrix(img)
        assert (sv >= -1e-6).all()                      # absorbance is never negative
        assert sv[0, 0] > sv[1, 0]                      # H absorbs more red than E


def test_macenko_preserves_brightness_with_background():
    # Tiles with white background (as real tiles have) and a reference fitted on a different
    # tile stacked into a mosaic: the output must stay a tissue image, not go dark.
    def with_background(img):
        out = img.copy(); out[:, : img.shape[1] // 3] = 245
        return out
    ref = np.concatenate([with_background(he_tile(s)) for s in (3, 4, 5)], axis=0)
    src = with_background(OTHER)
    out = MacenkoNormalizer().fit(ref).transform(src)
    assert out.shape == src.shape
    assert abs(float(out.mean()) - float(src.mean())) < 40
    assert out[:, : src.shape[1] // 3].mean() > 200     # background stays light
