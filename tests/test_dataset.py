"""Tests for the fold tables, tile dataset and loaders."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from stain_benchmark.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    TileDataset,
    fold_table,
    geometric_augment,
    make_loaders,
)
from stain_benchmark.data.freeze import FreezeMismatchError

def test_fold_roles_and_val_merge(frozen):
    t = fold_table(frozen, holdout=1)
    assert len(t) == 24                                   # QC failure dropped
    assert set(t.loc[t.center == 1, "role"]) == {"test_cross"}
    other = t[t.center != 1]
    # val (patient 2) merged into test_in alongside test (patient 3)
    assert set(other.loc[other.patient_id.str[-1] == "2", "role"]) == {"test_in"}
    assert set(other["role"]) == {"train", "test_in"}
    kept = fold_table(frozen, holdout=1, merge_val=False)
    assert "val" in set(kept["role"])


def test_no_patient_in_two_roles(frozen):
    for h in range(3):
        t = fold_table(frozen, holdout=h)
        assert (t.groupby("patient_id")["role"].nunique() == 1).all()


def test_unknown_holdout_raises(frozen):
    with pytest.raises(ValueError, match="not in the tile set"):
        fold_table(frozen, holdout=9)


def test_item_is_imagenet_normalised(frozen):
    t = fold_table(frozen, holdout=0)
    item = TileDataset(frozen, t)[0]
    raw = np.asarray(Image.open(frozen / t.path[0]).convert("RGB"), dtype=np.float32) / 255
    back = item["images"] * IMAGENET_STD + IMAGENET_MEAN
    assert torch.allclose(back.permute(1, 2, 0), torch.from_numpy(raw), atol=1e-6)
    assert item["labels"] == int(t.label[0]) and item["index"] == 0


def test_normalizer_is_applied(frozen):
    t = fold_table(frozen, holdout=0)
    item = TileDataset(frozen, t, normalizer=lambda rgb: np.zeros_like(rgb))[0]
    assert torch.allclose(item["images"], ((0 - IMAGENET_MEAN) / IMAGENET_STD).expand(3, 32, 32))


def test_geometric_augment_keeps_pixel_values():
    x = torch.rand(3, 8, 8)
    y = geometric_augment(x)
    assert torch.equal(torch.sort(x.flatten())[0], torch.sort(y.flatten())[0])


def test_loaders_verify_freeze(frozen):
    pd.read_csv(frozen / "manifest.csv").assign(label=0).to_csv(frozen / "manifest.csv", index=False)
    with pytest.raises(FreezeMismatchError):
        make_loaders(frozen, holdout=0, seed=42, num_workers=0)


def _train_order(frozen, seed):
    loaders, _ = make_loaders(frozen, holdout=0, seed=seed, batch_size=4, num_workers=0)
    return [i for b in loaders["train"] for i in b["index"].tolist()]


def test_loaders_reproducible_and_seed_sensitive(frozen):
    assert _train_order(frozen, 42) == _train_order(frozen, 42)
    assert _train_order(frozen, 42) != _train_order(frozen, 7)


def test_loader_roles_cover_table(frozen):
    loaders, table = make_loaders(frozen, holdout=2, seed=42, batch_size=5, num_workers=0)
    for role, dl in loaders.items():
        assert len(dl.dataset) == (table.role == role).sum()


def test_worker_augmentation_differs_between_epochs(frozen):
    loaders, _ = make_loaders(frozen, holdout=0, seed=42, batch_size=32, num_workers=2)
    dl = loaders["train"]
    order = lambda b: b["images"][torch.argsort(b["index"])]
    e1, e2 = order(next(iter(dl))), order(next(iter(dl)))
    assert not torch.equal(e1, e2)          # new random flips/rotations each epoch


def test_shift_is_applied_before_normalising(frozen):
    t = fold_table(frozen, holdout=0)
    plain = TileDataset(frozen, t)[0]["images"]
    shifted = TileDataset(frozen, t, shift=("blur", 2.0, 0))[0]["images"]
    again = TileDataset(frozen, t, shift=("blur", 2.0, 0))[0]["images"]
    assert not torch.equal(plain, shifted) and torch.equal(shifted, again)


def test_custom_table_roles(frozen):
    t = fold_table(frozen, holdout=0)
    t = t.assign(role=np.where(t.patient_id.str[-1].isin(["0", "1"]), "train", "pilot"))
    loaders, table = make_loaders(frozen, holdout=0, seed=1, batch_size=4, num_workers=0, table=t)
    assert set(loaders) == {"train", "pilot"} and len(loaders["pilot"].dataset) == (t.role == "pilot").sum()


def test_estimate_on_clean_passes_the_unshifted_tile_as_source(frozen):
    """P5 oracle: the normaliser sees the shifted tile but estimates on the clean one."""
    t = fold_table(frozen, 0)
    seen = {}

    def norm(rgb, source=None):
        seen["source"] = source
        return rgb

    TileDataset(frozen, t, normalizer=norm, shift=("blur", 2.0, 0), estimate_on_clean=True)[0]
    clean = np.array(Image.open(frozen / t["path"].iloc[0]).convert("RGB"))
    assert np.array_equal(seen["source"], clean)
    TileDataset(frozen, t, normalizer=norm, shift=("blur", 2.0, 0))[0]
    assert seen["source"] is None                                   # default: estimate on the tile as it arrives
    with pytest.raises(ValueError):
        TileDataset(frozen, t, normalizer=norm, estimate_on_clean=True)
