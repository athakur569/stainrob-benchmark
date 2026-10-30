"""Tests for the fold tables, tile dataset and loaders."""

import json

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
from stain_benchmark.data.freeze import SPLIT_FILES, FreezeMismatchError, freeze_dataset

# 3 centers x 4 patients; patient split per center: 2 train, 1 val, 1 test.
SPLIT = {0: "train", 1: "train", 2: "val", 3: "test"}


@pytest.fixture
def frozen(tmp_path):
    out = tmp_path / "tiles_v1"
    rows, pats = [], []
    rng = np.random.default_rng(0)
    for c in range(3):
        for p in range(4):
            pid = f"p{c}{p}"
            pats.append(dict(patient_id=pid, label=p % 2, center=c, tumour_tiles=2 * (p % 2), split=SPLIT[p]))
            for t in range(2):
                path = f"tiles/{c}/{pid}_{t}.png"
                (out / "tiles" / str(c)).mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(out / path)
                rows.append(dict(tile_id=f"{pid}_{t}", path=path, patient_id=pid, slide_id=f"{pid}_n0",
                                 center=c, label=p % 2, qc_pass=True))
    rows.append(dict(rows[-1], tile_id="bad", qc_pass=False))            # QC failure is excluded
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    (out / "split_input.csv").write_text("slide_id\ns\n")
    splits = out / "splits"; splits.mkdir()
    for f in SPLIT_FILES:
        (splits / f).write_text("x\n")
    pd.DataFrame(pats).to_csv(splits / "patient_splits.csv", index=False)
    (splits / "split_metadata.json").write_text(json.dumps({"seed": 1}))
    cfg = tmp_path / "cfg.yaml"; cfg.write_text("a: 1\n")
    meta = tmp_path / "metadata.csv"; meta.write_text("slide_id\ns\n")
    freeze_dataset(out, cfg, meta, splits, progress=False)
    return out


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
    assert torch.allclose(item["images"], ((0 - IMAGENET_MEAN) / IMAGENET_STD).expand(3, 16, 16))


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
