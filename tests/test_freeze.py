"""Tests for dataset freezing and verification."""

import json

import pandas as pd
import pytest

from stain_benchmark.data.freeze import SPLIT_FILES, FreezeMismatchError, freeze_dataset, verify_freeze


@pytest.fixture
def ds(tmp_path):
    out = tmp_path / "tiles_v1"; (out / "tiles").mkdir(parents=True)
    rows = []
    for i in range(4):
        p = out / "tiles" / f"t{i}.png"; p.write_bytes(bytes([i]) * 10)
        rows.append(dict(tile_id=f"t{i}", path=f"tiles/t{i}.png", center="0", label=i % 2, qc_pass=True))
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    (out / "split_input.csv").write_text("slide_id\ns\n")
    splits = out / "splits"; splits.mkdir()
    for f in SPLIT_FILES:
        (splits / f).write_text("x\n")
    (splits / "split_metadata.json").write_text(json.dumps({"seed": 1, "checksum": "abc"}))
    cfg = tmp_path / "cfg.yaml"; cfg.write_text("a: 1\n")
    meta = tmp_path / "metadata.csv"; meta.write_text("slide_id\ns\n")
    return out, cfg, meta, splits


def test_freeze_then_verify_passes(ds):
    out, cfg, meta, splits = ds
    rec = freeze_dataset(out, cfg, meta, splits, progress=False)
    assert rec["n_tiles"] == 4 and len(rec["master_sha256"]) == 64
    verify_freeze(out, full=True)


def test_refuses_to_overwrite(ds):
    out, cfg, meta, splits = ds
    freeze_dataset(out, cfg, meta, splits, progress=False)
    with pytest.raises(FileExistsError):
        freeze_dataset(out, cfg, meta, splits, progress=False)


def test_changed_split_detected_fast(ds):
    out, cfg, meta, splits = ds
    freeze_dataset(out, cfg, meta, splits, progress=False)
    (splits / "patient_splits.csv").write_text("tampered\n")
    with pytest.raises(FreezeMismatchError, match="patient_splits"):
        verify_freeze(out)


def test_changed_tile_detected_only_in_full(ds):
    out, cfg, meta, splits = ds
    freeze_dataset(out, cfg, meta, splits, progress=False)
    (out / "tiles" / "t2.png").write_bytes(b"different")
    verify_freeze(out)                       # fast mode does not re-hash tiles
    with pytest.raises(FreezeMismatchError, match="tiles"):
        verify_freeze(out, full=True)
