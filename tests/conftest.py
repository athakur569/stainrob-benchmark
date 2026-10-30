"""Shared fixtures: a tiny frozen tile set with three centers."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from stain_benchmark.data.freeze import SPLIT_FILES, freeze_dataset

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
                Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(out / path)
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
