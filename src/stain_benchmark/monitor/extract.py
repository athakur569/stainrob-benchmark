"""
Extract Tier 1a and Tier 1b features for a set of tiles, optionally under a synthetic shift.

Output directory:
    tiles.csv     tile_id, patient_id, slide_id, center, label (row order of the features)
    tier1a.csv    tile_id + the 23 Tier 1a statistics
    tier1b.npy    float16 (n_tiles, 2048) frozen ImageNet ResNet-50 features
    info.json     tile subset rule, shift family / strength / seed, encoder, freeze hash
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from stain_benchmark.data.shifts import apply_shift
from stain_benchmark.monitor.features import TIER1A_NAMES, Tier1bEncoder, tier1a


class _Tiles(Dataset):
    def __init__(self, tiles_dir: Path, table: pd.DataFrame, shift: str | None, strength: float, seed: int):
        self.tiles_dir, self.table = tiles_dir, table.reset_index(drop=True)
        self.shift, self.strength, self.seed = shift, strength, seed

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int):
        row = self.table.iloc[i]
        img = np.array(Image.open(self.tiles_dir / row["path"]).convert("RGB"))
        if self.shift:
            img = apply_shift(img, self.shift, self.strength, self.seed, row["tile_id"])
        return torch.from_numpy(img), torch.from_numpy(tier1a(img))


def extract_tier1(tiles_dir: str | Path, table: pd.DataFrame, out_dir: str | Path, subset_rule: str,
                  shift: str | None = None, strength: float = 0.0, seed: int = 0, num_workers: int = 8,
                  batch_size: int = 64, device: str | None = None, pretrained: bool = True,
                  progress: bool = True) -> dict:
    """Compute Tier 1a/1b for every row of ``table``; refuses to overwrite an existing extraction."""
    tiles_dir, out_dir = Path(tiles_dir), Path(out_dir)
    if (out_dir / "info.json").exists():
        raise FileExistsError(f"{out_dir} already holds an extraction")
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = Tier1bEncoder(pretrained=pretrained).to(dev)
    loader = DataLoader(_Tiles(tiles_dir, table, shift, strength, seed), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    f1a, f1b = [], []
    for imgs, stats in tqdm(loader, desc=f"tier1 {shift or 'unshifted'} {strength:g}", disable=not progress):
        f1a.append(stats.numpy())
        f1b.append(enc(imgs).half().cpu().numpy())
    keep = table[["tile_id", "patient_id", "slide_id", "center", "label"]].reset_index(drop=True)
    keep.to_csv(out_dir / "tiles.csv", index=False)
    pd.DataFrame(np.concatenate(f1a), columns=TIER1A_NAMES).assign(tile_id=keep["tile_id"])[
        ["tile_id", *TIER1A_NAMES]].to_csv(out_dir / "tier1a.csv", index=False)
    np.save(out_dir / "tier1b.npy", np.concatenate(f1b))
    freeze_path = tiles_dir / "freeze.json"
    info = {"n_tiles": len(keep), "subset_rule": subset_rule, "shift": shift, "strength": strength,
            "shift_seed": seed, "tier1b_encoder": "torchvision resnet50 IMAGENET1K_V2, frozen" if pretrained
            else "resnet50, random weights (test only)",
            "freeze_master_sha256": json.loads(freeze_path.read_text())["master_sha256"]
            if freeze_path.exists() else None}
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    return info
