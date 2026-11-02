"""
Tile datasets and loaders for one leave-one-center-out fold of a frozen tile set.

The frozen record is checked before anything is loaded (`verify_freeze`), so a run can only
ever see the tiles, splits and config that were hashed. Roles per tile come from `loco_fold`:
``train``, ``test_in`` (unseen patients from training centers) and ``test_cross`` (the
held-out center). The validation split is merged into ``test_in`` (DR-009): the training
schedule is fixed in advance, so nothing is selected on validation data.

Augmentation is geometric only — flips and 90° rotations, which tissue does not care about.
Colour augmentation is itself a stain-handling method, so it is never applied here; any
stain handling enters only through the ``normalizer`` of the condition being run (DR-010).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from stain_benchmark.data.freeze import verify_freeze
from stain_benchmark.data.shifts import apply_shift
from stain_benchmark.data.splitting import loco_fold
from stain_benchmark.utils.reproducibility import WorkerInitFn, get_generator

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
ROLES = ("train", "test_in", "test_cross")


def fold_table(out_dir: str | Path, holdout: str | int, merge_val: bool = True) -> pd.DataFrame:
    """QC-passing tiles of the frozen set with a ``role`` column for the fold holding out ``holdout``."""
    out_dir = Path(out_dir)
    tiles = pd.read_csv(out_dir / "manifest.csv")
    tiles = tiles[tiles["qc_pass"]].reset_index(drop=True)
    patient_splits = pd.read_csv(out_dir / "splits" / "patient_splits.csv")
    if str(holdout) not in set(tiles["center"].astype(str)):
        raise ValueError(f"holdout center {holdout!r} not in the tile set")
    role = loco_fold(tiles, patient_splits, holdout=str(holdout), group_column="center")
    if merge_val:
        role = role.replace({"val": "test_in"})
    return tiles.assign(role=role.to_numpy())


def geometric_augment(x: torch.Tensor) -> torch.Tensor:
    """Random flip and 90° rotation of a (3, H, W) tile, using the torch RNG."""
    if torch.rand(()) < 0.5:
        x = torch.flip(x, dims=[2])
    k = int(torch.randint(0, 4, ()))
    return torch.rot90(x, k, dims=[1, 2]) if k else x


class TileDataset(Dataset):
    """Tiles listed in ``table`` (rows of the frozen manifest), as ImageNet-normalised tensors.

    Each item is a dict: ``images`` (3, H, W) float, ``labels`` (0 normal / 1 tumour),
    ``index`` (the row's position in ``table``, so per-tile outputs can be joined back) and,
    if ``domains`` is given, ``domain`` (the tile's training-centre index, used by E5).
    ``shift`` = (family, strength, seed) applies a synthetic shift to each tile as it is read,
    before any normaliser — as a shifted slide would arrive at deployment.
    ``normalizer`` maps a uint8 RGB array to a uint8 RGB array (E2/E3 stain normalisation).
    """

    def __init__(self, out_dir: str | Path, table: pd.DataFrame,
                 normalizer: Callable[[np.ndarray], np.ndarray] | None = None,
                 augment: bool = False, domains: np.ndarray | None = None,
                 shift: tuple[str, float, int] | None = None):
        self.out_dir = Path(out_dir)
        self.paths = table["path"].tolist()
        self.tile_ids = table["tile_id"].astype(str).tolist()
        self.shift = shift
        self.labels = table["label"].astype(int).to_numpy()
        self.normalizer = normalizer
        self.augment = augment
        self.domains = None if domains is None else np.asarray(domains, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        rgb = np.array(Image.open(self.out_dir / self.paths[i]).convert("RGB"))
        if self.shift is not None:
            family, strength, seed = self.shift
            rgb = apply_shift(rgb, family, strength, seed, self.tile_ids[i])
        if self.normalizer is not None:
            rgb = self.normalizer(rgb)
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255)
        if self.augment:
            x = geometric_augment(x)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        item = {"images": x, "labels": int(self.labels[i]), "index": i}
        if self.domains is not None:
            item["domain"] = int(self.domains[i])
        return item


def make_loaders(out_dir: str | Path, holdout: str | int, seed: int, batch_size: int = 64,
                 num_workers: int = 8, normalizer: Callable[[np.ndarray], np.ndarray] | None = None,
                 verify: bool = True, merge_val: bool = True,
                 table: pd.DataFrame | None = None) -> tuple[dict[str, DataLoader], pd.DataFrame]:
    """Verify the freeze, then build one loader per role for the fold holding out ``holdout``.

    Returns the loaders and the fold table; each table row's ``role`` says which loader holds it,
    and a batch's ``index`` is the row position within that role's subset (``table[table.role == r]``).
    Only ``train`` is shuffled and augmented; shuffling uses a generator seeded from ``seed``.
    ``table`` (rows of the manifest with a ``role`` column) replaces the fold table, e.g. for the
    dose-range pilot inside the training split; loaders are then built for its roles.
    """
    if verify:
        verify_freeze(out_dir)
    if table is None:
        table = fold_table(out_dir, holdout, merge_val=merge_val)
    roles = ROLES if set(table["role"]) <= set(ROLES) else tuple(sorted(set(table["role"])))
    loaders = {}
    for role in roles:
        sub = table[table["role"] == role].reset_index(drop=True)
        train = role == "train"
        domains = None
        if train:  # training-centre index 0..K-1, for the domain-adversarial condition (E5)
            centres = sorted(sub["center"].astype(str).unique())
            domains = sub["center"].astype(str).map({c: k for k, c in enumerate(centres)}).to_numpy()
        loaders[role] = DataLoader(
            TileDataset(out_dir, sub, normalizer=normalizer, augment=train, domains=domains),
            batch_size=batch_size, shuffle=train, drop_last=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            worker_init_fn=WorkerInitFn(seed), generator=get_generator(seed),
            persistent_workers=num_workers > 0,
        )
    return loaders, table
