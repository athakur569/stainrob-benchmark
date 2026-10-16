"""Freeze and verify a tiled dataset (any dataset) with SHA256 fingerprints.

freeze_dataset() writes <out_dir>/freeze.json covering every tile file, the manifest, the
split input, the tiling config, the source metadata and all split artefacts
(patient_universe.txt, split_metadata.json, patient_splits.csv, ...). It refuses to
overwrite an existing freeze.

verify_freeze() recomputes the fingerprints and raises FreezeMismatchError on any change.
Training calls it at startup (fast mode: everything except per-tile re-hashing, plus the tile
count); full mode re-hashes every tile.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SPLIT_FILES = ("patient_universe.txt", "split_metadata.json", "patient_splits.csv",
               "slide_to_patient_mapping.csv", "train.csv", "val.csv", "test.csv",
               "train_patients.csv", "val_patients.csv", "test_patients.csv")


class FreezeMismatchError(RuntimeError):
    """The data on disk no longer matches its freeze record."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tile_hash(out_dir: Path, manifest: pd.DataFrame, progress: bool) -> tuple[str, int]:
    rows = manifest[["tile_id", "path"]].sort_values("tile_id").itertuples(index=False)
    if progress:
        from tqdm import tqdm
        rows = tqdm(rows, total=len(manifest), unit="tile", desc="hashing tiles", ascii=True, ncols=100, mininterval=2)
    agg = hashlib.sha256()
    for tid, rel in rows:
        agg.update(f"{tid}:{sha256_file(out_dir / rel)}\n".encode())
    return agg.hexdigest(), len(manifest)


def _parts(out_dir: Path, config_path: Path, metadata_path: Path, splits_dir: Path) -> dict[str, str]:
    parts = {"config": sha256_file(config_path), "metadata": sha256_file(metadata_path),
             "manifest": sha256_file(out_dir / "manifest.csv"),
             "split_input": sha256_file(out_dir / "split_input.csv")}
    for name in SPLIT_FILES:
        p = splits_dir / name
        if not p.exists():
            raise FileNotFoundError(f"split artefact missing: {p}")
        parts[f"splits/{name}"] = sha256_file(p)
    return parts


def _git_state(cwd: Path) -> tuple[str, bool]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd).stdout.strip())
        return head or "unknown", dirty
    except OSError:
        return "unknown", True


def freeze_dataset(out_dir: str | Path, config_path: str | Path, metadata_path: str | Path,
                   splits_dir: str | Path, repo_dir: str | Path = ".", progress: bool = True) -> dict:
    out_dir, splits_dir = Path(out_dir), Path(splits_dir)
    record_path = out_dir / "freeze.json"
    if record_path.exists():
        raise FileExistsError(f"{record_path} exists: dataset is frozen. Delete it deliberately to re-freeze.")
    manifest = pd.read_csv(out_dir / "manifest.csv")
    manifest = manifest[manifest["qc_pass"]]
    tiles_hash, n = _tile_hash(out_dir, manifest, progress)
    parts = {"tiles": tiles_hash, **_parts(out_dir, Path(config_path), Path(metadata_path), splits_dir)}
    master = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    split_meta = json.loads((splits_dir / "split_metadata.json").read_text())
    head, dirty = _git_state(Path(repo_dir))
    record = {
        "master_sha256": master, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_tiles": n, "parts": parts,
        "paths": {"config": str(config_path), "metadata": str(metadata_path), "splits_dir": str(splits_dir)},
        "split": {k: split_meta.get(k) for k in ("project_id", "split_version", "seed", "patient_universe_hash",
                                                  "checksum", "group_column", "patient_label_rule")},
        "tiles_by_center_label": {f"{c}_{l}": int(v) for (c, l), v in manifest.groupby(["center", "label"]).size().items()},
        "git_head": head, "git_dirty": dirty,
    }
    record_path.write_text(json.dumps(record, indent=2))
    return record


def verify_freeze(out_dir: str | Path, full: bool = False, progress: bool = False) -> dict:
    out_dir = Path(out_dir)
    rec = json.loads((out_dir / "freeze.json").read_text())
    p = rec["paths"]
    now = _parts(out_dir, Path(p["config"]), Path(p["metadata"]), Path(p["splits_dir"]))
    changed = [k for k, v in now.items() if rec["parts"].get(k) != v]
    manifest = pd.read_csv(out_dir / "manifest.csv")
    manifest = manifest[manifest["qc_pass"]]
    if len(manifest) != rec["n_tiles"]:
        changed.append(f"tile count {len(manifest)} != {rec['n_tiles']}")
    if full:
        tiles_hash, _ = _tile_hash(out_dir, manifest, progress)
        if tiles_hash != rec["parts"]["tiles"]:
            changed.append("tiles")
    if changed:
        raise FreezeMismatchError(f"frozen data changed: {changed}")
    return rec
