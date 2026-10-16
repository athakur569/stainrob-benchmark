"""The committed CAMELYON17 record must match the hashes in its freeze.json.

freeze.json was written by scripts/freeze_dataset.py when the tile set was frozen. These
tests re-hash the small files committed alongside it, so anyone with the repository can
check that the splits, metadata and tile manifest are the ones that were frozen. The tile
images themselves are not committed; `freeze_dataset.py --verify --full` checks them.
"""

import gzip
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "data" / "camelyon17"
FREEZE = json.loads((RECORD / "freeze.json").read_text())
PARTS = FREEZE["parts"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_master_hash_matches_parts():
    master = _sha256(json.dumps(PARTS, sort_keys=True).encode())
    assert master == FREEZE["master_sha256"]


def test_config_matches_frozen_config():
    config = ROOT / FREEZE["paths"]["config"]
    assert _sha256(config.read_bytes()) == PARTS["config"]


@pytest.mark.parametrize("part", sorted(k for k in PARTS if k.startswith("splits/")))
def test_split_files_match(part):
    assert _sha256((RECORD / part).read_bytes()) == PARTS[part]


def test_metadata_matches():
    assert _sha256((RECORD / "metadata.csv").read_bytes()) == PARTS["metadata"]


def test_split_input_matches():
    assert _sha256((RECORD / "split_input.csv").read_bytes()) == PARTS["split_input"]


def test_tile_manifest_matches():
    raw = gzip.decompress((RECORD / "manifest.csv.gz").read_bytes())
    assert _sha256(raw) == PARTS["manifest"]
