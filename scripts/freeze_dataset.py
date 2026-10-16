"""Freeze (or verify) a tiled dataset.

    PYTHONPATH=src python scripts/freeze_dataset.py --config configs/tiling_<dataset>.yaml
    PYTHONPATH=src python scripts/freeze_dataset.py --config configs/tiling_<dataset>.yaml --verify [--full]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from stain_benchmark.data.freeze import freeze_dataset, verify_freeze


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--verify", action="store_true", help="Verify an existing freeze instead of creating one")
    ap.add_argument("--full", action="store_true", help="With --verify: re-hash every tile")
    args = ap.parse_args()
    conf = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(conf["out_dir"])
    if args.verify:
        rec = verify_freeze(out_dir, full=args.full, progress=True)
        print(f"VERIFIED ({'full' if args.full else 'fast'}): master SHA256 {rec['master_sha256']}")
        return
    rec = freeze_dataset(out_dir, args.config, conf["metadata"], out_dir / "splits",
                         repo_dir=Path(__file__).resolve().parent.parent)
    print(f"FROZEN: {rec['n_tiles']} tiles, master SHA256 {rec['master_sha256']}")
    print(f"split: {rec['split']}")
    print(f"git: {rec['git_head'][:12]} dirty={rec['git_dirty']}")


if __name__ == "__main__":
    main()
