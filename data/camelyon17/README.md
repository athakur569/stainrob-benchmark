# CAMELYON17 frozen tile set: record

The tiles themselves (11,763 PNGs) are not in the repository; these small files are the
record of exactly which tiles were frozen and how they were split.

| File | What it is |
|---|---|
| `freeze.json` | SHA256 of every part, the master hash, split seed and checksum, tile counts per center and class |
| `metadata.csv` | slide table built from CAMELYON17 `stages.csv` and the tumour outlines |
| `manifest.csv.gz` | one row per tile: position, label, slide, patient, center, QC scores (gzip of the hashed `manifest.csv`) |
| `split_input.csv` | one row per slide with QC-passing tiles; input to `scripts/create_splits.py` |
| `splits/` | patient universe, split metadata, patient and slide assignments |

`tests/test_frozen_record.py` checks every file against `freeze.json`. `freeze.json` was
written before the code was committed, so its `git_head` names a commit that is no longer in
the history and `git_dirty` is `true`; the tiling code and config that produced it are the
ones committed on 2026-10-16 (the config hash matches).
