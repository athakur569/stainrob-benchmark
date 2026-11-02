#!/usr/bin/env python3
"""Run every (condition, held-out centre, seed) of the pre-registered design, resumably.

Order: fold by fold (0..4), and within a fold E1, E2, E3, E5, each with seeds 42, 99, 123.
A run whose summary.json exists is skipped, so the matrix can be stopped and restarted. Each run is
its own process (scripts/train.py) launched through claude-run, so it is visible in claude-watch.

    python scripts/run_matrix.py --tiles /media/abhi/camelyon17/derived/tiles_v1 \\
        --runs /media/abhi/camelyon17/derived/runs_v1
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

CONDITIONS = ("E1", "E2", "E3", "E5")
FOLDS = ("0", "1", "2", "3", "4")
SEEDS = (42, 99, 123)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiles", required=True)
    ap.add_argument("--runs", required=True)
    args = ap.parse_args()
    runs = Path(args.runs)
    todo = [(c, f, s) for f in FOLDS for c in CONDITIONS for s in SEEDS
            if not (runs / c / f"fold{f}" / f"seed{s}" / "summary.json").exists()]
    total = len(CONDITIONS) * len(FOLDS) * len(SEEDS)
    print(f"{total - len(todo)} of {total} runs already finished; {len(todo)} to go", flush=True)
    for k, (c, f, s) in enumerate(todo, 1):
        t0 = time.time()
        cmd = [str(Path.home() / ".local/bin/claude-run"), "--gpu", "--name", f"{c}-fold{f}-seed{s}", "--",
               "env", "PYTHONPATH=src", sys.executable, "scripts/train.py", "--tiles", args.tiles,
               "--runs", args.runs, "--condition", c, "--holdout", f, "--seed", str(s)]
        rc = subprocess.run(cmd, capture_output=True, text=True).returncode
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        print(f"[{k}/{len(todo)}] {c} fold {f} seed {s}: {status} · {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            sys.exit(f"stopping at the first failure: {c} fold {f} seed {s}")
    print("matrix complete", flush=True)


if __name__ == "__main__":
    main()
