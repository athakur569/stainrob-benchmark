#!/usr/bin/env python3
"""POST HOC (P4, ISS-048): the real-arm ΔG without individual tumour patients.

Part 1 (selection-free): for every fold × condition, the seed-averaged ΔG(AUROC) recomputed with each tumour patient
removed in turn from test_in and from test_cross (all of that patient's tiles) — reported as the range.
Named lines: hospital 2 (fold 2) without patients 042 and 052, singly and together, each with its patient-clustered
bootstrap interval and D5 label. Part 2 (patients a pathologist flags) waits for the review. No patient is removed
from the primary analysis. Specification: docs/posthoc_analyses.md, P4. Reads existing predictions only.

    python scripts/posthoc_patients.py --runs /media/abhi/camelyon17/derived/runs_v1 \\
        --out /media/abhi/camelyon17/derived/posthoc/p4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse import label_fold  # noqa: E402

from stain_benchmark.evaluation.bootstrap import delta_g_seeds_interval  # noqa: E402
from stain_benchmark.training.run import require_committed_plan  # noqa: E402

CONDITIONS, FOLDS, SEEDS = ("E1", "E2", "E3", "E5"), ("0", "1", "2", "3", "4"), (42, 99, 123)
NAMED = {"2": [("patient_042",), ("patient_052",), ("patient_042", "patient_052")]}


def point_dg(pin: list[pd.DataFrame], pcr: list[pd.DataFrame]) -> float:
    return float(np.mean([roc_auc_score(a.label, a.p_tumour) - roc_auc_score(b.label, b.p_tumour)
                          for a, b in zip(pin, pcr)]))


def without(preds: list[pd.DataFrame], patients) -> list[pd.DataFrame]:
    return [p[~p.patient_id.isin(patients)].reset_index(drop=True) for p in preds]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    plan = require_committed_plan()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loo, named, summary = [], [], []
    for f in FOLDS:
        for c in CONDITIONS:
            pin = [pd.read_csv(runs / c / f"fold{f}" / f"seed{s}" / "predictions_test_in.csv") for s in SEEDS]
            pcr = [pd.read_csv(runs / c / f"fold{f}" / f"seed{s}" / "predictions_test_cross.csv") for s in SEEDS]
            base = point_dg(pin, pcr)
            rows = []
            for role, preds in (("test_in", pin), ("test_cross", pcr)):
                for pid in sorted(preds[0].loc[preds[0].label == 1, "patient_id"].unique()):
                    a, b = (without(pin, [pid]), pcr) if role == "test_in" else (pin, without(pcr, [pid]))
                    rows.append({"fold": f, "condition": c, "role": role, "patient_removed": pid,
                                 "delta_g_without": point_dg(a, b), "delta_g_all": base})
            loo += rows
            d = pd.DataFrame(rows)
            summary.append({"fold": f, "condition": c, "delta_g_all": base, "loo_min": d.delta_g_without.min(),
                            "loo_max": d.delta_g_without.max(),
                            "most_influential": d.loc[(d.delta_g_without - base).abs().idxmax(), "patient_removed"]})
            for group in NAMED.get(f, []):
                iv = delta_g_seeds_interval(without(pin, group), without(pcr, group), args.boot, seed=0)
                named.append({"fold": f, "condition": c, "patients_removed": "+".join(group),
                              "delta_g": iv.estimate, "low": iv.low, "high": iv.high, "label_D5": label_fold(iv)})
            print(f"fold {f} {c}: ΔG {base:+.3f}, without one tumour patient {summary[-1]['loo_min']:+.3f} "
                  f"to {summary[-1]['loo_max']:+.3f}", flush=True)
    pd.DataFrame(loo).to_csv(out / "leave_one_patient_out.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "leave_one_patient_out_summary.csv", index=False)
    pd.DataFrame(named).to_csv(out / "hospital2_named.csv", index=False)
    (out / "provenance.txt").write_text(f"analysis plan {plan}; post-hoc P4 part 1; {args.boot} resamples\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
