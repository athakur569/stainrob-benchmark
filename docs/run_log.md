# Run log — CAMELYON17 rehearsal

Results in the order they were obtained, after the analysis plan was committed (`927e44b`,
2 Nov 2026 11:50 EST). Intervals are patient-clustered 95% bootstrap intervals (2,000 resamples).

## 2 Nov 2026 · Fold 0 (held-out hospital 0), E1 baseline, seeds 42 / 99 / 123

| Seed | test_in AUROC (14 tumour patients) | test_cross AUROC (5 tumour patients) | ΔG |
|---|---|---|---|
| 42 | 0.982 [0.957, 0.997] | 0.993 [0.987, 0.996] | −0.011 [−0.036, +0.007] |
| 99 | 0.956 [0.928, 0.996] | 0.994 [0.990, 0.996] | −0.038 [−0.066, +0.003] |
| 123 | 0.984 [0.964, 0.998] | 0.992 [0.983, 0.996] | −0.009 [−0.028, +0.009] |

- **Ceiling contingency (plan §11): did not fire — borderline.** In-domain AUROC averaged over seeds
  0.974 against the threshold 0.98; two of the three seeds individually exceed 0.98 (0.982, 0.984);
  the rule uses the seed average, so it was applied as written.
- **No measurable generalisation gap to hospital 0:** every ΔG interval includes 0.
- **Why the held-out interval is narrower than the in-domain one despite fewer tumour patients
  (checked):** heterogeneity, not an artefact. In `test_in`, two large tumour patients are poorly
  ranked (patient 042, 179 tiles, 65% scored below 0.5; patient 092, 357 tiles, 53%), so resamples that
  repeat them move AUROC a lot; leaving one tumour patient out gives 0.978–0.993. In `test_cross` all
  five tumour patients are ranked well; leaving one out gives 0.990–0.994. Caveat: with 4–7 tumour
  patients, a percentile bootstrap tends to understate uncertainty; held-out-hospital intervals should
  be read as lower bounds on the true uncertainty.
- **Provenance:** each run's `summary.json` records `git_head 4f0280d` (stamped at the end of the run);
  seed 42 started under `927e44b`. Between the two commits: `docs/deviations.md` (new),
  `scripts/run_matrix.py` and `scripts/score_synthetic.py` (new), `scripts/extract_tier1.py` (a `val-test`
  subset added), and in `training/run.py` only `score_tiles` gained an optional feature output. The
  training code path (`train_run`, data loading, model, optimiser) did not change. Every run records the
  plan commit `927e44b`.
- **To check across folds — possible hidden stratification:** two in-domain tumour patients are largely
  missed by the fold-0 models (patient 092: 357 tiles, 53% scored below 0.5; patient 042: 179 tiles, 65%).
  If the same patients, or patients like them, are missed in other folds, a subgroup the model handles badly
  is hiding inside a high aggregate AUROC. Per-patient tumour-tile scores will be tabulated for every fold.
- **For the paper:** every cross-hospital interval rests on 4–7 tumour patients, where a percentile bootstrap
  tends to be too narrow; all such intervals are reported as lower bounds on the uncertainty.
