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

## 2 Nov 2026 · Real arm, all five folds × E1, E2, E3, E5 (60 runs)

Seed-averaged ΔG(AUROC) = AUROC(`test_in`) − AUROC(`test_cross`), 95% patient-clustered bootstrap interval,
labelled by the D5 rule (fixed before folds 1–4 were scored).

| Held-out hospital | Tumour patients (in / cross) | E1 | E2 Macenko | E3 Reinhard | E5 DANN |
|---|---|---|---|---|---|
| 0 | 14 / 5 | −0.019 [−0.037, +0.006] clean | −0.022 [−0.070, +0.001] clean | −0.036 [−0.103, +0.000] clean | −0.017 [−0.043, −0.002] clean |
| 1 | 12 / 7 | −0.001 [−0.030, +0.015] clean | +0.038 [−0.002, +0.124] inconclusive | −0.010 [−0.052, +0.014] clean | −0.001 [−0.024, +0.012] clean |
| 2 | 14 / 7 | **+0.043 [+0.006, +0.136] harm** | **+0.048 [+0.007, +0.159] harm** | **+0.041 [+0.002, +0.148] harm** | **+0.024 [+0.010, +0.046] harm** |
| 3 | 13 / 6 | −0.029 [−0.059, +0.007] clean | −0.018 [−0.067, +0.007] clean | −0.020 [−0.076, +0.008] clean | −0.012 [−0.041, +0.005] clean |
| 4 | 15 / 4 | +0.008 [−0.043, +0.054] inconclusive | −0.011 [−0.084, +0.044] inconclusive | +0.015 [−0.089, +0.104] inconclusive | −0.004 [−0.056, +0.036] inconclusive |

- **Hospital 2 is the one measurable gap, and every condition shows it.** This agrees with WILDS (Koh et al. 2021),
  which holds out the same hospital as its test site because models drop there. DANN has the smallest gap
  (0.024) and the narrowest interval. Comparing conditions is H-level work for the full analysis, not this table.
- **In-domain AUROC is high everywhere** (0.96–0.997). The ceiling contingency (§11) is checked on fold 0 only, by
  design, and did not fire there (0.974); in folds 1 and 2 E1 averages 0.985 and 0.996, so the real arm has little
  headroom in those folds. This supports the plan's choice to make the synthetic arm the controlled evidence.
- **Negative ΔG (the held-out hospital scores higher than in-domain) is common.** It reflects which patients each
  test set happens to contain, not a benefit of being out of domain (next point).
- **Per-patient check (open question from fold 0), E1:** the same patients are missed whichever fold they fall in.
  Patient 042 (hospital 2, 179 tumour tiles) has 62–87% of its tumour tiles scored below 0.5 in every fold,
  in-domain or cross-hospital. Hospital 4's patient 096 (500 tumour tiles, 89% missed) is why fold 4 is
  inconclusive rather than clean. Hospital 2's patient 052 (321 tiles, 70% missed) contributes to fold 2's gap.
  So part of each "hospital" effect is a few hard patients. Figure: `docs/figures/missed_patients_e1.png`.
- **What the missed tiles look like (descriptive; pathologist review pending):** patients 042 and 096 —
  fibrous stroma with scattered single atypical cells, inside the outline of the metastasis; patient 052 —
  dense lymphoid tissue inside the outline, suggesting a coarse outline (label noise) rather than a model error;
  patient 092 — pale tumour cells among lymphocytes, missed with low confidence.
- **Annotation coverage:** tumour tiles at the edge of an outline (50–75% inside) are missed more often (37.5%)
  than tiles fully inside (20.7%), a modest partial-label effect. The four most-missed patients' tiles are almost
  entirely inside the outline (median 0.99–1.0), so edge effects do not explain them. Several tumour patients
  contribute only 1–7 tiles, all from micrometastasis or ITC slides.
