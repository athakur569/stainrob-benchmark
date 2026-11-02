# Post-hoc analyses — specified before they are run

The pre-registered analysis (`docs/analysis_plan.md`, commit `927e44b`) is run and reported as written. The
end-to-end check on fold 0 (E1) exposed three design problems (research log ISS-044, ISS-045, ISS-046). The
analyses below address them. They are **post hoc**: they were designed after seeing fold-0 monitor results.
To limit the freedom that gives, each is written down and committed here before it touches test tiles, and
each is reported **beside** the pre-registered result, never in place of it.

Status column: *specified* = written, not run · *run (training tiles only)* = needs no test data ·
*awaiting decision* = needs test tiles; runs only once Abhi approves.

| ID | Addresses | What | Data it reads | Status |
|---|---|---|---|---|
| P1 | ISS-046, ISS-049 | Patient-split null check (reference-fitted and pooled projection) | training tiles only | run (training tiles only) |
| P2 | ISS-044 | Consistent-direction stain family (`stain_batch`) | test_in tiles, shifted | awaiting decision |
| P3 | ISS-045, ISS-046 | Held-out reference and patient clusters for every tier | test_in, test_cross | awaiting decision |
| P4 | ISS-048 | ΔG without individual tumour patients | existing test predictions | awaiting pathologist review |

## P1 · Patient-split null check

The pre-registered null check (§9.7) halves each fold's training **slides**. A CAMELYON17 patient has five slides,
so both halves contain the same patients; deployment brings new patients. P1 halves the training **patients**
instead (200 random halves per fold, seeded by fold and split index), and runs every tier on the same splits twice:
with slide clusters (as pre-registered, D2) and with patient clusters. Tiers: 1a, 1b DenseNet-121, 1b ResNet-50,
2 (E1, seed 42). Outcome: realised false-alarm rate against α = 0.05.
Script: `scripts/null_check_posthoc.py`.

**Result (reference-fitted projection, as pre-registered), false-alarm rate across the five folds:**

| Tier | Slide clusters | Patient clusters |
|---|---|---|
| 1a pixel statistics | 0.285–0.375 | 0.015–0.035 |
| 1b DenseNet-121 | 0.40–0.55 | 0.035–0.08 |
| 1b ResNet-50 | 0.94–0.99 | 0.42–0.595 |
| 2 Mahalanobis | 0.000–0.010 | 0.000–0.010 |

Patient clusters fix Tier 1a. They do not fix ResNet-50, which a cluster permutation on a random patient split
should not allow (the observed split is itself one of the permutations). The cause is upstream of the test (ISS-049):
the standardisation and 50 principal components are fitted on the reference half only, so in 2,048 dimensions they
capture that half's own patients, and the halves become distinguishable. Check on fold 1, 40 splits, patient clusters:
ResNet-50 0.50 with reference-fitted projection, 0.00 with projection fitted on both halves pooled; DenseNet-121 0.05
and 0.00.

**P1b — added before running:** the same null check (Tier 1 rows, 200 splits, all folds) with `--fit-on pooled`:
standardisation and components fitted on reference and deployment together. This is label-free and available at
deployment, and it keeps the two batches symmetric.

## P2 · Consistent-direction stain family

The pre-registered stain family draws a direction per tile, so a batch's changes cancel (ISS-044). `stain_batch`
increases both haematoxylin and eosin by the same factor for every tile: a laboratory batch that drifts one way.
Doses: ΔE 2, 4, 6, 8, 10, 12, fixed on the pre-registered training-split calibration sample
(`configs/shift_doses_camelyon17_posthoc_stain_batch.json`). Everything else follows the synthetic arm of the plan:
the same 60 models, the same test_in tiles, the same ΔG and monitor computations.
Commands: `scripts/score_synthetic.py --doses <that file> --tag stain_batch`, and Tier 1 extraction for those doses.
Reported: ΔG and every tier's signal per dose, next to the pre-registered `stain` family. H3 is **not** re-tested
with it; it is a description of how the monitor responds to a batch-level stain drift.

## P3 · Held-out reference, patient clusters

Two changes to the monitor, made together because both concern what the reference is and how it is resampled:

1. **Reference = held-out in-domain tiles** for every tier (the pre-registered reference is the training tiles, which
   the classifier has nearly memorised; ISS-045).
   - Real arm: reference = all `test_in` tiles (unshifted); deployment = `test_cross`.
   - Synthetic arm: the reference cannot be the same tiles that are shifted. `test_in` patients are halved,
     stratified by patient label: half A (unshifted) is the reference, half B (shifted at each dose) is deployed.
     10 random halvings per fold (seeds 0–9); reported: median signal and the fraction of halvings that alarm.
     The unshifted B-vs-A comparison is the direct false-alarm check for this design.
   - ΔG is unchanged (all `test_in`), so the outcome is identical to the pre-registered one; only the predictors change.
2. **Clusters = patients** instead of slides, for permutation (Tier 1) and cross-fitting/bootstrap (Tier 2): P1 shows
   slide clusters exceed α by 6–20× on patient splits.
3. **Tier 1 projection fitted on reference and deployment pooled** (ISS-049), if P1b shows it holds α.

Everything else as pre-registered (D2: standardisation and PCA-50 fitted on the reference, at most 8 tiles per slide,
α = 0.05, AND rule Tier 1b AND Tier 2, seed majority D6). Reported: reference size (patients, tiles) per fold;
pre-registered and P3 results side by side, neither called primary for Tier 2.

## P4 · ΔG without individual tumour patients

Some tumour patients are missed in every fold they appear in (ISS-048); with 4–7 tumour patients per held-out hospital
one of them can change a fold's label. Two computations on the predictions already made (no new scoring):

1. **Leave-one-tumour-patient-out ΔG**, every fold × condition: the seed-averaged ΔG recomputed with each tumour patient
   removed in turn from `test_in` and from `test_cross` (the same patient-clustered bootstrap); reported as the range.
   This needs no judgement about which patients to question.
   Reported as its own line: hospital 2's ΔG (the only harm fold) without patients 042 and 052, singly and together.
2. **With and without the patients a pathologist flags** as having outlines that enclose little tumour, after the review
   of `docs/figures/missed_patients_e1.png` (and further tiles if needed). Both numbers are reported; no patient is
   removed from the primary analysis.

Patient roles, for reading the results: patient 042 is a validation-split patient of hospital 2 — never trained on; it
is in `test_in` when hospital 2 is a training hospital and in `test_cross` when it is held out. Patients 052 (hospital 2)
and 096 (hospital 4) are training-split patients: trained on in the folds where their hospital is a training hospital,
scored only when it is held out.
