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
| P1 | ISS-046, ISS-049 | Patient-split null check (P1–P1d: clusters × projection source) | training tiles only | run (training tiles only) |
| P2 | ISS-044 | Consistent-direction stain family (`stain_batch`) | test_in tiles, shifted | awaiting decision |
| P3 | ISS-045, ISS-046 | Held-out reference and patient clusters for every tier | test_in, test_cross | awaiting decision |
| P4 | ISS-048 | ΔG without individual tumour patients | existing test predictions | awaiting pathologist review |
| P5 | run log (synthetic arm) | Why stain normalisation adds fragility to JPEG and striping | test_in tiles, shifted | awaiting decision |

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

**P1b result (200 splits × 5 folds, patient-split nulls, α = 0.05).** False-alarm rate with patient clusters, mean
(range over folds), reference-fitted → pooled projection: Tier 1a 0.026 (0.015–0.035) → 0.023 (0.010–0.035);
DenseNet-121 0.058 (0.035–0.080) → 0.032 (0.010–0.055); ResNet-50 0.543 (0.42–0.595) → 0.042 (0.015–0.060).
With slide clusters, pooled: 0.21–0.43 for every Tier 1 row; slide clusters remain wrong whatever the projection.
Pooled projection with patient clusters holds α for every tier.

**P1c — added before running (raised by career-ops):** a third arm, `--fit-on external`: the projection is fitted on a
set disjoint from both batches, so it does not depend on the split at all. To stay on training tiles, each null split
divides the training patients into thirds (projection / reference / deployment). In P3 the external set is the fold's
training tiles, disjoint from `test_in` and `test_cross`. Principle behind P1b and P1c: a permutation test is exact
only if every quantity in its statistic is unchanged by permuting the batch labels; a projection fitted on one batch
is not. The arm that holds α is chosen by power under shift in P3, not by false-alarm rate alone.

**P1c result (thirds, patient clusters, mean (range) over folds):** Tier 1a 0.016 (0.010–0.025); DenseNet-121 0.023
(0.010–0.050); ResNet-50 0.021 (0.010–0.040). Holds α for every tier. Lower than P1b, but on thirds rather than halves;
P1d gives the matched contrast.

**P1d — added before running (raised by career-ops):** P1c's thirds give each side about a third of the patients, P1/P1b
halves. For a matched contrast, the reference-fitted and pooled arms are rerun on the **same** thirds as P1c (`--thirds`;
the projection third unused), so the three arms differ only in where the projection is fitted. Headline rates stay
those of P1/P1b (halves); P1d is for the contrast.

**P1d result (same thirds for all arms, patient clusters, mean (range) over folds):**

| Tier | Reference-fitted | Pooled | External |
|---|---|---|---|
| 1a pixel statistics | 0.023 (0.010–0.045) | 0.018 (0.010–0.030) | 0.016 (0.010–0.025) |
| 1b DenseNet-121 | 0.053 (0.030–0.090) | 0.024 (0.005–0.040) | 0.023 (0.010–0.050) |
| 1b ResNet-50 | 0.446 (0.345–0.530) | 0.026 (0.010–0.045) | 0.021 (0.010–0.040) |

On matched patients, pooled and external are indistinguishable under the null; the reference-fitted projection fails
for ResNet-50 at thirds as at halves. The choice between pooled and external is left to power under shift (P3).

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
3. **Tier 1 projection fitted on pooled or external data** (ISS-049), whichever holds α in P1b/P1c; if both do,
   both are reported under shift and the one with more power is named.

Everything else as pre-registered (D2: standardisation and PCA-50 fitted on the reference, at most 8 tiles per slide,
α = 0.05, AND rule Tier 1b AND Tier 2, seed majority D6). Reported: reference size (patients, tiles) per fold;
pre-registered and P3 results side by side, neither called primary for Tier 2.
Every power figure is reported beside the same arm's realised false-alarm rate from P1–P1d, so a reader can tell a
weak signal from a conservative alarm rule (all valid arms run below α, 0.016–0.042, because the rule splits α across
two correlated statistics). A combined test over the two statistics is a possible later change, decided after P3.

Independence of the two Tier 1 faults (P1–P1d): slide clusters false-alarm at 0.21–0.43 whatever the projection, and
with patient clusters the reference-fitted projection fails in high dimension (ResNet-50 0.45–0.54). For the
high-dimensional tiers both fixes are needed; for Tier 1a, which has no projection, patient clusters alone suffice.

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

## P5 · Why stain normalisation adds fragility to JPEG and striping

Descriptive finding (run log, synthetic arm): stain normalisation removes stain harm but loses more accuracy than the
baseline under JPEG and striping (E3 striping ΔE 25: 0.45–0.65 across folds vs E1 0.18–0.28; E2's range overlaps E1's).
Prior work found normalisation adds little over colour augmentation (Tellez et al., *Med Image Anal* 2019; as
summarised by career-ops); here colour augmentation is deliberately absent (DR-010), which exposes the trade-off.

Hypothesis: the normalisers read the incoming tile — Macenko estimates its stain vectors and concentrations, Reinhard
its colour mean and SD — so a shift that corrupts those statistics is amplified by the step meant to remove variation.
The reference (DR-013) is fixed and is not the issue.

Test: for E2 and E3, at the JPEG and striping doses, score `test_in` twice — (a) as pre-registered (statistics
estimated from the shifted tile); (b) with the tile's statistics estimated from its **clean** version and that transform
applied to the shifted tile. ΔG(a) − ΔG(b) is the harm attributable to corrupted estimation. Same models, tiles,
bootstrap. Descriptive; no between-condition test.
