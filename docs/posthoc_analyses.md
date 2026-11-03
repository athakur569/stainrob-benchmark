# Post-hoc analyses — specified before they are run

The pre-registered analysis (`docs/analysis_plan.md`, commit `927e44b`) is run and reported as written. The
end-to-end check on fold 0 (E1) exposed three design problems (research log ISS-044, ISS-045, ISS-046). The
analyses below address them. They are **post hoc**: they were designed after seeing fold-0 monitor results.
To limit the freedom that gives, each is written down and committed here before it touches test tiles, and
each is reported **beside** the pre-registered result, never in place of it.

Status column: *specified* = written, not run · *run (training tiles only)* = needs no test data ·
*awaiting decision* = needs test tiles; runs only once Abhi approves. P2–P5 were approved by Abhi on 3 Nov 2026.

| ID | Addresses | What | Data it reads | Status |
|---|---|---|---|---|
| P1 | ISS-046, ISS-049 | Patient-split null check (P1–P1d: clusters × projection source) | training tiles only | run (training tiles only) |
| P2 | ISS-044 | Consistent-direction stain family (`stain_batch`) | test_in tiles, shifted | approved 3 Nov; running |
| P3 | ISS-045, ISS-046 | Held-out reference and patient clusters for every tier | test_in, test_cross | approved 3 Nov |
| P4 | ISS-048 | ΔG without individual tumour patients | existing test predictions | part 1 approved 3 Nov; part 2 awaits pathologist review |
| P5 | run log (synthetic arm) | Why stain normalisation adds fragility to JPEG and striping | test_in tiles, shifted | approved 3 Nov |
| P5b | P5 | Does the normaliser itself amplify artefacts? (no model, no labels) | training-split calibration tiles only | run — estimation corruption not supported |

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

**Reading rule for P3's real arm (added 3 Nov before P3 ran; raised by career-ops).** The two hospitals with the
largest drops (2 and 4) are also the two with different scanners, and their drops are carried by a few patients whose
misses persist even when their hospital is in training (P4; patient 042). So a site alarm is credited as tracking harm
only if that site's harm survives removing its influential patients (P4: 042/052 at hospital 2, 096/092 at hospital 4).
If the harm vanishes without them while the alarm remains, the alarm is reported as tracking appearance, not harm.
With five hospitals this is a description, not a test.

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

**P4 part 1 result (3 Nov).** Leaving out one tumour patient at a time moves fold ΔG within these ranges (seed-averaged
points): hospital 0 −0.050 to −0.000; hospital 1 −0.017 to +0.067; hospital 2 +0.008 to +0.076; hospital 3 −0.041 to
+0.002; hospital 4 −0.032 to +0.052. Patient 042 is the most influential patient in 15 of 20 fold × condition cells.
Hospital 2, the only fold with harm detected, named lines (95% patient-bootstrap interval, D5 label):

| Condition | All patients | Without 042 | Without 052 | Without both |
|---|---|---|---|---|
| E1 | +0.043 harm | +0.014 [+0.002, +0.035] harm | +0.048 [+0.003, +0.186] harm | +0.008 [−0.000, +0.073] inconclusive |
| E2 | +0.048 harm | +0.013 [+0.000, +0.025] harm | +0.058 [+0.001, +0.216] harm | +0.009 [−0.003, +0.042] inconclusive |
| E3 | +0.041 harm | +0.008 [−0.001, +0.021] clean | +0.049 [−0.001, +0.198] inconclusive | +0.004 [−0.004, +0.040] inconclusive |
| E5 | +0.024 harm | +0.024 [+0.007, +0.051] harm | +0.014 [+0.007, +0.032] harm | +0.012 [+0.003, +0.065] harm |

For E1–E3 most of hospital 2's gap is patient 042 (67–80% of it) and, with 052 also removed, the gap is no longer detectable.
For E5 the gap survives both removals. Part 2 (patients a pathologist flags) awaits the review.

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

Arm (b) is an **oracle** condition: no laboratory has the clean version of a damaged tile, and applying a transform
estimated on one image to another is a counterfactual, not a deployable configuration. It is used only to separate
mechanisms.

Decomposition (raised by career-ops; E1's values already exist, so this is arithmetic):
- ΔG(E1) — the artefact's damage with no normaliser;
- ΔG(a) − ΔG(b) — **estimation corruption** (same model; only the estimate differs): fixable in principle (robust
  estimation, estimating on QC-passed tiles);
- ΔG(b) − ΔG(E1) — **transform amplification plus model difference**: E2/E3 models were trained on normalised tiles
  and E1's on raw tiles, so this term mixes what the transform does to the artefact with how the two models differ.
  It cannot be read as pure amplification, and is reported with that label.

### P5b · Amplification measured on the normaliser itself (raised by career-ops)

Whether a normaliser magnifies an artefact is a property of the function N, not of any classifier, so it can be measured
without a trained model, without labels, and on the **training-split calibration sample** already used to fix the dose
table (100 tiles; no test tile is read). For each fold's E2 and E3 normaliser (DR-013 references), each family and dose,
and each tile x with shifted version x′, three distances, in pixels (RMS, RGB) and in the frozen DenseNet-121 feature
space of Tier 1b (Euclidean):

- d_raw = d(x, x′) — the artefact with no normaliser;
- d_oracle = d(N_x(x), N_x(x′)) — the transform with its statistics estimated on the clean tile (oracle; counterfactual);
- d_norm = d(N_x(x), N_x′(x′)) — the pipeline as deployed: statistics estimated on each tile as it arrives.

Reported per family and dose (median over tiles, folds pooled, fold range shown): d_oracle / d_raw (**transform
amplification** with a correct estimate) and d_norm / d_oracle (**estimation corruption**). Ratios above 1 that rise
with dose are evidence of amplification at that stage. Limitation: a larger feature-space distance is not task harm by
itself; it corroborates, and P5's model arms give the ΔG side. The two spaces (raw vs normalised) differ in distribution,
so d_oracle / d_raw compares distances across spaces; the stain family, which normalisation should shrink, is the
built-in check that the ratio moves in the expected direction.

**P5b result (100 calibration tiles × 5 folds; median over tiles, range over folds).** Built-in check passes: for
stain, the deployed normaliser removes most of the shift (d_norm / d_oracle: feature 0.51–0.77, pixel 0.25–0.48).
- **Estimation corruption is negligible for the families where normalisation adds fragility:** d_norm / d_oracle in
  feature space 0.97–1.05 (JPEG) and 0.90–1.14 (striping), for both normalisers. The hypothesis that P5 was written to
  test — the normaliser amplifies harm because it reads corrupted statistics — is **not supported** at the
  representation level.
- **Transform amplification is modest:** JPEG 1.04–1.40 (E3) and 1.02–1.20 (E2) in feature space, largest at mild
  doses (pixels 1.2–1.95); striping 0.97–1.11; blur 0.92–1.07; clipped 1.01–1.07.
- So the large striping fragility of E3 (ΔG 0.57 vs 0.24 at ΔE 25) is not explained by what the normaliser does to
  the image as a generic encoder sees it. The remaining candidate is the model-difference term: models trained on
  normalised tiles respond differently to the artefact. Caveat: distance in a generic ImageNet encoder is not what the
  trained classifier attends to; P5's model arms (test tiles, awaiting decision) measure the ΔG side directly.
- The negative is informative in itself: a plausible explanation (a normaliser fed a damaged tile estimates damaged
  statistics) is ruled out cheaply. JPEG's pixel-space amplification is largest at the mildest dose in every fold
  (E3 1.95 at ΔE 2, fold range 1.44–2.08; 1.36 at ΔE 10; 1.51 at ΔE 15) — not monotone; noted, not interpreted.
- Open and **untested** (career-ops): models trained on normalised tiles have less colour signal and may lean on
  structure, which would make them more exposed to structural artefacts. Not adopted; a test would compare reliance on
  texture between E1 and E2/E3 (the saliency work of Paper 3), not part of this rehearsal.
