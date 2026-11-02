# Analysis plan — CAMELYON17 rehearsal

> **Status: in force from the commit that added this file.** Committing it is what opens the test
> sets in code (design rule DR-012); the commit's timestamp shows the plan came before any test result.
> Any later change is a numbered deviation (§12). Decision and issue numbers (DR-, ISS-) refer to
> `docs/research_log.md` and the author's decision log.

## 1. Purpose and scope

The study asks whether a **label-free monitor** can tell when a tumour classifier's accuracy will
drop at a new site — and whether it tracks **harm** (lost accuracy) rather than mere **change** in
the images.

CAMELYON17 is a **rehearsal** for the full study on TCGA (DR-003). It exercises every step of the
pipeline end to end on a small dataset, so problems surface early. Because each CAMELYON17 hospital
contributes only 4–7 patients with tumour tiles (ISS-017), results from its five real hospitals are
reported as **descriptive**; the inferential weight rests on the controlled synthetic-shift arm
(§7), where the same patients are compared under different shifts. No value, threshold or tuned
setting from this rehearsal is carried into the TCGA study.

## 2. Sample sizes (stated plainly)

| | |
|---|---|
| Tiles | 11,763 (5,890 tumour, 5,873 normal), 128 µm at 0.5 µm/px |
| Hospitals / scanner types | 5 / 3 (hospitals 0, 1, 3 share one scanner type) |
| Patients with usable tiles | 98; patients with tumour tiles: 4–7 per hospital |
| Per fold | `test_in`: 12–15 tumour patients; `test_cross` (held-out hospital): 4–7 tumour patients |
| Tile dependence | tiles do not overlap (max 0.10% of a tile); 3,698 tumour–tumour tile pairs touch edge to edge |
| Training runs | 4 conditions × 5 folds × 3 seeds = 60 |
| Synthetic conditions | 5 families; 36 doses (25 on the common ΔE grid + 11 extended) |

## 3. Data (frozen)

- CAMELYON17 lymph-node H&E, 5 hospitals, 98 patients with usable tiles; 11,763 tiles of
  128 µm at 0.5 µm/px; frozen record SHA256 `1a98ed80…d192381` (`data/camelyon17/freeze.json`),
  verified by every run before any tile is read.
- Patient splits within each hospital (70/15/15); no patient in more than one split.

## 4. Design

- **Folds:** leave one hospital out (5 folds). Roles per fold: `train` (training patients of the
  four other hospitals), `test_in` (their validation + test patients — validation is merged because
  nothing is selected on it, DR-009), `test_cross` (every patient of the held-out hospital).
- **Conditions** (identical model and training; only stain handling differs):
  E1 baseline · E2 Macenko · E3 Reinhard · E5 DANN over the training hospitals only (DR-008).
  Stain references for E2/E3: 200 training tiles of the fold, fixed seed (DR-013).
  E4 (CycleGAN) and E6 (SimCLR) are not part of this rehearsal.
- **Model and schedule:** ResNet-50 from ImageNet weights, dropout + linear head; WILDS Camelyon17
  schedule — SGD, learning rate 1e-3, momentum 0.9, weight decay 1e-2, batch 32, 10 epochs, no early
  stopping (DR-011). Flips and 90° rotations only; no colour augmentation (DR-010). DANN settings as
  published (DR-017).
- **Seeds:** 42, 99, 123. Seeds are a nuisance: results are averaged over seeds within a fold, never
  tested across seeds (ISS-036).
- **Runs:** 4 conditions × 5 folds × 3 seeds = 60 runs (~4 GPU-hours); bit-for-bit reproducible.

## 5. Outcomes

- **Primary:** ΔG(AUROC) = AUROC(test_in) − AUROC(test_cross) for the real arm, and
  ΔG(d) = AUROC(test_in) − AUROC(test_in under shift d) for the synthetic arm.
- **Secondary:** the same with the calibration slope (DR-006). Absolute calibration (ECE) is not
  used because tiles are sampled 1:1 tumour:normal.
- Tile-level scores; uncertainty from a **patient-clustered bootstrap**, 2,000 resamples, stratified
  by patient class (ISS-031). Conditions that share patients are bootstrapped **jointly**. For each
  resample, ΔG is computed for each seed with the same patient draws and then averaged over seeds.

## 6. Monitor signals

All signals compare a **deployment batch** with the **reference** = the fold's training tiles.
Deployment batch = `test_cross` (real arm) or `test_in` under a shift (synthetic arm). Each signal
is used as a continuous value (not only as an alarm flag). Slide ids are passed as clusters, so
correlated tiles are permuted or resampled together (ISS-039).

| Tier | Input | Statistic |
|---|---|---|
| 1a | 23 pixel statistics per tile, no model (DR-016) | MMD² and mean Wasserstein distance |
| 1b | frozen ImageNet DenseNet-121 features (1,024-d), no fine-tuning (DR-005, DR-018); ResNet-50 as sensitivity | MMD² and mean Wasserstein distance |
| 2 | the deployed classifier's 2,048 penultimate features; its logits | mean Mahalanobis distance; mean energy |

Tier 1 is computed on the tiles as they arrive (before stain normalisation) and is therefore the same
for every condition.

## 7. Synthetic-shift arm (the controlled experiment)

- **Families** (DR-007): blur (out of focus), striping, clipped tissue, JPEG compression, stain
  intensity/colour. No family is assumed harmless (see the pilot below).
- **Doses** (DR-015, DR-021): matched across families on the mean per-pixel CIELAB colour difference ΔE
  relative to the original tile (ISS-042 explains why SSIM could not be used). Common grid ΔE 2, 4, 6, 8, 10
  for every family; extended doses where the first pilot showed no effect: striping ΔE 15/20/25/30,
  clipped 15/20/30/40, JPEG 12/15, stain 12 — 36 doses. Fixed on training-split tiles only
  (`configs/shift_doses_camelyon17.json`); SSIM recorded for every dose.
- Shifts are applied to `test_in` tiles at inference; models never see shifted images in training.
- **Dose-range pilot (DR-020, as amended; done before this commit, training split only).** E1 models
  trained by 5-fold cross-fitting inside the training split, so every training-split patient (58; 12 with
  tumour) was scored by a model that had not seen them. Results (`docs/pilot/dose_response_rotated_5fold.csv`):
  in-domain AUROC 0.954 [0.912, 0.988]; AUROC drop at ΔE 10 — blur 0.285, JPEG 0.136, stain 0.041,
  striping 0.036, clipped 0.035; at the extended doses striping reaches 0.238 (ΔE 30) and clipped 0.401
  (ΔE 40); stain rises steadily but shallowly to 0.066 at ΔE 12. Within every family, dose orders harm
  almost perfectly (Spearman 0.96–1.0); across families, ΔE ranks harm with ρ = 0.78 and 1 − SSIM with
  ρ = 0.85 (0.63 and 0.68 on the common grid). A first, single-split pilot held out only 2 tumour patients
  and was superseded (`docs/pilot/dose_response_single_split_2_tumour_patients.csv`). The pilot bounded the
  dose range; it is not an estimate of the effects this plan tests.

## 8. Hypotheses

**H1 — Detect.** Within each fold, across the 36 synthetic conditions, each monitor signal is positively
associated with ΔG(AUROC): Spearman ρ > 0 (one-sided), for Tier 2 and for Tier 1b (DenseNet-121).

**H2 — Magnitude is not harm, and the monitor tracks harm (DR-019).** Across the 36 conditions, ΔG is
ranked by (a) each monitor signal, (b) ΔE, (c) 1 − SSIM. The quantity of interest is the difference in ρ
between each monitor signal and each baseline, with joint patient-bootstrap intervals. The bar is high:
on training data the baselines already reach ρ = 0.78 (ΔE) and 0.85 (1 − SSIM). And ΔE and SSIM are
**paired** — they need the original tile, which a deployed monitor never has — so the comparison favours
the baselines. A monitor that matches them without the original tile is already informative; one that
beats them is the strong result. Whichever way it falls is reported.

**H3 — Pre-registered prediction, written in the light of the pilot (falsifiable).** Tier 1a's signal
will rise monotonically with stain dose across the whole stain range, while ΔG for stain stays small
(the pilot: ≤ 0.012 up to ΔE 6, 0.066 at ΔE 12 — about a third of blur's 0.213 at ΔE 8). So, across
families, Tier 1a will over-weight stain: at equal Tier 1a signal, stain conditions will show less ΔG
than blur conditions, and Tier 1a's pooled ρ with ΔG will be lower than Tier 1b's and Tier 2's. Tier 1b
and Tier 2 will rank the families by harm (blur and JPEG above stain at equal ΔE). This prediction was
written after the training-split pilot and before any test result; it can fail if Tier 1a does not rise
with stain, if stain harm is not small relative to blur, or if Tier 1a ranks harm as well as Tier 1b.

## 9. Analysis

1. **Per fold, all shown.** Every per-fold estimate is reported with its interval; no mean ± SD over
   five points; no mixed models or cluster-robust errors with five clusters.
2. **Within-family dose–response (necessary-condition check):** Spearman ρ between signal and ΔG over
   each family's doses plus dose 0, with intervals from the joint bootstrap. A monitor that does not rise
   with dose inside a family fails here. Because dose orders harm almost perfectly within every family
   (pilot ρ 0.96–1.0), any monotone signal passes, so this check does not discriminate between monitors
   and baselines; the pooled ranking across families (item 3) does.
3. **Pooled predictor comparison (H2, primary for the reframe):** ρ of each predictor with ΔG over all 36
   conditions, and on the 25 common-grid conditions, with the differences in ρ (monitor − ΔE,
   monitor − (1 − SSIM)); intervals from the same joint resamples.
4. **Real arm (descriptive):** for each held-out hospital and condition, ΔG with its interval against the
   monitor signals — five points per condition; no inferential claim (DR-003).
5. **Secondary:** ΔE-matched comparison between families at equal dose; SSIM-matched comparison where
   reachable (SSIM ≥ 0.95, mild doses only); calibration-slope ΔG (the pilot shows it is far noisier than
   AUROC at this sample size); Tier 1a colour-only subset; Tier 1b with ResNet-50 as the sensitivity
   analysis for the shared-starting-weights question (DR-018).
6. **Alarm behaviour:** at α = 0.05 per tier and the AND rule, the proportion of synthetic conditions
   flagged, by family and dose.
7. **Realised false-alarm rate, as a standing check.** For every fold and tier, the monitor is run on
   null deployments — the fold's own training tiles split into two halves by slide, 200 random splits —
   and the realised alarm rate is reported against the nominal α. A tier whose realised rate exceeds
   α by more than its simulation error is reported as miscalibrated, and its alarms are not
   interpreted. (ISS-039 showed that tile-level calibration fired on 100% of null deployments at a
   nominal 10%; this check is how such a failure would be seen on real data.)

## 10. What would count against the claim

- H1: ρ ≤ 0, or intervals spanning 0 in most folds, for Tier 2 and Tier 1b.
- H2: the monitor's ρ below ΔE's or SSIM's, with the difference interval excluding 0.
- H3: any of its three failure routes (stated in §8).
Each outcome is reported whichever way it falls.

## 11. Ceiling contingency (pre-specified)

If in-domain AUROC (`test_in`, E1, averaged over seeds) is ≥ 0.98 in the **first fold scored**
(fold 0), the five-hospital arm is treated as descriptive only for every hypothesis, and the
synthetic arm is the primary evidence for all of them. A ceiling compresses the real-arm ΔG and
makes it uninformative about harm. The rotated pilot (§7) gave an in-domain AUROC of 0.954 [0.912, 0.988]
(12 tumour patients), so the contingency is not expected to apply; it is declared here so that its
use, if fold 0 crosses the threshold, cannot read as a post-hoc choice. (A first pilot on 2 tumour patients
gave 0.998, which is why the threshold is checked on fold 0, not on a pilot.)

## 12. Deviations

Any change after this plan is committed is logged as a numbered deviation with its reason and date,
and reported in the paper. If training visibly fails under the fixed schedule, validation is taken from
the training hospitals only (never `test_in` or the held-out hospital) and the deviation is reported
(DR-009).

## 13. Decisions this plan rests on

All decided before this commit (decision log numbers): ΔE dose matching (DR-015); Tier 1b DenseNet-121
primary with ResNet-50 sensitivity (DR-018); the H2 predictor comparison (DR-019); the dose-range pilot and
its rotation (DR-020); dose extensions (DR-021); training schedule (DR-011); stain references (DR-013);
DANN design (DR-008, DR-017); augmentation (DR-010); validation merged into test_in (DR-009).

## Attestation

At the time this plan was written and committed, **no test-set result had been computed or seen.**
Precisely:

- No model was ever scored on a hospital it had not been trained on: no cross-hospital (`test_cross`)
  performance, and no ΔG, exists.
- No tile of any `test_in` set (validation and test patients) was read by any model or monitor.
- Everything run before this commit used **training-split patients** only: the real-tile checks of each
  step, the dose calibration, the dose-range pilot (models trained and scored within the training split,
  all five hospitals) and a monitor null check (training-split slides split into random halves).
  Under leave-one-hospital-out, a hospital's training-split tiles become part of `test_cross` in the fold
  where that hospital is held out; in the pilot and checks they were only ever scored by models trained on
  that same hospital, and no per-hospital divergence between a fold's reference and its held-out hospital
  was computed.
- The code refuses to score `test_in` or `test_cross` until this file is committed as
  `docs/analysis_plan.md` (DR-012); every run records the commit it ran under.

