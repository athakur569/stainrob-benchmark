# Analysis plan — CAMELYON17 rehearsal (DRAFT, not yet in force)

> **Status: draft for review.** Nothing below has been run on test data. Test sets are sealed in
> code until this plan is committed as `docs/analysis_plan.md` (design rule DR-012); the commit's
> timestamp then shows the plan came before any test result. Items marked **OPEN** must be decided
> before that commit. Decision and issue numbers (DR-, ISS-) refer to `docs/research_log.md` and
> the author's decision log.

## 1. Purpose and scope

The study asks whether a **label-free monitor** can tell when a tumour classifier's accuracy will
drop at a new site — and whether it tracks **harm** (lost accuracy) rather than mere **change** in
the images.

CAMELYON17 is a **rehearsal** for the full study on TCGA (DR-003). It exercises every step of the
pipeline end to end on a small dataset, so problems surface early. Because each CAMELYON17 hospital
contributes only 4–7 patients with tumour tiles (ISS-017), results from its five real hospitals are
reported as **descriptive**; the inferential weight rests on the controlled synthetic-shift arm
(§5), where the same patients are compared under different shifts. No value, threshold or tuned
setting from this rehearsal is carried into the TCGA study.

## 2. Data (frozen)

- CAMELYON17 lymph-node H&E, 5 hospitals, 98 patients with usable tiles; 11,763 tiles of
  128 µm at 0.5 µm/px; frozen record SHA256 `1a98ed80…d192381` (`data/camelyon17/freeze.json`),
  verified by every run before any tile is read.
- Patient splits within each hospital (70/15/15); no patient in more than one split.

## 3. Design

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

## 4. Outcomes

- **Primary:** ΔG(AUROC) = AUROC(test_in) − AUROC(test_cross) for the real arm, and
  ΔG(d) = AUROC(test_in) − AUROC(test_in under shift d) for the synthetic arm.
- **Secondary:** the same with the calibration slope (DR-006). Absolute calibration (ECE) is not
  used because tiles are sampled 1:1 tumour:normal.
- Tile-level scores; uncertainty from a **patient-clustered bootstrap**, 2,000 resamples, stratified
  by patient class (ISS-031). Conditions that share patients are bootstrapped **jointly**. For each
  resample, ΔG is computed for each seed with the same patient draws and then averaged over seeds.

## 5. Monitor signals

All signals compare a **deployment batch** with the **reference** = the fold's training tiles.
Deployment batch = `test_cross` (real arm) or `test_in` under a shift (synthetic arm). Each signal
is used as a continuous value (not only as an alarm flag). Slide ids are passed as clusters, so
correlated tiles are permuted or resampled together (ISS-039).

| Tier | Input | Statistic |
|---|---|---|
| 1a | 23 pixel statistics per tile, no model (DR-016) | MMD² and mean Wasserstein distance |
| 1b | frozen ImageNet encoder features, no fine-tuning (DR-005) — **OPEN: encoder, ISS-043** | MMD² and mean Wasserstein distance |
| 2 | the deployed classifier's 2,048 penultimate features; its logits | mean Mahalanobis distance; mean energy |

Tier 1 is computed on the tiles as they arrive (before stain normalisation) and is therefore the same
for every condition.

## 6. Synthetic-shift arm (the controlled experiment)

- **Families** (DR-007): blur (out of focus; primary harmful), striping, clipped tissue, JPEG
  compression (secondary), stain intensity/colour (expected harmless control).
- **Doses:** five per family at mean per-pixel CIELAB ΔE 2, 4, 6, 8, 10 relative to the original
  tile, fixed on training-split tiles only (`configs/shift_doses_camelyon17.json`; DR-015 —
  **OPEN: confirmation**, ISS-042 explains why SSIM could not be used). SSIM is recorded for every dose.
- Shifts are applied to `test_in` tiles at inference; the model never sees shifted images in training.
- **Dose-range pilot — OPEN (proposed):** before this plan is committed, a pilot model is trained on
  ~80% of training-split patients and the AUROC drop under each shift is measured on the other ~20%
  (no test tile touched). The pilot only **bounds** the dose range ("does this dose do anything?"); it
  does not estimate effects. Rule: extend a family's range unless the pilot shows a clear, large
  response; prefer wider ranges.

## 7. Hypotheses

**Primary (Detect claim).** Within each fold, across the synthetic conditions, the monitor signal is
positively associated with ΔG: Spearman ρ > 0 (one-sided), for Tier 2 and for Tier 1b.

**Pre-registered prediction (falsifiable).** Tier 1a will track shift **magnitude** and fail on the
stain family (signal rises with dose while ΔG stays flat); Tier 1b and Tier 2 will track **harm**.

**Reframed claim under review — OPEN.** "Magnitude is not harm, and the monitor tracks harm": across all
synthetic conditions, ΔG is predicted by (a) each monitor signal, (b) ΔE, (c) SSIM; the monitor should
predict ΔG materially better than the image-difference measures. Caveat to state: ΔE and SSIM are
**paired** (they need the original tile), which a deployed monitor never has, so the comparison favours
the baselines.

## 8. Analysis

1. **Per fold, all shown.** Every per-fold estimate is reported with its interval; no mean ± SD over
   five points; no mixed models or cluster-robust errors with five clusters.
2. **Within-family dose–response:** Spearman ρ between signal and ΔG over the six points of each family
   (dose 0 plus five), with intervals from the joint bootstrap. Guards against a between-family
   confound (a monitor that only separates "structure" from "colour" shifts would look like it tracks harm).
3. **Pooled predictor comparison** (if §7's reframe is adopted): ρ of each predictor with ΔG over all
   25 conditions and the differences in ρ (monitor − ΔE, monitor − SSIM), intervals from the same
   joint resamples.
4. **Real arm (descriptive):** for each held-out hospital and condition, ΔG with its interval against the
   monitor signals — five points per condition; no inferential claim (DR-003).
5. **Secondary:** ΔE-matched comparison between families at equal dose; SSIM-matched comparison where
   reachable (SSIM ≥ 0.95, mild doses only); calibration-slope ΔG; Tier 1a colour-only subset.
6. **Alarm behaviour:** at α = 0.05 per tier and the AND rule, the proportion of synthetic conditions
   flagged, by family and dose.

## 9. What would count against the claim

- ρ ≤ 0, or intervals spanning 0 in most folds, for Tier 2 and Tier 1b on the harmful families.
- Tier 1b tracking magnitude as Tier 1a does (rising on stain with flat ΔG).
- ΔE or SSIM predicting ΔG as well as or better than the monitor.
Each outcome is reported whichever way it falls.

## 10. Deviations

Any change after this plan is committed is logged as a numbered deviation with its reason and date,
and reported in the paper. If training visibly fails under the fixed schedule, validation is taken from
the training hospitals only (never `test_in` or the held-out hospital) and the deviation is reported
(DR-009).

## 11. Open decisions before commit

| Item | Question |
|---|---|
| DR-015 | Confirm ΔE as the dose yardstick |
| ISS-043 | Tier 1b encoder: ResNet-50 (shares the classifier's starting weights), DenseNet-121, or both |
| §7 | Adopt the "magnitude is not harm" predictor comparison as a pre-specified analysis |
| §6 | Approve the dose-range pilot design |
