# Research log

Problems met in the data pipeline and study design, their causes, and how each was resolved.
Entries keep the IDs of the full working log; tooling and repository-housekeeping entries are
not included here, so the numbering has gaps.

Status: **RESOLVED** · **OPEN** (work pending) · **DECISION** (design choice still to be made) ·
**ACCEPTED** (known limitation, documented).

| ID | Date | Area | Summary | Status |
|---|---|---|---|---|
| ISS-001 | 2026-02 | splits | Patients with many slides would leak across splits | RESOLVED |
| ISS-002 | 2026-02 | splits | Splits changed when metadata was refreshed or reordered | RESOLVED |
| ISS-003 | 2026-02 | splits | Stratification impossible for very small classes | RESOLVED |
| ISS-004 | 2026-08 | splits | Splits were not institution-stratified | RESOLVED (CAMELYON17) / OPEN (TCGA) |
| ISS-005 | 2026-09 | splits | TCGA-COAD: institution × label strata too small | ACCEPTED |
| ISS-012 | 2026-10 | data | TCGA tumour vs normal is confounded with frozen vs FFPE preparation | DECISION |
| ISS-014 | 2026-10 | tiling | Pixel size differs by scanner | RESOLVED |
| ISS-015 | 2026-10 | tiling | Black unscanned glass would count as tissue | RESOLVED |
| ISS-016 | 2026-10 | tiling | Pen-ink marks counted as tissue | RESOLVED |
| ISS-017 | 2026-10 | data | Tumour tiles concentrated in few patients | ACCEPTED |
| ISS-018 | 2026-10 | tiling | Center 2 at the tissue-threshold ceiling | RESOLVED |
| ISS-019 | 2026-10 | QC | QC scores correlate with label and center | RESOLVED |
| ISS-020 | 2026-10 | splits | A one-tile tumour patient landed in a test split | RESOLVED |
| ISS-022 | 2026-10 | splits | Bugs in the original splitting code | RESOLVED |
| ISS-023 | 2026-10 | normalisation | Reinhard (E3) produced near-black images | RESOLVED |
| ISS-024 | 2026-10 | environment | pandas 2.x silently turned labels into text | RESOLVED |
| ISS-026 | 2026-10 | reproducibility | GPU determinism needs `CUBLAS_WORKSPACE_CONFIG` | RESOLVED |
| ISS-028 | 2026-10 | data | CAMELYON17 labels were guessed from file paths | RESOLVED |
| ISS-029 | 2026-10 | splits | TCGA bootstrap script rebuilds splits on every run | OPEN |
| ISS-030 | 2026-10 | methods | DANN as written is domain adaptation, not generalisation | RESOLVED |
| ISS-031 | 2026-10 | statistics | Bootstrap resamples tiles, not patients | RESOLVED |
| ISS-036 | 2026-10 | statistics | Three seeds cannot support the planned paired tests | RESOLVED |
| ISS-037 | 2026-10 | methods | E5 (DANN) classifier head differs from E1 | RESOLVED |
| ISS-039 | 2026-10 | statistics | Neighbouring tiles are correlated; tests must resample patients or slides | RESOLVED |
| ISS-040 | 2026-10 | reproducibility | Worker seeding replayed the same augmentations every epoch | RESOLVED |
| ISS-041 | 2026-10 | normalisation | Macenko (E2) turned real tiles nearly black | RESOLVED |
| ISS-042 | 2026-10 | methods | SSIM cannot match stain shifts to the other shift families | RESOLVED |
| ISS-043 | 2026-10 | methods | The generic-encoder monitor tier starts from the same weights as the classifier | RESOLVED |

---

### ISS-001 · splits · Patients with many slides would leak across splits
- **Problem:** TCGA has up to 14 slides per patient, CAMELYON17 five lymph nodes per patient.
  Splitting by slide would put one patient's slides in both training and test.
- **Cause:** repeated measures — slides from one patient are not independent.
- **Resolution:** `splitting.py` splits by patient and keeps all of a patient's slides together,
  detects repeated measures, and checks for leakage at patient and slide level.

### ISS-002 · splits · Splits changed when metadata was refreshed or reordered
- **Problem:** re-querying the GDC portal or reading a CSV in a different row order produced
  different splits.
- **Resolution:** deterministic sorting; a frozen, hashed list of all patients
  (`patient_universe.txt`); a seed derived from project, universe hash and split version; an
  overwrite guard; `split_metadata.json` records hashes and a checksum. The extended code
  reproduces the existing TCGA BRCA/COAD/LUAD split checksums exactly.

### ISS-003 · splits · Stratification impossible for very small classes
- **Problem:** stratified splitting fails when a class has fewer than two patients.
- **Resolution:** stratification is disabled for that case and the reason is recorded.

### ISS-004 · splits · Splits were not institution-stratified
- **Problem:** earlier splits were stratified by label only. Without a held-out institution the
  generalisation gap ΔG cannot be measured.
- **Resolution:** splits are made within each institution (`group_column`), with
  leave-one-center-out folds (`loco_fold`); TCGA tissue-source-site codes are extracted as the
  institution. Done for CAMELYON17 (5 folds); the TCGA partition waits on ISS-012.

### ISS-005 · splits · TCGA-COAD institution × label strata too small
- **Problem:** some (site × normal) cells hold one or two patients and cannot be split three ways.
- **Resolution:** such groups fall back to label-only stratification, recorded per group in
  `split_metadata.json`; every split is checked to contain both classes.

### ISS-012 · data · TCGA tumour vs normal is confounded with preparation
- **Problem:** in BRCA, COAD and LUAD every normal-tissue slide is a frozen section; none of the
  FFPE diagnostic slides is normal. Normals also come from only 7, 5 and 14 sites.
- **Consequence:** a model could learn "frozen-section artefact = normal", and most held-out
  sites would have no normals, so ΔG could not be computed there.
- **Current approach:** Paper 1 uses CAMELYON17 only; the slide table records preparation
  (`slide_prep`). Options for TCGA: a frozen-only task, or a different FFPE-only label checked
  for site balance.

### ISS-014 · tiling · Pixel size differs by scanner
- **Problem:** CAMELYON17 pixel sizes are 0.243 µm (centers 0, 1, 3), 0.227 µm (center 2) and
  0.250 µm (center 4); the objective-power field is missing.
- **Resolution:** every tile covers the same physical area (128 µm at 0.5 µm/px), resampled from
  the recorded pixel size. Tests cover three scanner resolutions.

### ISS-015 · tiling · Black unscanned glass
- **Problem:** centers 0, 1 and 3 store unscanned glass as pure black, which a "dark = tissue"
  detector would tile.
- **Resolution:** tissue must be both saturated and above a brightness cutoff.

### ISS-016 · tiling · Pen-ink marks counted as tissue
- **Problem:** dark ink stars (brightness 29–55) passed the saturation test — 100 tiles on one slide.
- **Resolution:** after inspecting the worst slide and 20 slides across all centers, the brightness
  cutoff was raised from 30 to 80 (real tissue's 5th percentile is ≥ 99 on the sampled slides).

### ISS-017 · data · Tumour tiles concentrated in few patients
- **Problem:** only 34 slides have tumour outlines (5–8 per center); in center 4 one slide held 89%
  of the tumour tiles; 112 tumour-containing slides have no outline and cannot be tiled reliably.
- **Resolution:** at most 500 tumour tiles per slide (no slide then exceeds 55% of its center);
  uncertainty to be reported with patient-clustered intervals (ISS-031). Stated as a limitation.

### ISS-018 · tiling · Center 2 at the tissue-threshold ceiling
- **Problem:** pale, fatty center-2 slides hit the upper clamp of the per-slide saturation threshold.
- **Checked:** previews show tumour and lymph-node tissue detected; only fat septa are excluded.
  No change needed.

### ISS-019 · QC · QC scores correlate with label and center
- **Problem:** lymphocyte-rich normal tissue scores "sharper" than tumour, and full-resolution
  tissue fraction under-counts dark, dense tissue (centers 1, 3) — aggressive QC would bias the
  tile set by class and center.
- **Resolution:** the lowest-scoring tiles were reviewed visually and minimal thresholds chosen
  (blur ≥ 15, entropy ≥ 4.5): 17 of 11,780 tiles removed, none of them tumour. Full-resolution
  tissue fraction is not used as a filter.

### ISS-020 · splits · A one-tile tumour patient landed in a test split
- **Problem:** matching tumour-tile fractions exactly put a patient with a single tumour tile into
  center 3's test split, and left some validation splits with 10–40 tumour tiles.
- **Resolution:** the test split of every center must hold at least 100 tumour tiles when
  achievable; the result is 110–357 per center.

### ISS-022 · splits · Bugs in the original splitting code
- **Problem:** 12 failing tests: a column-name clash in one-slide-per-patient mode, a duplicated
  column during aggregation, a repeated-measures warning that could never fire, and a missing
  patch-level verification function.
- **Resolution:** all fixed and covered by tests.

### ISS-023 · normalisation · Reinhard (E3) produced near-black images
- **Problem:** three mixed colour-space conventions: statistics on 0–255 Lab with a 0–100 tissue
  threshold, clipping to the wrong ranges, and 0–1 floats cast to 8-bit integers.
- **Resolution:** rewritten on floating-point CIE Lab throughout; tests check round trip,
  self-normalisation and movement toward the reference on tissue pixels.

### ISS-024 · environment · pandas 2.x silently turned labels into text
- **Problem:** assigning a string and an integer to two columns at once made `label` a text column
  under pandas 2.1 but not pandas 3.
- **Resolution:** columns assigned separately and cast; the full test suite runs in the pinned
  training environment.

### ISS-026 · reproducibility · GPU determinism needs `CUBLAS_WORKSPACE_CONFIG`
- **Problem:** `set_seed(deterministic=True)` enables deterministic algorithms, but only
  `make_deterministic` sets the cuBLAS workspace variable; GPU matrix multiplies fail without it.
- **Resolution:** the training run sets both; ResNet's adaptive average pool, which has no deterministic GPU
  backward pass, is replaced by an identical spatial mean. Two runs with the same seed on the real tiles
  give bit-identical outputs. Test sets are scored only after the analysis plan is committed to git.

### ISS-028 · data · CAMELYON17 labels were guessed from file paths
- **Problem:** the original metadata step inferred labels and centers from file paths, giving
  empty values.
- **Resolution:** labels and centers now come from the official `stages.csv` and the annotation files.

### ISS-029 · splits · TCGA bootstrap script rebuilds splits on every run
- **Problem:** `tcga_bootstrap.sh` recreates splits with `--force`, which would silently replace
  splits after a TCGA tile set is frozen.
- **Plan:** skip the split step when a freeze record exists.

### ISS-030 · methods · DANN as written is domain adaptation
- **Problem:** `DANNModule` trains with unlabelled images from the held-out center. That is domain
  adaptation (the new site's images are seen in training), not domain generalisation, and it
  changes what E5 can claim.
- **Resolution (2026-10-30):** generalisation. E5's domain classifier distinguishes the four training
  centres; the held-out centre is never seen. Adaptation would train on the deployment site's images
  and score the monitor on images the features were aligned to; it is deferred to Paper 2.

### ISS-031 · statistics · Bootstrap resamples tiles, not patients
- **Problem:** `bootstrap_ci` resamples individual values; tiles are clustered within patients, so
  intervals for AUC and ΔG would be too narrow.
- **Resolution:** a patient-clustered bootstrap (`evaluation/bootstrap.py`): patients are resampled with
  replacement, stratified by whether they have tumour tiles, for AUROC, calibration slope and ΔG; conditions
  that share patients (synthetic shifts) are resampled jointly so their differences get honest intervals.

### ISS-036 · statistics · Three seeds cannot support the planned paired tests
- **Problem:** the plan compares conditions with paired t-tests and Wilcoxon tests over three seeds
  (42, 99, 123), Bonferroni-corrected. With three pairs the smallest two-sided Wilcoxon p-value is
  2/2³ = 0.25, so it can never reach 0.05; a t-test with 2 degrees of freedom needs a very large,
  very consistent effect, and Bonferroni over 8 comparisons tightens this further. Seeds also
  capture only training variability, not the sampling of test patients.
- **Options:** patient-clustered bootstrap intervals (ISS-031) as the main inference, with seeds
  reported as training variability; or more seeds. Using the five leave-one-center-out folds as
  the unit of replication does not solve it for a Wilcoxon test: with 5 pairs the smallest
  two-sided p-value is 2/2⁵ = 0.0625.
- **Evidence:** `tests/test_statistical_tests.py::test_wilcoxon_cannot_reach_005_with_three_pairs`.
- **Resolution (2026-10-30):** estimation, not significance testing. The centre is the unit of
  replication; seeds are averaged within each fold; every per-fold value is shown with a
  patient-clustered interval; the controlled synthetic-shift arm carries the inferential weight.

### ISS-037 · methods · E5 classifier head differs from E1
- **Problem:** E1–E3 use ResNet-50 with dropout and one linear layer; `DANNModule` (E5) uses a
  2048 → 256 → 2 head. The design requires identical downstream models with only stain handling
  varying, so an E5-vs-E1 difference would mix the effect of DANN with the effect of the head.
- **Resolution (2026-10-30):** E5 uses E1's head (dropout + one linear layer), so any difference
  comes from DANN alone and Tier 2 reads the same layer in every condition.

### ISS-039 · statistics · Neighbouring tiles are correlated
- **Checked:** frozen tiles do not overlap — the grid stride equals the tile size, and the largest
  overlap between any two tiles is 0.10% of a tile (rounding). The 500-tile tumour cap is a
  ceiling, not a quota: 26 of 32 tumour slides contributed every tile they had.
- **Problem:** tumour tiles are mostly edge-to-edge neighbours from a few lesions, and neighbouring
  tiles are correlated. Permutation tests and bootstraps that treat tiles as independent would give
  too many false alarms and intervals that are too narrow.
- **Resolution:** evaluation intervals resample patients (ISS-031); the monitor's alarm tests (package
  wsidrift) now permute or bootstrap whole slides. In simulation with strong slide effects and no real shift,
  at a 10% false-alarm setting: tile-level tests alarmed 100% (Tier 1) and 52% (Tier 2) of the time;
  slide-level tests 6% and 11%.

### ISS-040 · reproducibility · Worker seeding replayed augmentations every epoch
- **Problem:** the data-loader worker seeding reset each worker's random generator to the same value at
  every epoch, so random flips and rotations repeated epoch after epoch.
- **Resolution:** workers are now seeded from the loader's own seeded generator — different per worker
  and per epoch, identical for the same base seed. Augmentation itself is geometric only (flips and 90°
  rotations); colour augmentation is excluded because it is a stain-handling method in its own right.

### ISS-041 · normalisation · Macenko (E2) turned real tiles nearly black
- **Found:** checking E2 on real training tiles before any E2 run: every normalised tile came out nearly
  black. Synthetic test tiles (no background) had not exposed it.
- **Cause:** haematoxylin and eosin rows were ordered backwards, SVD signs were not oriented toward
  positive optical density, and the stain-amount rescaling had no guard; a tile and the reference ordered
  differently were rescaled against the wrong dye.
- **Resolution:** signs oriented, rows ordered by red absorbance (haematoxylin first), rescaling guarded.
  On real tiles: no dark outputs, H&E appearance preserved. Regression tests added. No E2 results existed.

### ISS-042 · methods · SSIM cannot match stain shifts to the other shift families
- **Found:** before fixing the synthetic-shift doses, the reachable range of each family was measured on
  training-split tiles. Stain changes barely move SSIM (≥ 0.94 even at maximum strength), while blur reaches
  0.22. Matching on SSIM would leave only mild doses and remove the harmful arm.
- **Resolution:** doses are matched on the mean per-pixel CIELAB colour difference ΔE (2, 4, 6, 8, 10), which
  every family reaches; SSIM is reported for every dose. The dose table was fixed on training-split tiles
  only and is committed (`configs/shift_doses_camelyon17.json`). Limitation stated: at equal ΔE, blur
  changes structure far more than stain does, so any equal-size yardstick embeds a judgement.

### ISS-043 · methods · The generic-encoder monitor tier starts from the same weights as the classifier
- **Problem:** Tier 1b uses a frozen ImageNet ResNet-50; the classifier being monitored is a ResNet-50 fine-tuned
  from the same ImageNet weights. The two are identical at initialisation, so Tier 1b is not fully independent of
  what it monitors.
- **Options:** keep ResNet-50 and state the coupling; switch Tier 1b to a different open architecture
  (DenseNet-121 is implemented and checked on real tiles); or report both, DenseNet-121 primary and ResNet-50 as a
  sensitivity analysis.
- **Resolution (2026-11-02):** DenseNet-121 is the primary generic encoder; ResNet-50 is reported as a
  sensitivity analysis, which also measures how much the shared starting point matters.

