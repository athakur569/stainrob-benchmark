# stainrob-benchmark

Research code for a PhD dissertation (Data Science, Harrisburg University of Science and
Technology) on **label-free validity monitoring for computational pathology**: detecting
when stain or scanner variation at a new site has made a deployed model's outputs
unreliable, without needing new labels at that site.

> **Status: work in progress (November 2026).** The full pipeline for the CAMELYON17
> rehearsal is built and tested end to end — data, training of four conditions, synthetic
> shifts, monitor features and statistics — and checked on real tiles. **No test-set result
> exists yet:** test sets are sealed in code until the analysis plan
> ([draft](docs/analysis_plan_DRAFT.md)) is committed. See [What exists today](#what-exists-today).

## The question

A tumour-detection model is validated at one set of hospitals. A new hospital's slides
look different (stain, scanner). Is the model still valid there? Accreditation (CAP)
requires revalidation after significant changes but does not specify a method, and the
labels needed to measure performance directly are usually unavailable at deployment.

The approach, developed in three papers:

1. **Detect (Paper 1, in progress).** During development, where labels exist, measure the
   generalisation gap **ΔG = AUC on unseen patients from seen centers − AUC on an unseen
   center**. Relate ΔG to label-free signals: input divergence (MMD, Wasserstein; *Tier 1*)
   and model response (Mahalanobis distance, energy score; *Tier 2*). At deployment only the
   label-free signals are computed, giving a *predicted* ΔG.
2. **Prescribe (Paper 2, planned).** Map the type of shift to a corrective action.
3. **Verify (Paper 3, planned).** Check that correct predictions rest on the right tissue.

The central claim to test: **image change is not the same as harm.** A monitor should flag the
shifts that cost accuracy (for example, out-of-focus scanning) and not the ones that don't (for
example, a modest stain-colour change) — something a simple measure of image difference cannot do.

The method assumes **covariate shift**: image appearance changes, while the relationship
between tissue and diagnosis does not. That fits stain and scanner variation. It does not
cover changes in label definitions or patient population.

## Data

Public, de-identified data only (IRB: Exempt).

- **CAMELYON17 — the rehearsal.** Lymph-node H&E slides, 5 hospitals (three scanner types),
  100 patients, 500 slides, tumour outlines on 50 slides. Small enough to exercise and debug
  every step quickly. Only 4–7 patients per hospital have outlined tumour, so its five
  real hospitals give descriptive results; the inferential weight rests on a controlled
  synthetic-shift experiment, where the same patients are compared under different shifts.
- **TCGA (BRCA, COAD, LUAD) — the full study, later.** Starts afresh; nothing tuned on
  CAMELYON17 is carried over. In these cohorts every normal-tissue slide is a frozen section,
  so the task must first be redesigned (`docs/research_log.md`, ISS-012).

Slides are not in this repository. `scripts/download_camelyon17.py` registers a local
copy and builds the slide table.

## What exists today

Every row is built and covered by the test suite (173 tests); "checked on real tiles" means the
step was also run on the frozen CAMELYON17 tiles and inspected before use.

| Step | Where | Checked on real tiles |
|---|---|---|
| Slide table, tiling at 0.5 µm/px, quality control, patient splits, SHA256 freeze | `data/` | yes — 11,763 tiles frozen; tiles verified against raw slides on all 5 scanners |
| Fold loader: leave-one-hospital-out roles, freeze check at start-up | `data/dataset.py` | yes — 12–15 tumour patients in-domain per fold, 4–7 at the held-out hospital |
| Training run: WILDS schedule, per-tile predictions and 2,048-d features saved | `training/run.py`, `scripts/train.py` | yes — bit-for-bit reproducible on GPU; loss levels off by epoch 8–10; ~4 min per run |
| Conditions E1 (baseline), E2 (Macenko), E3 (Reinhard), E5 (DANN over training hospitals) | `training/run.py` | yes — see figure below; DANN's hospital-recognition falls from 83% to 32% (chance 25%) |
| Synthetic shifts: blur, striping, clipped tissue, JPEG, stain colour; doses matched by colour difference | `data/shifts.py`, `configs/shift_doses_camelyon17.json` | yes — see figure below |
| Monitor Tier 1: 23 pixel statistics; frozen generic-encoder features | `monitor/` | yes — 7,913 training-split tiles |
| Monitor Tier 1/2 statistics and alarms (with slide-level calibration) | package [`wsidrift`](https://pypi.org/project/wsidrift/) | simulation: false alarms at the set rate |
| Uncertainty: patient-clustered bootstrap, joint across conditions; predictor comparison | `evaluation/bootstrap.py` | tests |
| Analysis plan | [`docs/analysis_plan_DRAFT.md`](docs/analysis_plan_DRAFT.md) | draft for review |
| CycleGAN (E4), SimCLR (E6) | — | not part of the rehearsal |

### Checks on real tiles

Stain normalisers on one training tile per hospital (left: original; middle: Macenko; right: Reinhard).
This check caught a bug that turned every Macenko output black (ISS-041) before any run used it.

![Stain normalisers on real tiles](docs/figures/stain_normalisers_real_tiles.png)

Each synthetic shift family at the five doses (colour difference ΔE 2–10), on one training tile:

![Synthetic shifts on a real tile](docs/figures/synthetic_shifts_real_tile.png)

### How results are protected

- The tiles, splits and configuration are frozen and hashed; every run verifies the hashes first.
- Test sets cannot be scored until `docs/analysis_plan.md` is committed to git
  (`training/run.py`, `require_committed_plan`). The commit's timestamp shows the plan came first.
  So far only training data has been scored.
- Every problem met and every design decision is logged with its reasoning in
  [`docs/research_log.md`](docs/research_log.md).

### The frozen CAMELYON17 tile set

`data/camelyon17/` holds the record of the frozen tile set: `freeze.json` (SHA256 of every
part), the slide table, the tile manifest, and every split file. `tests/test_frozen_record.py`
re-hashes them against `freeze.json`.

| | |
|---|---|
| Slides used | 352 of 500 eligible (318 negative slides + 34 outlined tumour slides; ITC and outline-free tumour slides excluded); 351 contribute tiles |
| Tiles | 11,780 extracted, **11,763 pass QC** (5,890 tumour, 5,873 normal) |
| Sampling | ≤ 500 tumour tiles per slide; normals 1:1 with tumour per center, spread equally across slides; seed 42 |
| Splits | 98 patients with usable tiles → 58 train / 21 validation / 19 test, split within each center; every center's test set holds ≥ 100 tumour tiles |
| Freeze | master SHA256 `1a98ed80…d192381` |

Splits are derived from a hashed list of all patients (`patient_universe.txt`) with a seed
derived from it, so they reproduce exactly. No patient appears in more than one split.

## Reproduce

Two conda environments: `stainrob` (`environment.yml`, exact versions; splits, training,
tests) and a separate environment with OpenSlide for the slide-reading scripts.

```bash
conda env create -f environment.yml && conda activate stainrob
pip install -e .
pytest tests                      # full test suite, no slides needed
```

With CAMELYON17 downloaded (edit the paths at the top of `configs/tiling_camelyon17.yaml`):

```bash
OUT=/path/to/camelyon17/derived/tiles_v1      # = out_dir in the config
python scripts/download_camelyon17.py --output /path/to/camelyon17/raw --create-metadata
python scripts/tiling_plan.py    --config configs/tiling_camelyon17.yaml      # candidate tiles
python scripts/tiling_extract.py --config configs/tiling_camelyon17.yaml      # tiles + QC
python scripts/create_splits.py --metadata $OUT/split_input.csv --output $OUT/splits \
    --project-id CAMELYON17 --patient-id-column patient_id --group-column center \
    --patient-label-rule any --weight-column tumour_tiles --min-test-weight 100
python scripts/freeze_dataset.py --config configs/tiling_camelyon17.yaml          # freeze
python scripts/freeze_dataset.py --config configs/tiling_camelyon17.yaml --verify --full
```

A run that reproduces the same tiles produces the same `freeze.json` hashes.

Then, per condition, fold and seed (scoring training tiles only until the plan is committed):

```bash
python scripts/calibrate_shifts.py --tiles $OUT --out configs/shift_doses_camelyon17.json
python scripts/train.py --tiles $OUT --runs /path/to/runs --condition E1 --holdout 0 --seed 42 --train-only
python scripts/extract_tier1.py --tiles $OUT --out /path/to/tier1 --subset train-split
```

## Repository map

```
configs/                 tiling/QC/split settings; synthetic-shift dose table
data/camelyon17/         record of the frozen CAMELYON17 tile set (hashes, splits, manifest)
docs/analysis_plan_DRAFT.md   the analysis plan under review (not yet in force)
docs/research_log.md     problems met, their causes and resolutions, and design decisions
docs/figures/            checks on real tiles
scripts/                 command-line steps (tile, split, freeze, train, calibrate shifts, extract features)
src/stain_benchmark/     data, normalization, models, training, monitor features, evaluation
tests/                   pytest suite
```

## Open questions for review

Listed at the end of the [analysis plan draft](docs/analysis_plan_DRAFT.md) and in the research log:

- **Dose yardstick (ISS-042).** SSIM could not put stain changes and blur on one scale; doses are matched
  by colour difference instead. Any yardstick embeds a judgement about "equal size", so the plan also
  proposes testing directly whether the monitor predicts harm better than image-difference measures.
- **Generic-encoder tier (ISS-043).** The ImageNet ResNet-50 used for Tier 1b starts from the same
  weights as the classifier it monitors; an alternative encoder (DenseNet-121) is ready.
- **Small rehearsal dataset (ISS-017).** 4–7 tumour patients per hospital limit what the five real
  hospitals can show; this is why the controlled synthetic arm carries the inference.
- **TCGA task (ISS-012).** Tumour-vs-normal is confounded with frozen vs FFPE preparation.

## Related

- [`wsidrift`](https://pypi.org/project/wsidrift/) — the label-free Tier 1 / Tier 2 monitor,
  as a separate package (PyPI 0.1.1; slide-level calibration is on its main branch, unreleased).

## License

MIT. Author: Abhishek Thakur.
