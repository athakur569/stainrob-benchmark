# stainrob-benchmark

Research code for a PhD dissertation (Data Science, Harrisburg University of Science and
Technology) on **label-free validity monitoring for computational pathology**: detecting
when stain or scanner variation at a new site has made a deployed model's outputs
unreliable, without needing new labels at that site.

> **Status: work in progress (October 2026).** The data pipeline for CAMELYON17 is built,
> tested and frozen. Model training and all experiments (E1–E9) have **not been run yet**;
> this repository contains **no results**. See [What exists today](#what-exists-today).

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

The method assumes **covariate shift**: image appearance changes, while the relationship
between tissue and diagnosis does not. That fits stain and scanner variation. It does not
cover changes in label definitions or patient population.

## Data

Public, de-identified data only (IRB: Exempt).

- **CAMELYON17** (Paper 1): lymph-node H&E slides, 5 centers, 100 patients, 500 slides,
  slide-level stage labels, and tumour outlines on 50 slides.
- **TCGA** (BRCA, COAD, LUAD; later): deferred. In these cohorts every normal-tissue
  slide is a frozen section, so tumour-vs-normal is confounded with tissue preparation
  (see `docs/research_log.md`, ISS-012).

Slides are not in this repository. `scripts/download_camelyon17.py` registers a local
copy and builds the slide table.

## What exists today

| Step | State | Where |
|---|---|---|
| One slide table for every dataset (slide, patient, center, label, role) | Built, tested | `data/metadata.py` |
| Tiling: 0.5 µm/px, 256 px tiles, tissue ≥ 70%, tumour ≥ 50% inside an outline, normals ≥ 128 µm from any outline | Built, tested | `data/tiling.py`, `scripts/tiling_*.py` |
| Quality control (blur, entropy) | Built, tested | `data/tiling.py` |
| Patient-level splits within each center; leave-one-center-out folds | Built, tested | `data/splitting.py`, `scripts/create_splits.py` |
| SHA256 freeze of tiles, config, metadata and splits | Built, tested | `data/freeze.py`, `scripts/freeze_dataset.py` |
| Stain normalisation: Macenko (E2), Reinhard (E3) | Built, tested | `normalization/` |
| ResNet-50 classifier (E1–E3), DANN with gradient reversal (E5) | Model code only; smoke-tested; never trained | `models/` |
| Metrics: AUC, ΔG, per-center AUC, paired tests, Bonferroni | Built, tested | `evaluation/` |
| Dataset loader, training loop, E1–E9 runs | **Not built** | — |
| CycleGAN (E4), SimCLR (E6) | **Not built** | — |
| Tier 1 / Tier 2 monitor | Separate package, `wsidrift` (synthetic tests only) | — |

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

## Repository map

```
configs/                 tiling, QC and split settings (one file per dataset)
data/camelyon17/         record of the frozen CAMELYON17 tile set (hashes, splits, manifest)
docs/research_log.md     problems met, their causes and resolutions, and open design decisions
scripts/                 command-line steps (download, tile, QC review, split, freeze)
src/stain_benchmark/     data, normalization, models, evaluation, reproducibility utilities
tests/                   pytest suite
```

## Known open questions

Recorded in `docs/research_log.md`; the main ones:

- **Statistical power (ISS-036).** Three seeds per condition are too few for the planned
  paired tests (a Wilcoxon test on 3 pairs cannot reach p < 0.05). The analysis plan is
  being revised toward patient-clustered bootstrap intervals (ISS-031).
- **DANN design (ISS-030).** As written, E5 uses unlabelled images from the held-out center
  (domain adaptation) and a different classifier head from E1 (ISS-037).
- **TCGA task (ISS-012).** Tumour-vs-normal is confounded with frozen vs FFPE preparation.

## Related

- [`wsidrift`](https://pypi.org/project/wsidrift/) — the label-free Tier 1 / Tier 2 monitor,
  as a separate package.

## License

MIT. Author: Abhishek Thakur.
