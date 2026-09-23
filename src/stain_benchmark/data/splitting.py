"""
Patient-Level Data Splitting for CAMELYON16 and TCGA
=============================================================================
CRITICAL: This module handles data splitting at the PATIENT level to prevent
data leakage and address the repeated measures problem.
CAMELYON16: slide_id == patient_id (1:1 mapping)
    → Simple case, split by slide_id
TCGA / CAMELYON17: patient_id → multiple slide_ids (1:N mapping)
    → REPEATED MEASURES PROBLEM!
    → Must split by patient_id, then include ALL slides from that patient
    → Failure to do this causes DATA LEAKAGE and statistical non-independence
Benchmark-grade determinism additions:
- Optional frozen patient universe list (membership + ordering locked)
- Deterministic ordering independent of CSV row order
- Optional derived seed from (project_id, patient_universe_hash, split_version)
- Overwrite guard: refuses to overwrite existing splits unless --force / force=True
=============================================================================
Author: Abhishek Thakur
Date: February 2026
"""
from __future__ import annotations
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
logger = logging.getLogger(__name__)
class DataLeakageError(Exception):
    """Raised when data leakage is detected between splits."""
    pass
class SplitVerificationError(Exception):
    """Raised when split verification fails."""
    pass
class RepeatedMeasuresWarning(UserWarning):
    """Warning for potential repeated measures issues."""
    pass
# =============================================================================
# NEW HELPERS: determinism + guarding
# =============================================================================
def _md5_seed(s: str) -> int:
    """Stable int seed in [0, 2^32) derived from a string."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
def _read_patient_universe(path: str | Path) -> list[str]:
    """Read patient IDs (one per line). Preserves file order."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"patient universe file not found: {p}")
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    patients = [ln for ln in lines if ln]
    if not patients:
        raise ValueError(f"patient universe file is empty: {p}")
    return patients
def _file_sha256(path: Path) -> str:
    """Hash file contents without loading entire file if huge."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
# =============================================================================
# Main Split Creation Function (UPDATED)
# =============================================================================
def create_patient_splits(
    metadata_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    # Column configuration
    slide_id_column: str = "slide_id",
    patient_id_column: Optional[str] = None,
    label_column: str = "label",
    # Options
    stratify: bool = True,
    verify: bool = True,
    warn_repeated_measures: bool = True,
    # --- NEW: benchmark-grade determinism / guarding ---
    patient_universe_path: Optional[str | Path] = None,
    project_id: str = "unknown_project",
    split_version: str = "v1",
    derive_seed: bool = False,
    force: bool = False,
) -> dict[str, list[str]]:
    """
    Create strict patient-level splits for CAMELYON16 / TCGA / CAMELYON17 with:
      - optional frozen patient universe (membership + ordering locked)
      - deterministic ordering (independent of CSV/API row order)
      - derived seed from experiment identity
      - overwrite guard (refuses to overwrite existing splits unless force=True)
    IMPORTANT:
      - TCGA: set patient_id_column='patient_id' (or your column)
      - CAMELYON16: leave patient_id_column=None
      - CAMELYON17: set patient_id_column='patient_id' (or your column)
    """
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_meta_path = output_dir / "split_metadata.json"
    if split_meta_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing splits: {split_meta_path}. "
            f"Pass force=True / --force to overwrite intentionally."
        )
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.4f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )
    # -------------------------------------------------------------------------
    # Load and validate metadata
    # -------------------------------------------------------------------------
    df = pd.read_csv(metadata_path)
    if slide_id_column not in df.columns:
        raise ValueError(f"Column '{slide_id_column}' not found. Available: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found. Available: {list(df.columns)}")
    if patient_id_column is None:
        patient_id_column = slide_id_column
        logger.info("No patient_id_column specified. Assuming slide_id == patient_id (CAMELYON16 mode)")
    else:
        if patient_id_column not in df.columns:
            raise ValueError(f"Column '{patient_id_column}' not found. Available: {list(df.columns)}")
    # enforce stable types
    df[slide_id_column] = df[slide_id_column].astype(str)
    df[patient_id_column] = df[patient_id_column].astype(str)
    # -------------------------------------------------------------------------
    # Optional: lock membership + order via patient_universe.txt
    # -------------------------------------------------------------------------
    patient_universe: Optional[list[str]] = None
    if patient_universe_path is not None:
        patient_universe = _read_patient_universe(patient_universe_path)
        universe_set = set(patient_universe)
        present = set(df[patient_id_column].unique())
        missing = universe_set - present
        if missing:
            raise ValueError(
                f"metadata missing {len(missing)} patient_ids from patient_universe.txt "
                f"(example: {list(sorted(missing))[:5]})"
            )
        df = df[df[patient_id_column].isin(universe_set)].copy()
    # deterministic ordering independent of csv row order
    df = df.sort_values([patient_id_column, slide_id_column]).reset_index(drop=True)
    logger.info("Creating patient-level splits")
    logger.info("Project: %s | Split version: %s", project_id, split_version)
    logger.info("Ratios: train=%.2f, val=%.2f, test=%.2f", train_ratio, val_ratio, test_ratio)
    logger.info("Splitting by: %s", patient_id_column)
    if patient_universe_path:
        logger.info("Patient universe locked: %s", str(patient_universe_path))
    # -------------------------------------------------------------------------
    # Detect repeated measures
    # -------------------------------------------------------------------------
    slides_per_patient = df.groupby(patient_id_column)[slide_id_column].nunique()
    max_slides = int(slides_per_patient.max()) if len(slides_per_patient) else 0
    patients_with_multiple = int((slides_per_patient > 1).sum()) if len(slides_per_patient) else 0
    if patients_with_multiple > 0:
        logger.warning(
            "REPEATED MEASURES DETECTED: %d patients have multiple slides (max: %d slides/patient)",
            patients_with_multiple, max_slides
        )
        if warn_repeated_measures and patient_id_column == slide_id_column:
            import warnings
            warnings.warn(
                f"Detected {patients_with_multiple} patients with multiple slides, "
                f"but splitting by slide_id. This may cause data leakage! "
                f"Consider specifying patient_id_column separately.",
                RepeatedMeasuresWarning
            )
    else:
        logger.info("No repeated measures detected (1:1 slide-patient mapping)")
    # -------------------------------------------------------------------------
    # Patient labels + patient->slides mapping (deterministic)
    # -------------------------------------------------------------------------
    def _mode_first(x: pd.Series):
        m = x.mode()
        return m.iloc[0] if len(m) else x.iloc[0]
    patient_labels = (
        df.groupby(patient_id_column, sort=True)
          .agg({
              label_column: _mode_first,
              slide_id_column: lambda s: sorted(s.astype(str).tolist())
          })
          .reset_index()
    )
    patient_ids = patient_labels[patient_id_column].astype(str).tolist()
    labels = patient_labels[label_column].tolist()
    patient_to_slides = dict(zip(patient_ids, patient_labels[slide_id_column]))
    # If a universe file is provided, enforce EXACT order from that file
    if patient_universe is not None:
        labels_lookup = dict(zip(
            patient_labels[patient_id_column].astype(str),
            patient_labels[label_column]
        ))
        patient_ids = [pid for pid in patient_universe if pid in patient_to_slides]
        labels = [labels_lookup[pid] for pid in patient_ids]
    # Convert string labels to numeric if needed
    if len(labels) > 0 and isinstance(labels[0], str):
        label_map = {"normal": 0, "tumor": 1, "negative": 0, "positive": 1}
        labels = [label_map.get(str(l).lower(), l) for l in labels]
    labels = np.array(labels)
    # -------------------------------------------------------------------------
    # Derived seed (experiment identity)
    # -------------------------------------------------------------------------
    universe_blob = "\n".join(patient_ids) + "\n"
    patient_universe_hash = _sha256_hex(universe_blob)[:16]
    if derive_seed:
        seed_key = f"{project_id}::{patient_universe_hash}::{split_version}"
        seed = _md5_seed(seed_key)
        logger.info("Derived seed=%d from key=%s", seed, seed_key)
    else:
        logger.info("Using provided seed=%d", seed)
    n_patients = len(patient_ids)
    n_slides = len(df[slide_id_column].unique())
    logger.info("Total patients: %d", n_patients)
    logger.info("Total slides: %d", n_slides)
    if len(slides_per_patient):
        logger.info(
            "Slides per patient: min=%d, max=%d, mean=%.2f",
            int(slides_per_patient.min()),
            int(slides_per_patient.max()),
            float(slides_per_patient.mean()),
        )
    logger.info("Label distribution: %s", dict(Counter(labels.tolist())))
    # -------------------------------------------------------------------------
    # Perform stratified split at PATIENT level (same math as before)
    # -------------------------------------------------------------------------
    val_test_ratio = val_ratio + test_ratio
    stratify_labels = labels if stratify else None
    train_patients, val_test_patients, _, val_test_labels = train_test_split(
        patient_ids,
        labels,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )
    test_fraction = test_ratio / val_test_ratio
    stratify_val_test = val_test_labels if stratify else None
    val_patients, test_patients, _, _ = train_test_split(
        val_test_patients,
        val_test_labels,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify_val_test,
    )
    # -------------------------------------------------------------------------
    # Map patients -> slides (critical for TCGA/CAMELYON17)
    # -------------------------------------------------------------------------
    def patients_to_slides(patient_list: list[str]) -> list[str]:
        slides: list[str] = []
        for patient in patient_list:
            slides.extend(patient_to_slides[patient])
        return slides
    train_slides = patients_to_slides(list(train_patients))
    val_slides = patients_to_slides(list(val_patients))
    test_slides = patients_to_slides(list(test_patients))
    splits = {
        "train": train_slides,
        "val": val_slides,
        "test": test_slides,
        "train_patients": list(train_patients),
        "val_patients": list(val_patients),
        "test_patients": list(test_patients),
    }
    logger.info("Split summary:")
    logger.info("  Train: %d patients, %d slides", len(train_patients), len(train_slides))
    logger.info("  Val:   %d patients, %d slides", len(val_patients), len(val_slides))
    logger.info("  Test:  %d patients, %d slides", len(test_patients), len(test_slides))
    # -------------------------------------------------------------------------
    # Verify no leakage (patient + slide)
    # -------------------------------------------------------------------------
    if verify:
        verify_no_leakage(splits["train_patients"], splits["val_patients"], splits["test_patients"], level="patient")
        verify_no_leakage(splits["train"], splits["val"], splits["test"], level="slide")
        logger.info("✓ Leakage verification passed (patient and slide level)")
    # -------------------------------------------------------------------------
    # Save splits
    # -------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_df = pd.DataFrame({slide_id_column: splits[split_name]})
        split_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        logger.info("Saved %s slides to %s", split_name, split_path)
    for split_name in ["train_patients", "val_patients", "test_patients"]:
        split_df = pd.DataFrame({patient_id_column: splits[split_name]})
        split_path = output_dir / f"{split_name.replace('_patients', '')}_patients.csv"
        split_df.to_csv(split_path, index=False)
    mapping_df = df[[slide_id_column, patient_id_column]].drop_duplicates()
    mapping_path = output_dir / "slide_to_patient_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info("Saved slide-to-patient mapping to %s", mapping_path)
    # -------------------------------------------------------------------------
    # Save split metadata (provenance)
    # -------------------------------------------------------------------------
    metadata_hash = _file_sha256(metadata_path)[:16]
    metadata = {
        "project_id": project_id,
        "split_version": split_version,
        "derive_seed": derive_seed,
        "seed": int(seed),
        "patient_universe_hash": patient_universe_hash,
        "patient_universe_path": str(patient_universe_path) if patient_universe_path else None,
        "metadata_hash": metadata_hash,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "stratified": stratify,
        "slide_id_column": slide_id_column,
        "patient_id_column": patient_id_column,
        "n_train_patients": len(splits["train_patients"]),
        "n_val_patients": len(splits["val_patients"]),
        "n_test_patients": len(splits["test_patients"]),
        "n_train_slides": len(splits["train"]),
        "n_val_slides": len(splits["val"]),
        "n_test_slides": len(splits["test"]),
        "repeated_measures": patients_with_multiple > 0,
        "max_slides_per_patient": int(max_slides),
        "source_file": str(metadata_path),
        "checksum": _compute_split_checksum(splits),
    }
    with open(split_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved split metadata to %s", split_meta_path)
    return splits
# =============================================================================
# Verification Functions
# =============================================================================
def verify_no_leakage(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    level: str = "patient",
) -> bool:
    """Assert that no ID appears in multiple splits."""
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    errors = []
    if train_val_overlap:
        errors.append(f"Train-Val {level} overlap ({len(train_val_overlap)}): {list(train_val_overlap)[:5]}")
    if train_test_overlap:
        errors.append(f"Train-Test {level} overlap ({len(train_test_overlap)}): {list(train_test_overlap)[:5]}")
    if val_test_overlap:
        errors.append(f"Val-Test {level} overlap ({len(val_test_overlap)}): {list(val_test_overlap)[:5]}")
    if errors:
        error_msg = (
            "\n" + "=" * 70 + "\n"
            f"CRITICAL: DATA LEAKAGE DETECTED AT {level.upper()} LEVEL!\n"
            "=" * 70 + "\n"
            "Overlaps found between splits:\n\n"
            + "\n".join(f"  • {e}" for e in errors) + "\n\n"
            "This invalidates ALL experimental results.\n"
            "Please recreate splits.\n"
            "=" * 70
        )
        raise DataLeakageError(error_msg)
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        if len(ids) != len(set(ids)):
            # keep deterministic: show unique duplicates only
            seen = set()
            dups = set()
            for x in ids:
                if x in seen:
                    dups.add(x)
                seen.add(x)
            raise DataLeakageError(f"Duplicate {level} IDs found in {name} split: {sorted(dups)[:10]}")
    return True
# =============================================================================
# Load Functions
# =============================================================================
def load_splits(
    splits_dir: str | Path,
    slide_id_column: str = "slide_id",
    patient_id_column: Optional[str] = None,
) -> dict[str, list[str]]:
    """Load previously created splits from CSV files."""
    splits_dir = Path(splits_dir)
    splits: dict[str, list[str]] = {}
    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        df = pd.read_csv(split_path)
        splits[split_name] = df[slide_id_column].astype(str).tolist()
    if patient_id_column:
        for split_name in ["train", "val", "test"]:
            patient_path = splits_dir / f"{split_name}_patients.csv"
            if patient_path.exists():
                df = pd.read_csv(patient_path)
                splits[f"{split_name}_patients"] = df[patient_id_column].astype(str).tolist()
    verify_no_leakage(splits["train"], splits["val"], splits["test"], level="slide")
    logger.info("Loaded splits: train=%d, val=%d, test=%d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))
    return splits
def load_slide_to_patient_mapping(
    splits_dir: str | Path,
    slide_id_column: str = "slide_id",
    patient_id_column: str = "patient_id",
) -> dict[str, str]:
    """Load slide-to-patient mapping as a dictionary."""
    mapping_path = Path(splits_dir) / "slide_to_patient_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    df = pd.read_csv(mapping_path)
    return dict(zip(df[slide_id_column].astype(str), df[patient_id_column].astype(str)))
# =============================================================================
# Patient-level aggregation helpers (unchanged)
# =============================================================================
def aggregate_predictions_to_patient(
    predictions_df: pd.DataFrame,
    slide_to_patient_mapping: pd.DataFrame | str | Path,
    slide_id_column: str = "slide_id",
    patient_id_column: str = "patient_id",
    prediction_column: str = "prediction",
    probability_column: Optional[str] = "probability",
    aggregation: Literal["mean", "max", "majority", "any"] = "mean",
) -> pd.DataFrame:
    """Aggregate slide/patch predictions to patient level."""
    if isinstance(slide_to_patient_mapping, (str, Path)):
        mapping_df = pd.read_csv(slide_to_patient_mapping)
    else:
        mapping_df = slide_to_patient_mapping
    merged = predictions_df.merge(
        mapping_df[[slide_id_column, patient_id_column]],
        on=slide_id_column,
        how="left"
    )
    missing = int(merged[patient_id_column].isna().sum())
    if missing > 0:
        logger.warning("%d slides have no patient mapping", missing)
        merged = merged.dropna(subset=[patient_id_column])
    aggregated = merged.groupby(patient_id_column).agg({
        prediction_column: list,
        **({probability_column: list} if probability_column in merged.columns else {}),
        slide_id_column: "count"
    }).reset_index()
    aggregated = aggregated.rename(columns={slide_id_column: "n_slides"})
    aggregated["slide_predictions"] = aggregated[prediction_column]
    if aggregation == "mean" and probability_column in merged.columns:
        aggregated["probability"] = aggregated[probability_column].apply(np.mean)
        aggregated[prediction_column] = (aggregated["probability"] >= 0.5).astype(int)
    elif aggregation == "max" and probability_column in merged.columns:
        aggregated["probability"] = aggregated[probability_column].apply(np.max)
        aggregated[prediction_column] = (aggregated["probability"] >= 0.5).astype(int)
    elif aggregation == "majority":
        aggregated[prediction_column] = aggregated["slide_predictions"].apply(lambda x: int(np.mean(x) >= 0.5))
    elif aggregation == "any":
        aggregated[prediction_column] = aggregated["slide_predictions"].apply(lambda x: int(any(x)))
    else:
        aggregated[prediction_column] = aggregated["slide_predictions"].apply(lambda x: int(np.mean(x) >= 0.5))
    logger.info("Aggregated %d slides to %d patients using '%s'",
                len(predictions_df), len(aggregated), aggregation)
    return aggregated
def aggregate_predictions_hierarchical(
    patch_predictions: pd.DataFrame,
    slide_id_column: str = "slide_id",
    patient_id_column: str = "patient_id",
    probability_column: str = "probability",
    slide_aggregation: Literal["mean", "max", "percentile"] = "mean",
    patient_aggregation: Literal["mean", "max", "majority", "any"] = "mean",
    percentile: int = 95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hierarchical aggregation: Patch → Slide → Patient."""
    if slide_aggregation == "mean":
        agg_func = np.mean
    elif slide_aggregation == "max":
        agg_func = np.max
    elif slide_aggregation == "percentile":
        agg_func = lambda x: np.percentile(x, percentile)
    else:
        agg_func = np.mean
    slide_preds = patch_predictions.groupby(slide_id_column).agg({
        probability_column: agg_func,
        patient_id_column: "first",
        "patch_id": "count" if "patch_id" in patch_predictions.columns else lambda x: len(x),
    }).reset_index()
    slide_preds = slide_preds.rename(columns={"patch_id": "n_patches"})
    slide_preds["prediction"] = (slide_preds[probability_column] >= 0.5).astype(int)
    patient_preds = aggregate_predictions_to_patient(
        predictions_df=slide_preds,
        slide_to_patient_mapping=slide_preds[[slide_id_column, patient_id_column]],
        slide_id_column=slide_id_column,
        patient_id_column=patient_id_column,
        prediction_column="prediction",
        probability_column=probability_column,
        aggregation=patient_aggregation,
    )
    return slide_preds, patient_preds
# =============================================================================
# Helper: checksum
# =============================================================================
def _compute_split_checksum(splits: dict[str, list[str]]) -> str:
    content = json.dumps(
        {k: sorted(v) for k, v in sorted(splits.items()) if isinstance(v, list)},
        sort_keys=True
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]
# =============================================================================
# CLI (UPDATED)
# =============================================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Create patient-level splits for CAMELYON16 / TCGA / CAMELYON17")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to metadata CSV with slide_id, patient_id (optional), and label")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for split files")
    parser.add_argument("--slide-id-column", type=str, default="slide_id",
                        help="Column name for slide IDs (default: slide_id)")
    parser.add_argument("--patient-id-column", type=str, default=None,
                        help="Column name for patient IDs. If not specified, assumes slide_id == patient_id")
    parser.add_argument("--label-column", type=str, default="label",
                        help="Column name for labels (default: label)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used when --derive-seed is NOT set (default: 42)")
    parser.add_argument("--derive-seed", action="store_true",
                        help="Derive seed from (project_id, patient_universe_hash, split_version)")
    parser.add_argument("--project-id", type=str, default="unknown_project",
                        help="Identifier for provenance + derived seed (e.g., TCGA-BRCA)")
    parser.add_argument("--split-version", type=str, default="v1",
                        help="Protocol/version tag; bump when you intentionally change split logic")
    parser.add_argument("--patient-universe", type=str, default=None,
                        help="Path to frozen patient universe txt (one patient_id per line)")
    parser.add_argument("--train-ratio", type=float, default=0.70,
                        help="Training set ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Test set ratio (default: 0.15)")
    parser.add_argument("--no-stratify", action="store_true",
                        help="Disable stratified splitting by label")
    parser.add_argument("--force", action="store_true",
                        help="Allow overwriting existing split_metadata.json and split csvs")
    args = parser.parse_args()
    splits = create_patient_splits(
        metadata_path=args.metadata,
        output_dir=args.output,
        slide_id_column=args.slide_id_column,
        patient_id_column=args.patient_id_column,
        label_column=args.label_column,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify=not args.no_stratify,
        project_id=args.project_id,
        split_version=args.split_version,
        derive_seed=args.derive_seed,
        patient_universe_path=args.patient_universe,
        force=args.force,
    )
    print("\n✓ Splits created successfully!")
    print(f"  Train: {len(splits['train_patients'])} patients, {len(splits['train'])} slides")
    print(f"  Val:   {len(splits['val_patients'])} patients, {len(splits['val'])} slides")
    print(f"  Test:  {len(splits['test_patients'])} patients, {len(splits['test'])} slides")
