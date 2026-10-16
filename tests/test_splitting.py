"""
Tests for Patient-Level Data Splitting

=============================================================================
These tests verify the most critical component of the benchmark:
ensuring NO DATA LEAKAGE between train/val/test splits.

Includes tests for:
1. CAMELYON16: 1:1 slide-patient mapping
2. TCGA: Multiple slides per patient (repeated measures)
3. Patient-level aggregation

Run with: pytest tests/test_splitting.py -v
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stain_benchmark.data.splitting import (
    DataLeakageError,
    RepeatedMeasuresWarning,
    aggregate_predictions_to_patient,
    aggregate_predictions_hierarchical,
    create_patient_splits,
    load_splits,
    load_slide_to_patient_mapping,
    verify_no_leakage,
    verify_split_for_patches,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metadata_camelyon(tmp_path: Path) -> Path:
    """
    Create CAMELYON16-style metadata where slide_id == patient_id.

    This represents the simple case with 1:1 mapping.
    """
    np.random.seed(42)
    n_slides = 50

    data = {
        "slide_id": [f"slide_{i:03d}" for i in range(n_slides)],
        "label": np.random.choice([0, 1], size=n_slides, p=[0.6, 0.4]).tolist(),
    }
    df = pd.DataFrame(data)

    metadata_path = tmp_path / "camelyon_metadata.csv"
    df.to_csv(metadata_path, index=False)

    return metadata_path


@pytest.fixture
def sample_metadata_tcga(tmp_path: Path) -> Path:
    """
    Create TCGA-style metadata where one patient has multiple slides.

    This represents the REPEATED MEASURES problem:
    - 30 unique patients
    - 1-4 slides per patient (total ~60 slides)
    - Must split by patient to avoid leakage
    """
    np.random.seed(42)
    n_patients = 30

    records = []
    slide_counter = 0

    for patient_idx in range(n_patients):
        patient_id = f"TCGA-{patient_idx:02d}"
        patient_label = np.random.choice([0, 1], p=[0.6, 0.4])

        # Each patient has 1-4 slides (repeated measures)
        n_slides_for_patient = np.random.randint(1, 5)

        for slide_idx in range(n_slides_for_patient):
            records.append({
                "slide_id": f"TCGA-{patient_idx:02d}-{slide_idx:02d}Z",
                "patient_id": patient_id,
                "label": patient_label,
            })
            slide_counter += 1

    df = pd.DataFrame(records)
    metadata_path = tmp_path / "tcga_metadata.csv"
    df.to_csv(metadata_path, index=False)

    return metadata_path


@pytest.fixture
def sample_patch_metadata(sample_metadata_camelyon: Path) -> pd.DataFrame:
    """Create patch-level metadata from slide metadata."""
    slide_df = pd.read_csv(sample_metadata_camelyon)

    # Create multiple patches per slide
    records = []
    for _, row in slide_df.iterrows():
        n_patches = np.random.randint(5, 20)
        for i in range(n_patches):
            records.append({
                "patch_id": f"{row['slide_id']}_patch_{i:03d}",
                "slide_id": row["slide_id"],
                "label": row["label"],
            })

    return pd.DataFrame(records)


@pytest.fixture
def sample_predictions_df() -> pd.DataFrame:
    """Create sample slide-level predictions for aggregation tests."""
    return pd.DataFrame({
        "slide_id": [
            "TCGA-01-00Z", "TCGA-01-01Z", "TCGA-01-02Z",  # Patient 1: 3 slides
            "TCGA-02-00Z", "TCGA-02-01Z",                  # Patient 2: 2 slides
            "TCGA-03-00Z",                                  # Patient 3: 1 slide
        ],
        "patient_id": [
            "TCGA-01", "TCGA-01", "TCGA-01",
            "TCGA-02", "TCGA-02",
            "TCGA-03",
        ],
        "prediction": [1, 0, 1, 0, 0, 1],
        "probability": [0.8, 0.3, 0.7, 0.2, 0.4, 0.9],
    })


# =============================================================================
# Tests for verify_no_leakage
# =============================================================================


class TestVerifyNoLeakage:
    """Tests for the leakage verification function."""

    def test_no_leakage_returns_true(self):
        """Verify returns True when no leakage exists."""
        train = ["slide_001", "slide_002", "slide_003"]
        val = ["slide_004", "slide_005"]
        test = ["slide_006", "slide_007"]

        assert verify_no_leakage(train, val, test) is True

    def test_train_val_overlap_raises(self):
        """Verify raises error on train-val overlap."""
        train = ["slide_001", "slide_002", "slide_003"]
        val = ["slide_003", "slide_004"]  # slide_003 overlaps!
        test = ["slide_005", "slide_006"]

        with pytest.raises(DataLeakageError) as exc_info:
            verify_no_leakage(train, val, test)

        assert "Train-Val" in str(exc_info.value)
        assert "slide_003" in str(exc_info.value)

    def test_train_test_overlap_raises(self):
        """Verify raises error on train-test overlap."""
        train = ["slide_001", "slide_002"]
        val = ["slide_003", "slide_004"]
        test = ["slide_001", "slide_005"]  # slide_001 overlaps!

        with pytest.raises(DataLeakageError) as exc_info:
            verify_no_leakage(train, val, test)

        assert "Train-Test" in str(exc_info.value)

    def test_val_test_overlap_raises(self):
        """Verify raises error on val-test overlap."""
        train = ["slide_001", "slide_002"]
        val = ["slide_003", "slide_004"]
        test = ["slide_004", "slide_005"]  # slide_004 overlaps!

        with pytest.raises(DataLeakageError) as exc_info:
            verify_no_leakage(train, val, test)

        assert "Val-Test" in str(exc_info.value)

    def test_empty_splits_no_error(self):
        """Verify handles empty splits correctly."""
        train = ["slide_001", "slide_002"]
        val = []
        test = ["slide_003"]

        # Should not raise (empty sets have no overlap)
        assert verify_no_leakage(train, val, test) is True

    def test_duplicate_within_split_raises(self):
        """Verify raises error on duplicates within a split."""
        train = ["slide_001", "slide_001", "slide_002"]  # Duplicate!
        val = ["slide_003"]
        test = ["slide_004"]

        with pytest.raises(DataLeakageError) as exc_info:
            verify_no_leakage(train, val, test)

        assert "Duplicate" in str(exc_info.value)

    def test_level_parameter_in_error_message(self):
        """Verify level parameter is included in error messages."""
        train = ["patient_001", "patient_002"]
        val = ["patient_002", "patient_003"]  # Overlap!
        test = ["patient_004"]

        with pytest.raises(DataLeakageError) as exc_info:
            verify_no_leakage(train, val, test, level="patient")

        assert "PATIENT" in str(exc_info.value)


# =============================================================================
# Tests for CAMELYON16 (1:1 slide-patient)
# =============================================================================


class TestCreatePatientSplitsCAMELYON:
    """Tests for CAMELYON16-style splitting (slide_id == patient_id)."""

    def test_creates_correct_number_of_splits(
        self, sample_metadata_camelyon: Path, tmp_path: Path
    ):
        """Verify creates train/val/test with correct counts."""
        output_dir = tmp_path / "splits"

        splits = create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )

        # Should have both slide and patient keys (identical for CAMELYON16)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert "train_patients" in splits

        # Check approximate ratios
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 50  # All slides assigned

        train_ratio = len(splits["train"]) / total
        assert 0.65 <= train_ratio <= 0.75  # 70% ± 5%

    def test_ratios_must_sum_to_one(
        self, sample_metadata_camelyon: Path, tmp_path: Path
    ):
        """Verify raises error if ratios don't sum to 1."""
        output_dir = tmp_path / "splits"

        with pytest.raises(ValueError) as exc_info:
            create_patient_splits(
                metadata_path=sample_metadata_camelyon,
                output_dir=output_dir,
                train_ratio=0.70,
                val_ratio=0.20,
                test_ratio=0.20,  # Total = 1.10
                seed=42,
            )

        assert "must sum to 1.0" in str(exc_info.value)

    def test_no_leakage_in_created_splits(
        self, sample_metadata_camelyon: Path, tmp_path: Path
    ):
        """Verify created splits have no leakage."""
        output_dir = tmp_path / "splits"

        splits = create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir,
            seed=42,
        )

        # This should not raise
        verify_no_leakage(splits["train"], splits["val"], splits["test"])

    def test_reproducibility_with_same_seed(
        self, sample_metadata_camelyon: Path, tmp_path: Path
    ):
        """Verify same seed produces same splits."""
        output_dir1 = tmp_path / "splits1"
        output_dir2 = tmp_path / "splits2"

        splits1 = create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir1,
            seed=42,
        )

        splits2 = create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir2,
            seed=42,
        )

        assert set(splits1["train"]) == set(splits2["train"])
        assert set(splits1["val"]) == set(splits2["val"])
        assert set(splits1["test"]) == set(splits2["test"])

    def test_creates_output_files(
        self, sample_metadata_camelyon: Path, tmp_path: Path
    ):
        """Verify creates CSV files and metadata."""
        output_dir = tmp_path / "splits"

        create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir,
            seed=42,
        )

        assert (output_dir / "train.csv").exists()
        assert (output_dir / "val.csv").exists()
        assert (output_dir / "test.csv").exists()
        assert (output_dir / "split_metadata.json").exists()
        assert (output_dir / "slide_to_patient_mapping.csv").exists()


# =============================================================================
# Tests for TCGA (Multiple Slides per Patient - Repeated Measures)
# =============================================================================


class TestCreatePatientSplitsTCGA:
    """
    Tests for TCGA-style splitting (multiple slides per patient).

    This is the CRITICAL test for the repeated measures problem!
    """

    def test_splits_by_patient_not_slide(
        self, sample_metadata_tcga: Path, tmp_path: Path
    ):
        """
        CRITICAL TEST: Verify splits at patient level, not slide level.

        All slides from a patient MUST be in the same split.
        """
        output_dir = tmp_path / "splits"

        splits = create_patient_splits(
            metadata_path=sample_metadata_tcga,
            output_dir=output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id",  # Explicitly split by patient
            seed=42,
        )

        # Load original metadata
        df = pd.read_csv(sample_metadata_tcga)

        # For each patient, check ALL their slides are in the SAME split
        for patient_id in df["patient_id"].unique():
            patient_slides = set(df[df["patient_id"] == patient_id]["slide_id"])

            # Check which split contains this patient's slides
            in_train = patient_slides & set(splits["train"])
            in_val = patient_slides & set(splits["val"])
            in_test = patient_slides & set(splits["test"])

            # Patient slides should be in EXACTLY ONE split
            splits_with_patient = sum([
                len(in_train) > 0,
                len(in_val) > 0,
                len(in_test) > 0,
            ])

            assert splits_with_patient == 1, (
                f"Patient {patient_id} has slides in {splits_with_patient} splits! "
                f"(train: {len(in_train)}, val: {len(in_val)}, test: {len(in_test)})"
            )

            # ALL slides from this patient should be in that split
            if len(in_train) > 0:
                assert in_train == patient_slides, (
                    f"Patient {patient_id} has incomplete slides in train"
                )
            elif len(in_val) > 0:
                assert in_val == patient_slides, (
                    f"Patient {patient_id} has incomplete slides in val"
                )
            else:
                assert in_test == patient_slides, (
                    f"Patient {patient_id} has incomplete slides in test"
                )

    def test_no_patient_leakage(self, sample_metadata_tcga: Path, tmp_path: Path):
        """Verify no patient appears in multiple splits."""
        output_dir = tmp_path / "splits"

        splits = create_patient_splits(
            metadata_path=sample_metadata_tcga,
            output_dir=output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id",
            seed=42,
        )

        # Verify at patient level
        verify_no_leakage(
            splits["train_patients"],
            splits["val_patients"],
            splits["test_patients"],
            level="patient"
        )

    def test_detects_repeated_measures(
        self, sample_metadata_tcga: Path, tmp_path: Path
    ):
        """Verify logs warning about repeated measures."""
        output_dir = tmp_path / "splits"

        # This should log a warning about repeated measures
        splits = create_patient_splits(
            metadata_path=sample_metadata_tcga,
            output_dir=output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id",
            seed=42,
        )

        # Check metadata records repeated measures
        import json
        with open(output_dir / "split_metadata.json") as f:
            metadata = json.load(f)

        assert metadata["repeated_measures"] is True
        assert metadata["max_slides_per_patient"] >= 2

    def test_warns_if_splitting_by_slide_with_repeated_measures(
        self, sample_metadata_tcga: Path, tmp_path: Path
    ):
        """
        Verify warns if user tries to split by slide_id when patients have multiple slides.

        This is a common mistake that causes data leakage!
        """
        output_dir = tmp_path / "splits"

        with pytest.warns(RepeatedMeasuresWarning):
            create_patient_splits(
                metadata_path=sample_metadata_tcga,
                output_dir=output_dir,
                slide_id_column="slide_id",
                patient_id_column=None,  # Treating slide_id as patient_id (WRONG!)
                seed=42,
                warn_repeated_measures=True,
            )

    def test_patient_level_stratification(
        self, sample_metadata_tcga: Path, tmp_path: Path
    ):
        """Verify stratification works at patient level, not slide level."""
        output_dir = tmp_path / "splits"

        splits = create_patient_splits(
            metadata_path=sample_metadata_tcga,
            output_dir=output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id",
            stratify=True,
            seed=42,
        )

        # Get patient-level labels
        df = pd.read_csv(sample_metadata_tcga)
        patient_labels = df.groupby("patient_id")["label"].first()
        overall_ratio = patient_labels.mean()

        # Check each split has similar patient-level ratio
        for split_key in ["train_patients", "val_patients", "test_patients"]:
            split_patients = splits[split_key]
            split_labels = [patient_labels[p] for p in split_patients]
            split_ratio = np.mean(split_labels)

            # Allow 25% relative tolerance (smaller sample sizes)
            assert abs(split_ratio - overall_ratio) < 0.25, (
                f"{split_key} has patient ratio {split_ratio:.2f}, "
                f"expected ~{overall_ratio:.2f}"
            )


# =============================================================================
# Tests for Aggregation Functions
# =============================================================================


class TestAggregation:
    """Tests for patient-level prediction aggregation."""

    def test_aggregate_mean(self, sample_predictions_df: pd.DataFrame):
        """Test mean aggregation strategy."""
        mapping = sample_predictions_df[["slide_id", "patient_id"]].drop_duplicates()

        result = aggregate_predictions_to_patient(
            predictions_df=sample_predictions_df,
            slide_to_patient_mapping=mapping,
            aggregation="mean",
        )

        # Patient 1: probs [0.8, 0.3, 0.7] -> mean 0.6 -> pred 1
        patient1 = result[result["patient_id"] == "TCGA-01"].iloc[0]
        assert np.isclose(patient1["probability"], 0.6, atol=0.01)
        assert patient1["prediction"] == 1

        # Patient 2: probs [0.2, 0.4] -> mean 0.3 -> pred 0
        patient2 = result[result["patient_id"] == "TCGA-02"].iloc[0]
        assert np.isclose(patient2["probability"], 0.3, atol=0.01)
        assert patient2["prediction"] == 0

    def test_aggregate_max(self, sample_predictions_df: pd.DataFrame):
        """Test max aggregation strategy (most confident positive)."""
        mapping = sample_predictions_df[["slide_id", "patient_id"]].drop_duplicates()

        result = aggregate_predictions_to_patient(
            predictions_df=sample_predictions_df,
            slide_to_patient_mapping=mapping,
            aggregation="max",
        )

        # Patient 1: probs [0.8, 0.3, 0.7] -> max 0.8 -> pred 1
        patient1 = result[result["patient_id"] == "TCGA-01"].iloc[0]
        assert np.isclose(patient1["probability"], 0.8, atol=0.01)

    def test_aggregate_majority(self, sample_predictions_df: pd.DataFrame):
        """Test majority vote aggregation."""
        mapping = sample_predictions_df[["slide_id", "patient_id"]].drop_duplicates()

        result = aggregate_predictions_to_patient(
            predictions_df=sample_predictions_df,
            slide_to_patient_mapping=mapping,
            aggregation="majority",
        )

        # Patient 1: preds [1, 0, 1] -> majority 1
        patient1 = result[result["patient_id"] == "TCGA-01"].iloc[0]
        assert patient1["prediction"] == 1

        # Patient 2: preds [0, 0] -> majority 0
        patient2 = result[result["patient_id"] == "TCGA-02"].iloc[0]
        assert patient2["prediction"] == 0

    def test_aggregate_any(self, sample_predictions_df: pd.DataFrame):
        """Test 'any' aggregation (high sensitivity)."""
        mapping = sample_predictions_df[["slide_id", "patient_id"]].drop_duplicates()

        result = aggregate_predictions_to_patient(
            predictions_df=sample_predictions_df,
            slide_to_patient_mapping=mapping,
            aggregation="any",
        )

        # Patient 1: preds [1, 0, 1] -> any=1
        patient1 = result[result["patient_id"] == "TCGA-01"].iloc[0]
        assert patient1["prediction"] == 1

        # Patient 2: preds [0, 0] -> any=0
        patient2 = result[result["patient_id"] == "TCGA-02"].iloc[0]
        assert patient2["prediction"] == 0

    def test_tracks_slide_count(self, sample_predictions_df: pd.DataFrame):
        """Verify aggregation tracks number of slides per patient."""
        mapping = sample_predictions_df[["slide_id", "patient_id"]].drop_duplicates()

        result = aggregate_predictions_to_patient(
            predictions_df=sample_predictions_df,
            slide_to_patient_mapping=mapping,
            aggregation="mean",
        )

        # Patient 1: 3 slides
        patient1 = result[result["patient_id"] == "TCGA-01"].iloc[0]
        assert patient1["n_slides"] == 3

        # Patient 3: 1 slide
        patient3 = result[result["patient_id"] == "TCGA-03"].iloc[0]
        assert patient3["n_slides"] == 1


# =============================================================================
# Tests for load_splits
# =============================================================================


class TestLoadSplits:
    """Tests for loading saved splits."""

    def test_load_matches_create(
        self, sample_metadata_camelyon: Path, tmp_path: Path
    ):
        """Verify loaded splits match created splits."""
        output_dir = tmp_path / "splits"

        original = create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir,
            seed=42,
        )

        loaded = load_splits(output_dir)

        assert set(original["train"]) == set(loaded["train"])
        assert set(original["val"]) == set(loaded["val"])
        assert set(original["test"]) == set(loaded["test"])

    def test_load_verifies_on_load(self, tmp_path: Path):
        """Verify loading checks for leakage."""
        output_dir = tmp_path / "splits"
        output_dir.mkdir()

        # Create corrupted splits with leakage
        pd.DataFrame({"slide_id": ["a", "b", "c"]}).to_csv(
            output_dir / "train.csv", index=False
        )
        pd.DataFrame({"slide_id": ["c", "d"]}).to_csv(  # 'c' overlaps!
            output_dir / "val.csv", index=False
        )
        pd.DataFrame({"slide_id": ["e", "f"]}).to_csv(
            output_dir / "test.csv", index=False
        )

        with pytest.raises(Exception):  # SplitVerificationError
            load_splits(output_dir)

    def test_load_slide_to_patient_mapping(
        self, sample_metadata_tcga: Path, tmp_path: Path
    ):
        """Verify can load slide-to-patient mapping."""
        output_dir = tmp_path / "splits"

        create_patient_splits(
            metadata_path=sample_metadata_tcga,
            output_dir=output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id",
            seed=42,
        )

        mapping = load_slide_to_patient_mapping(
            output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id",
        )

        # Check mapping is correct
        assert "TCGA-00-00Z" in mapping
        assert mapping["TCGA-00-00Z"] == "TCGA-00"


# =============================================================================
# Tests for verify_split_for_patches
# =============================================================================


class TestVerifySplitForPatches:
    """Tests for assigning patches to splits."""

    def test_assigns_patches_correctly(
        self, sample_metadata_camelyon: Path, sample_patch_metadata: pd.DataFrame, tmp_path: Path
    ):
        """Verify patches are assigned to same split as their slide."""
        output_dir = tmp_path / "splits"

        splits = create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir,
            seed=42,
        )

        patch_splits = verify_split_for_patches(
            patch_metadata=sample_patch_metadata,
            splits=splits,
        )

        # Check each patch is in correct split
        for split_name, split_df in patch_splits.items():
            expected_slides = set(splits[split_name])

            # All patches should come from slides in this split
            actual_slides = set(split_df["slide_id"].unique())
            assert actual_slides.issubset(expected_slides), (
                f"Patches in {split_name} come from slides not in split"
            )

    def test_no_patch_leakage(
        self, sample_metadata_camelyon: Path, sample_patch_metadata: pd.DataFrame, tmp_path: Path
    ):
        """Verify no patches from same slide in different splits."""
        output_dir = tmp_path / "splits"

        splits = create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir,
            seed=42,
        )

        patch_splits = verify_split_for_patches(
            patch_metadata=sample_patch_metadata,
            splits=splits,
        )

        # Get slide sets for each split
        train_slides = set(patch_splits["train"]["slide_id"].unique())
        val_slides = set(patch_splits["val"]["slide_id"].unique())
        test_slides = set(patch_splits["test"]["slide_id"].unique())

        # Verify no overlap
        assert len(train_slides & val_slides) == 0, "Train-Val slide overlap!"
        assert len(train_slides & test_slides) == 0, "Train-Test slide overlap!"
        assert len(val_slides & test_slides) == 0, "Val-Test slide overlap!"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_camelyon_workflow(
        self, sample_metadata_camelyon: Path, tmp_path: Path
    ):
        """Test complete CAMELYON16 workflow."""
        output_dir = tmp_path / "splits"

        # 1. Create splits
        splits = create_patient_splits(
            metadata_path=sample_metadata_camelyon,
            output_dir=output_dir,
            seed=42,
            stratify=True,
        )

        # 2. Verify no leakage
        verify_no_leakage(splits["train"], splits["val"], splits["test"])

        # 3. Load splits
        loaded = load_splits(output_dir)

        # 4. Verify loaded matches original
        for split_name in ["train", "val", "test"]:
            assert set(splits[split_name]) == set(loaded[split_name])

        # 5. Check all slides are assigned
        all_slides = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        df = pd.read_csv(sample_metadata_camelyon)
        expected_slides = set(df["slide_id"])
        assert all_slides == expected_slides

    def test_full_tcga_workflow(self, sample_metadata_tcga: Path, tmp_path: Path):
        """
        Test complete TCGA workflow with repeated measures handling.

        This is the CRITICAL integration test for proper patient-level splitting.
        """
        output_dir = tmp_path / "splits"

        # 1. Create splits with patient_id_column
        splits = create_patient_splits(
            metadata_path=sample_metadata_tcga,
            output_dir=output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id",
            seed=42,
            stratify=True,
        )

        # 2. Verify no patient leakage
        verify_no_leakage(
            splits["train_patients"],
            splits["val_patients"],
            splits["test_patients"],
            level="patient"
        )

        # 3. Verify no slide leakage
        verify_no_leakage(splits["train"], splits["val"], splits["test"], level="slide")

        # 4. Load mapping and verify
        mapping = load_slide_to_patient_mapping(
            output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id"
        )

        # 5. Verify all slides from each patient are in same split
        df = pd.read_csv(sample_metadata_tcga)
        for patient_id in splits["train_patients"]:
            patient_slides = df[df["patient_id"] == patient_id]["slide_id"].tolist()
            for slide in patient_slides:
                assert slide in splits["train"], (
                    f"Slide {slide} from patient {patient_id} not in train split"
                )

        # 6. Check all patients are assigned
        all_patients = (
            set(splits["train_patients"]) |
            set(splits["val_patients"]) |
            set(splits["test_patients"])
        )
        expected_patients = set(df["patient_id"])
        assert all_patients == expected_patients

    def test_aggregation_workflow(
        self, sample_metadata_tcga: Path, tmp_path: Path
    ):
        """Test prediction aggregation workflow for TCGA."""
        output_dir = tmp_path / "splits"

        # 1. Create splits
        splits = create_patient_splits(
            metadata_path=sample_metadata_tcga,
            output_dir=output_dir,
            slide_id_column="slide_id",
            patient_id_column="patient_id",
            seed=42,
        )

        # 2. Simulate slide-level predictions for test set
        np.random.seed(42)
        test_slides = splits["test"]
        predictions_df = pd.DataFrame({
            "slide_id": test_slides,
            "prediction": np.random.randint(0, 2, len(test_slides)),
            "probability": np.random.rand(len(test_slides)),
        })

        # 3. Aggregate to patient level
        patient_preds = aggregate_predictions_to_patient(
            predictions_df=predictions_df,
            slide_to_patient_mapping=output_dir / "slide_to_patient_mapping.csv",
            slide_id_column="slide_id",
            patient_id_column="patient_id",
            aggregation="mean",
        )

        # 4. Verify one prediction per patient
        assert len(patient_preds) == len(splits["test_patients"])

        # 5. Verify all test patients have predictions
        assert set(patient_preds["patient_id"]) == set(splits["test_patients"])


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# Institution-grouped splitting + leave-one-group-out folds
# =============================================================================
@pytest.fixture
def grouped_metadata(tmp_path: Path) -> Path:
    """Slide table across 3 centers; per center 5 tumour patients with skewed tumour-tile
    weights (one metastatic node each) and 15 negative patients; 5 slides per patient."""
    rows = []
    for center in range(3):
        weights = [500, 500, 110, 6, 3]
        for p in range(20):
            pid = f"patient_{center}{p:02d}"
            for node in range(5):
                tumour = p < 5 and node == 0
                rows.append({"slide_id": f"{pid}_node_{node}", "patient_id": pid, "center": center,
                             "label": int(tumour), "tumour_tiles": weights[p] if tumour else 0})
    path = tmp_path / "grouped.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestGroupedSplits:
    def _make(self, path, tmp_path, **kw):
        return create_patient_splits(
            metadata_path=path, output_dir=tmp_path / "s", patient_id_column="patient_id",
            group_column="center", patient_label_rule="any", weight_column="tumour_tiles",
            min_test_weight=100, seed=7, **kw)

    def test_any_rule_labels_tumour_patients(self, grouped_metadata, tmp_path):
        self._make(grouped_metadata, tmp_path)
        ps = pd.read_csv(tmp_path / "s" / "patient_splits.csv")
        assert ps.label.sum() == 15                      # 5 tumour patients x 3 centers
        assert {"patient_id", "center", "label", "tumour_tiles", "split"} <= set(ps.columns)

    def test_every_center_has_every_split_with_tumour(self, grouped_metadata, tmp_path):
        self._make(grouped_metadata, tmp_path)
        ps = pd.read_csv(tmp_path / "s" / "patient_splits.csv")
        tum = ps[ps.label == 1]
        assert (tum.groupby("center").split.nunique() == 3).all()
        assert (tum[tum.split == "test"].groupby("center").tumour_tiles.sum() >= 100).all()

    def test_no_leakage_and_metadata(self, grouped_metadata, tmp_path):
        splits = self._make(grouped_metadata, tmp_path)
        verify_no_leakage(splits["train_patients"], splits["val_patients"], splits["test_patients"], level="patient")
        meta = json.loads((tmp_path / "s" / "split_metadata.json").read_text())
        assert meta["group_column"] == "center" and meta["patient_label_rule"] == "any"
        assert set(meta["groups"]) == {"0", "1", "2"}

    def test_deterministic(self, grouped_metadata, tmp_path):
        a = self._make(grouped_metadata, tmp_path / "a")
        b = self._make(grouped_metadata, tmp_path / "b")
        assert a["test_patients"] == b["test_patients"]

    def test_patient_in_two_groups_rejected(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"slide_id": ["a", "b"], "patient_id": ["x", "x"], "center": [0, 1], "label": [0, 1]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="span"):
            create_patient_splits(metadata_path=p, output_dir=tmp_path / "s", patient_id_column="patient_id", group_column="center")

    def test_loco_fold_roles(self, grouped_metadata, tmp_path):
        from stain_benchmark.data.splitting import loco_fold
        self._make(grouped_metadata, tmp_path)
        ps = pd.read_csv(tmp_path / "s" / "patient_splits.csv")
        tiles = pd.read_csv(grouped_metadata).rename(columns={"center": "group"})
        role = loco_fold(tiles, ps, holdout="2")
        assert set(role[tiles.group == 2]) == {"test_cross"}
        assert set(role[tiles.group != 2]) == {"train", "val", "test_in"}
        assert (pd.DataFrame({"p": tiles.patient_id, "r": role}).groupby("p").r.nunique() == 1).all()
