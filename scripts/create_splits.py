#!/usr/bin/env python3
"""
Create Patient-Level Data Splits (robust CLI wrapper)

This wrapper always enables the benchmark-grade determinism features exposed by
src/stain_benchmark/data/splitting.py unless explicitly disabled.

Goals:
- Works cleanly for TCGA, CAMELYON16, CAMELYON17, or similar metadata tables
- Auto-detects patient_id column when possible
- Auto-generates and locks a frozen patient universe file
- Always derives a stable seed by default
- Infers project_id from metadata when possible
- Falls back safely when stratification is not possible
- Supports verify-only mode

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

# Add repo root/src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stain_benchmark.data.splitting import (  # noqa: E402
    create_patient_splits,
    load_splits,
)

console = Console()
logger = logging.getLogger(__name__)


def infer_patient_id_column(df: pd.DataFrame, user_value: str | None) -> str | None:
    """
    Infer patient_id column if not explicitly provided.

    Priority:
    1. user-provided value
    2. common repeated-measures column names
    3. None -> splitting.py will assume slide_id == patient_id
    """
    if user_value:
        return user_value

    candidates = [
        "patient_id",
        "case_id",
        "patient",
        "case",
        "submitter_id",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    return None


def infer_project_id(df: pd.DataFrame, metadata_path: Path, user_value: str | None) -> str:
    """
    Infer project_id from metadata if possible.
    """
    if user_value:
        return user_value

    if "project_id" in df.columns:
        vals = sorted({str(x) for x in df["project_id"].dropna().astype(str).unique() if str(x).strip()})
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            return f"mixed_{len(vals)}projects"

    stem = metadata_path.parent.name.strip()
    return stem if stem else "unknown_project"


def can_stratify(df: pd.DataFrame, patient_id_column: str, label_column: str,
                 patient_label_rule: str = "mode") -> tuple[bool, str]:
    """
    Decide whether stratification is safe.

    We aggregate to patient level first and inspect class counts. Stratification
    is disabled if:
    - label column missing
    - only one class present
    - any class has < 2 patients
    """
    if label_column not in df.columns:
        return False, f"label column '{label_column}' not found"

    def mode_first(x: pd.Series):
        m = x.mode()
        return m.iloc[0] if len(m) else x.iloc[0]

    agg = mode_first if patient_label_rule == "mode" else "max"
    patient_labels = (
        df.groupby(patient_id_column, sort=True)[label_column]
        .agg(agg)
        .reset_index()
    )

    counts = patient_labels[label_column].value_counts(dropna=False).to_dict()
    if len(counts) < 2:
        return False, f"only one patient-level class present: {counts}"

    if min(counts.values()) < 2:
        return False, f"at least one class has <2 patients: {counts}"

    return True, f"patient-level class counts: {counts}"


def ensure_patient_universe(
    df: pd.DataFrame,
    patient_id_column: str,
    output_dir: Path,
    requested_path: str | None,
) -> Path:
    """
    Create or reuse a deterministic frozen patient universe file.

    If requested_path is provided:
    - use it if it exists
    - create it if it does not exist

    Otherwise create:
      output_dir / "patient_universe.txt"
    """
    if requested_path:
        universe_path = Path(requested_path)
    else:
        universe_path = output_dir / "patient_universe.txt"

    universe_path.parent.mkdir(parents=True, exist_ok=True)

    if universe_path.exists():
        logger.info("Using existing patient universe: %s", universe_path)
        return universe_path

    patients = sorted(df[patient_id_column].astype(str).dropna().unique().tolist())
    universe_path.write_text("".join(f"{p}\n" for p in patients))
    logger.info("Created patient universe with %d patients: %s", len(patients), universe_path)
    return universe_path


def print_config_table(config: dict[str, str]) -> None:
    table = Table(show_header=False, box=None)
    for k, v in config.items():
        table.add_row(f"{k}:", v)
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create patient-level train/val/test splits with benchmark-grade determinism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--metadata", type=str, help="Path to metadata CSV")
    parser.add_argument("--output", "-o", type=str, help="Output directory for split files")
    parser.add_argument("--slide-id-column", type=str, default="slide_id")
    parser.add_argument("--patient-id-column", type=str, default=None)
    parser.add_argument("--label-column", type=str, default="label")

    parser.add_argument("--seed", type=int, default=42,
                        help="Fallback seed only used if --no-derive-seed is set")
    parser.add_argument("--no-derive-seed", action="store_true",
                        help="Disable derived seed and use --seed directly")

    parser.add_argument("--project-id", type=str, default=None,
                        help="Explicit project ID; if omitted, inferred from metadata")
    parser.add_argument("--split-version", type=str, default="v1")
    parser.add_argument("--patient-universe", type=str, default=None,
                        help="Path to frozen patient universe txt; if omitted, auto-created")
    parser.add_argument("--no-freeze-universe", action="store_true",
                        help="Do not auto-create/use a frozen patient universe file")

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--no-stratify", action="store_true",
                        help="Force-disable stratification")
    parser.add_argument("--group-column", type=str, default=None,
                        help="Institution column (e.g. center, tss): split within each group")
    parser.add_argument("--patient-label-rule", choices=["mode", "any"], default="mode",
                        help="Patient label from slide labels: majority (mode) or any positive")
    parser.add_argument("--weight-column", type=str, default=None,
                        help="Per-slide weight (e.g. tumour_tiles) used to place positive patients")
    parser.add_argument("--min-test-weight", type=float, default=0.0,
                        help="Minimum positive weight in each group's test split, when achievable")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing split outputs")
    parser.add_argument("--verify-only", type=str, metavar="SPLITS_DIR",
                        help="Only verify an existing splits directory")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console.print("\n" + "=" * 68)
    console.print("[bold cyan]Patient-Level Data Splitting[/bold cyan]")
    console.print("=" * 68)

    # Verify-only mode
    if args.verify_only:
        splits_dir = Path(args.verify_only)
        console.print(f"\n[blue]Verifying splits in:[/blue] {splits_dir}")
        try:
            splits = load_splits(splits_dir, slide_id_column=args.slide_id_column)
            table = Table(title="Verified Split Summary")
            table.add_column("Split", style="cyan")
            table.add_column("Slides", justify="right")
            for name in ["train", "val", "test"]:
                table.add_row(name.upper(), str(len(splits[name])))
            console.print(table)
            console.print("\n[bold green]✓ Splits verified successfully![/bold green]\n")
            return 0
        except Exception as e:
            console.print(f"\n[bold red]✗ Verification failed![/bold red]\n{e}\n")
            logger.exception("Verification failed")
            return 1

    if not args.metadata or not args.output:
        parser.error("--metadata and --output are required unless using --verify-only")

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output)

    if not metadata_path.exists():
        console.print(f"[red]Error:[/red] metadata file not found: {metadata_path}")
        return 1

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        console.print(f"[red]Error:[/red] ratios must sum to 1.0, got {total_ratio:.4f}")
        return 1

    # Load metadata once here so wrapper can make smart decisions
    df = pd.read_csv(metadata_path)

    if args.slide_id_column not in df.columns:
        console.print(f"[red]Error:[/red] slide ID column '{args.slide_id_column}' not found.")
        console.print(f"Available columns: {list(df.columns)}")
        return 1

    patient_id_column = infer_patient_id_column(df, args.patient_id_column)
    project_id = infer_project_id(df, metadata_path, args.project_id)

    # Determine whether this dataset is repeated-measures or 1:1
    if patient_id_column is None:
        patient_id_for_checks = args.slide_id_column
        patient_mode = "1:1 mode (slide_id == patient_id)"
    else:
        if patient_id_column not in df.columns:
            console.print(f"[red]Error:[/red] inferred patient ID column '{patient_id_column}' not found.")
            console.print(f"Available columns: {list(df.columns)}")
            return 1
        patient_id_for_checks = patient_id_column
        patient_mode = f"patient-level mode ({patient_id_column})"

    # Auto-create / use frozen patient universe unless disabled
    patient_universe_path = None
    if not args.no_freeze_universe:
        patient_universe_path = ensure_patient_universe(
            df=df,
            patient_id_column=patient_id_for_checks,
            output_dir=output_dir,
            requested_path=args.patient_universe,
        )

    # Smart stratification decision
    if args.no_stratify:
        stratify = False
        stratify_reason = "disabled explicitly by --no-stratify"
    else:
        stratify, stratify_reason = can_stratify(
            df=df,
            patient_id_column=patient_id_for_checks,
            label_column=args.label_column,
            patient_label_rule=args.patient_label_rule,
        )

    derive_seed = not args.no_derive_seed

    console.print("\n[bold]Configuration:[/bold]")
    print_config_table({
        "Metadata file": str(metadata_path),
        "Output directory": str(output_dir),
        "Project ID": project_id,
        "Split version": args.split_version,
        "Slide ID column": args.slide_id_column,
        "Patient ID column": patient_id_column if patient_id_column else "(auto: slide_id)",
        "Mode": patient_mode,
        "Label column": args.label_column,
        "Derived seed": str(derive_seed),
        "Fallback seed": str(args.seed),
        "Frozen patient universe": str(patient_universe_path) if patient_universe_path else "(disabled)",
        "Train/Val/Test": f"{args.train_ratio:.0%} / {args.val_ratio:.0%} / {args.test_ratio:.0%}",
        "Stratified": f"{stratify} ({stratify_reason})",
        "Group column": str(args.group_column),
        "Patient label rule": args.patient_label_rule,
        "Weight rule": f"{args.weight_column} (test >= {args.min_test_weight:g})" if args.weight_column else "none",
        "Force overwrite": str(args.force),
    })

    console.print("\n[blue]Creating patient-level splits...[/blue]")

    try:
        splits = create_patient_splits(
            metadata_path=metadata_path,
            output_dir=output_dir,
            slide_id_column=args.slide_id_column,
            patient_id_column=patient_id_column,
            label_column=args.label_column,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            stratify=stratify,
            project_id=project_id,
            split_version=args.split_version,
            derive_seed=derive_seed,
            patient_universe_path=patient_universe_path,
            force=args.force,
            group_column=args.group_column,
            patient_label_rule=args.patient_label_rule,
            weight_column=args.weight_column,
            min_test_weight=args.min_test_weight,
        )

        total_slides = len(splits["train"]) + len(splits["val"]) + len(splits["test"])

        console.print("\n[bold green]✓ Splits created successfully![/bold green]")

        table = Table(title="Split Summary")
        table.add_column("Split", style="cyan")
        table.add_column("Slides", justify="right")
        table.add_column("Patients", justify="right")
        table.add_column("Pct", justify="right")

        for name in ["train", "val", "test"]:
            pct = (len(splits[name]) / total_slides * 100) if total_slides else 0.0
            table.add_row(
                name.upper(),
                str(len(splits[name])),
                str(len(splits[f"{name}_patients"])),
                f"{pct:.1f}%",
            )

        console.print(table)
        console.print(f"\n[green]Files saved to:[/green] {output_dir}/")
        console.print("  - train.csv / val.csv / test.csv")
        console.print("  - train_patients.csv / val_patients.csv / test_patients.csv")
        console.print("  - slide_to_patient_mapping.csv")
        console.print("  - split_metadata.json")
        if patient_universe_path:
            console.print(f"  - {Path(patient_universe_path).name}")
        console.print()
        return 0

    except Exception as e:
        console.print(f"\n[bold red]✗ Failed to create splits![/bold red]\n{e}\n")
        logger.exception("Split creation failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())