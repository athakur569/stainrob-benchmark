#!/usr/bin/env python3
"""
Download TCGA Whole Slide Images using GDC Data Transfer Tool

=============================================================================
This script downloads TCGA whole-slide images for the stain robustness
benchmark. It uses the GDC API to query available slides and the gdc-client
CLI tool to download them.

IMPORTANT: TCGA has multiple slides per patient (repeated measures problem).
This script creates a proper metadata file with patient_id for splitting.

Key behavior:
- Default mode: diagnostic-only cohort (API Diagnostic Slide + DX barcode filter)
- --all-slides: broad cohort (removes API Diagnostic Slide restriction and DX-only
  barcode restriction), while still keeping only sample codes 01/06/11
- Always refreshes full metadata from the latest query
- Builds both a full manifest and a pending-only manifest after checking local slides/raw
- Supports manifest-only mode for inventory generation without download

Prerequisites:
    - gdc-client installed and in PATH
    - Sufficient disk space (~500GB+ for full datasets)

Usage:
    # Download TCGA-BRCA diagnostic-only cohort
    python scripts/download_tcga.py --project TCGA-BRCA --output data/tcga

    # Download multiple projects
    python scripts/download_tcga.py --project TCGA-BRCA TCGA-LUAD --output data/tcga

    # Download with sample limit (for testing)
    python scripts/download_tcga.py --project TCGA-BRCA --max-cases 50

    # Generate manifest only (no download)
    python scripts/download_tcga.py --project TCGA-BRCA --manifest-only

    # Generate broad slide inventory (DX + TS + BS at barcode layer, 01/06/11 only)
    python scripts/download_tcga.py --project TCGA-BRCA --manifest-only --all-slides
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# GDC API Configuration
# =============================================================================

GDC_API_BASE = "https://api.gdc.cancer.gov"
GDC_FILES_ENDPOINT = f"{GDC_API_BASE}/files"

TCGA_PROJECTS = {
    "TCGA-BRCA": {
        "name": "Breast Invasive Carcinoma",
        "tissue": "Breast",
        "approx_cases": 1098,
        "approx_slides": 1800,
        "approx_size_gb": 450,
    },
    "TCGA-LUAD": {
        "name": "Lung Adenocarcinoma",
        "tissue": "Lung",
        "approx_cases": 585,
        "approx_slides": 1000,
        "approx_size_gb": 250,
    },
    "TCGA-LUSC": {
        "name": "Lung Squamous Cell Carcinoma",
        "tissue": "Lung",
        "approx_cases": 504,
        "approx_slides": 850,
        "approx_size_gb": 210,
    },
    "TCGA-COAD": {
        "name": "Colon Adenocarcinoma",
        "tissue": "Colon",
        "approx_cases": 521,
        "approx_slides": 800,
        "approx_size_gb": 200,
    },
    "TCGA-READ": {
        "name": "Rectum Adenocarcinoma",
        "tissue": "Rectum",
        "approx_cases": 177,
        "approx_slides": 280,
        "approx_size_gb": 70,
    },
    "TCGA-PRAD": {
        "name": "Prostate Adenocarcinoma",
        "tissue": "Prostate",
        "approx_cases": 550,
        "approx_slides": 900,
        "approx_size_gb": 225,
    },
    "TCGA-STAD": {
        "name": "Stomach Adenocarcinoma",
        "tissue": "Stomach",
        "approx_cases": 478,
        "approx_slides": 750,
        "approx_size_gb": 190,
    },
    "TCGA-BLCA": {
        "name": "Bladder Urothelial Carcinoma",
        "tissue": "Bladder",
        "approx_cases": 433,
        "approx_slides": 700,
        "approx_size_gb": 175,
    },
    "TCGA-HNSC": {
        "name": "Head and Neck Squamous Cell Carcinoma",
        "tissue": "Head/Neck",
        "approx_cases": 566,
        "approx_slides": 900,
        "approx_size_gb": 225,
    },
    "TCGA-KIRC": {
        "name": "Kidney Renal Clear Cell Carcinoma",
        "tissue": "Kidney",
        "approx_cases": 611,
        "approx_slides": 1000,
        "approx_size_gb": 250,
    },
}


# =============================================================================
# Cohort v1 — locked include/exclude decision
# =============================================================================

SAMPLE_CODE_META: dict[str, dict[str, Any]] = {
    "01": {"name": "Primary Solid Tumor", "label": 1},
    "06": {"name": "Metastatic", "label": 1},
    "11": {"name": "Solid Tissue Normal", "label": 0},
    "10": {"name": "Blood Derived Normal", "label": -1},
}

DEFAULT_COHORT_SAMPLE_CODES: tuple[str, ...] = ("01", "06", "11")

_SAMPLE_TOKEN_RE = re.compile(r"^\d{2}[A-Z]$")
_PORTION_TOKEN_RE = re.compile(r"^(DX|TS|BS)\d+(\..+)?$", re.IGNORECASE)


# =============================================================================
# Barcode parsing
# =============================================================================


def _tcga_barcode_tokens(file_name: str) -> dict[str, str]:
    """
    Parse TCGA slide filename tokens (best-effort; returns "" if parse fails).

    Expected barcode fragment inside the filename:
        TCGA-XX-YYYY-SSZ-PP-CCCn
        SSZ: sample code token (e.g., 01Z, 06Z, 11Z)
        CCCn: slide portion token (e.g., DX1, TS1, BS1)
    """
    base = file_name.rsplit("/", 1)[-1]
    base = base[:-4] if base.lower().endswith(".svs") else base
    parts = base.split("-")

    sample_token = ""
    portion_token = ""

    for token in parts:
        tok = token.upper().strip()
        if _SAMPLE_TOKEN_RE.fullmatch(tok):
            sample_token = tok
            break

    if not sample_token and len(parts) >= 4:
        sample_token = parts[3].upper().strip()

    for token in parts:
        tok = token.upper().strip()
        if _PORTION_TOKEN_RE.fullmatch(tok):
            portion_token = tok
            break

    if not portion_token and len(parts) >= 6:
        portion_token = parts[5].upper().strip()

    sample_code = sample_token[:2] if len(sample_token) >= 2 else ""
    portion_code = portion_token[:2] if len(portion_token) >= 2 else ""

    return {
        "sample_token": sample_token,
        "sample_code": sample_code,
        "portion_token": portion_token,
        "portion_code": portion_code,
    }


# =============================================================================
# GDC API Functions
# =============================================================================


def check_gdc_client() -> bool:
    """Check if gdc-client is installed and accessible."""
    try:
        result = subprocess.run(
            ["gdc-client", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            console.print(f"[green]✓[/green] gdc-client found: {version}")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    console.print(
        "[red]✗[/red] gdc-client not found!\n"
        "\nInstall instructions:\n"
        "  1. Download from: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool\n"
        "  2. Extract and add to PATH\n"
        "  3. Or use: pip install gdc-client"
    )
    return False


def _paged_fetch(filters_obj: dict[str, Any], fields: list[str]) -> list[dict[str, Any]]:
    """Paginate through the GDC files endpoint and return all hits."""
    all_hits: list[dict[str, Any]] = []
    page_size = 500
    offset = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("fetching file list...", total=None)

        while True:
            params = {
                "filters": json.dumps(filters_obj),
                "fields": ",".join(fields),
                "format": "JSON",
                "size": page_size,
                "from": offset,
            }

            resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            hits = data.get("data", {}).get("hits", [])
            if not hits:
                break

            all_hits.extend(hits)
            offset += len(hits)
            progress.update(task, description=f"fetching file list... ({len(all_hits)} files)")

            total = data.get("data", {}).get("pagination", {}).get("total", 0)
            if offset >= total:
                break

    return all_hits


def query_gdc_slides(
    project_id: str,
    max_cases: Optional[int] = None,
    cohort_sample_codes: tuple[str, ...] = DEFAULT_COHORT_SAMPLE_CODES,
    dx_only: bool = True,
    api_diagnostic_only: bool = True,
) -> list[dict[str, Any]]:
    """
    Query GDC for TCGA slide images and curate stain-robust cohort v1.

    Default:
        - API-level experimental_strategy == "Diagnostic Slide"
        - barcode-level DX only
        - sample codes 01 / 06 / 11

    With --all-slides:
        - no API Diagnostic Slide restriction
        - no barcode DX restriction
        - sample codes 01 / 06 / 11 still enforced
    """
    console.print(f"\n[bold blue]Querying GDC API for {project_id}[/bold blue]")

    base_content = [
        {
            "op": "=",
            "content": {"field": "cases.project.project_id", "value": project_id},
        },
        {
            "op": "=",
            "content": {"field": "data_type", "value": "Slide Image"},
        },
        {
            "op": "=",
            "content": {"field": "data_format", "value": "SVS"},
        },
    ]

    if api_diagnostic_only:
        base_content.append(
            {
                "op": "=",
                "content": {"field": "experimental_strategy", "value": "Diagnostic Slide"},
            }
        )

    base_filters = {"op": "and", "content": base_content}

    fallback_filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {"field": "cases.project.project_id", "value": project_id},
            },
            {
                "op": "=",
                "content": {"field": "data_type", "value": "Slide Image"},
            },
            {
                "op": "=",
                "content": {"field": "data_format", "value": "SVS"},
            },
        ],
    }

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "data_type",
        "data_format",
        "experimental_strategy",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.diagnoses.primary_diagnosis",
        "cases.diagnoses.tumor_stage",
        "cases.samples.sample_type",
        "cases.samples.tissue_type",
    ]

    if api_diagnostic_only:
        console.print("[dim]  Layer 1: querying with experimental_strategy='Diagnostic Slide'...[/dim]")
    else:
        console.print("[dim]  Layer 1: querying all SVS slide images (no experimental_strategy restriction)...[/dim]")

    all_files = _paged_fetch(base_filters, fields)

    if len(all_files) == 0 and api_diagnostic_only:
        console.print(
            "[yellow]  warning:[/yellow] zero hits using experimental_strategy='Diagnostic Slide'. "
            "Retrying without experimental_strategy (fallback)."
        )
        all_files = _paged_fetch(fallback_filters, fields)

    console.print(f"[green]✓[/green] found {len(all_files)} SVS slide files (pre-curation)")

    before = len(all_files)
    kept: list[dict[str, Any]] = []
    counts = {
        "kept": 0,
        "skip_bad_parse": 0,
        "skip_non_dx": 0,
        "skip_sample_code": 0,
    }
    sample_code_hist: dict[str, int] = {}
    portion_code_hist: dict[str, int] = {}

    for f in all_files:
        fn = (f.get("file_name") or "").strip()
        tok = _tcga_barcode_tokens(fn)
        f["_barcode"] = tok

        sc = tok["sample_code"]
        pc = tok["portion_code"]

        sample_code_hist[sc] = sample_code_hist.get(sc, 0) + 1
        portion_code_hist[pc] = portion_code_hist.get(pc, 0) + 1

        if sc == "" or pc == "":
            counts["skip_bad_parse"] += 1
            continue

        if dx_only and pc != "DX":
            counts["skip_non_dx"] += 1
            continue

        if sc not in cohort_sample_codes:
            counts["skip_sample_code"] += 1
            continue

        kept.append(f)

    counts["kept"] = len(kept)

    console.print(
        f"[green]✓[/green] after curation: {len(kept)} files kept "
        f"(skipped: bad_parse={counts['skip_bad_parse']}, "
        f"non_dx={counts['skip_non_dx']}, "
        f"sample_code={counts['skip_sample_code']})"
    )

    if max_cases is not None:
        case_ids: set[str] = set()
        filtered: list[dict[str, Any]] = []
        skipped_no_case = 0

        for f in kept:
            cases = f.get("cases", [])
            if not cases:
                skipped_no_case += 1
                continue
            case_id = cases[0].get("case_id")
            if not case_id:
                skipped_no_case += 1
                continue

            if len(case_ids) < max_cases:
                case_ids.add(case_id)
                filtered.append(f)
            elif case_id in case_ids:
                filtered.append(f)

        if skipped_no_case:
            console.print(
                f"[yellow]  warning: {skipped_no_case} files had no case_id "
                f"and were dropped during max_cases filtering[/yellow]"
            )

        console.print(
            f"[yellow]max_cases={max_cases}: kept {len(filtered)} slides "
            f"from {len(case_ids)} cases[/yellow]"
        )
        kept = filtered

    def _top_k(d: dict[str, int], k: int = 8) -> str:
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
        return ", ".join([f"{a or '??'}:{b}" for a, b in items]) if items else "-"

    table = Table(title=f"TCGA cohort summary — {project_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("pre-curation files", str(before))
    table.add_row("kept files (post-curation)", str(counts["kept"]))
    if max_cases is not None:
        table.add_row("kept files (post max_cases)", str(len(kept)))
    table.add_row("api_diagnostic_only", str(api_diagnostic_only))
    table.add_row("dx_only", str(dx_only))
    table.add_row("sample codes kept", ", ".join(cohort_sample_codes))
    table.add_row("raw portion_code hist (top)", _top_k(portion_code_hist))
    table.add_row("raw sample_code hist (top)", _top_k(sample_code_hist))
    console.print(table)

    return kept


# =============================================================================
# Existing-download detection
# =============================================================================


def collect_existing_slide_filenames(output_dir: Path) -> set[str]:
    """Collect already-present slide filenames from flattened and raw download areas."""
    existing: set[str] = set()

    slides_dir = output_dir / "slides"
    if slides_dir.exists():
        existing.update(p.name for p in slides_dir.glob("*.svs"))
        existing.update(p.name for p in slides_dir.glob("*.SVS"))

    raw_dir = output_dir / "raw"
    if raw_dir.exists():
        for pattern in ("*.svs", "*.SVS"):
            existing.update(p.name for p in raw_dir.rglob(pattern))

    return existing


def split_files_by_existing(
    files: list[dict[str, Any]],
    existing_filenames: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split queried files into already-present vs pending using filename match."""
    existing_files: list[dict[str, Any]] = []
    pending_files: list[dict[str, Any]] = []

    for file_rec in files:
        filename = (file_rec.get("file_name") or "").strip()
        if filename in existing_filenames:
            existing_files.append(file_rec)
        else:
            pending_files.append(file_rec)

    return existing_files, pending_files


def _manifest_row_count(manifest_path: Path) -> int:
    """Return number of data rows in a GDC manifest (excluding header)."""
    if not manifest_path.exists():
        return 0
    with manifest_path.open("r") as f:
        return max(sum(1 for _ in f) - 1, 0)


def print_download_plan(
    all_files: list[dict[str, Any]],
    existing_files: list[dict[str, Any]],
    pending_files: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Print a simple preflight summary of what will download vs be skipped."""
    slides_dir = output_dir / "slides"
    raw_dir = output_dir / "raw"

    table = Table(title="TCGA download preflight")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Queried cohort slides", str(len(all_files)))
    table.add_row("Already present locally", str(len(existing_files)))
    table.add_row("Pending download", str(len(pending_files)))
    table.add_row("slides/ directory", str(slides_dir))
    table.add_row("raw/ directory", str(raw_dir))
    console.print(table)

    if existing_files:
        console.print(
            "[yellow]Skip examples:[/yellow] "
            + ", ".join((f.get("file_name") or "") for f in existing_files[:5])
        )
    if pending_files:
        console.print(
            "[green]Download examples:[/green] "
            + ", ".join((f.get("file_name") or "") for f in pending_files[:5])
        )

    if len(existing_files) + len(pending_files) != len(all_files):
        console.print("[red]warning:[/red] preflight counts do not sum to queried cohort size.")


# =============================================================================
# Manifest and Metadata
# =============================================================================


def create_manifest(
    files: list[dict[str, Any]],
    output_path: Path,
    manifest_name: str = "gdc_manifest.txt",
) -> Path:
    """Create GDC manifest file for downloading."""
    manifest_path = output_path / manifest_name

    with open(manifest_path, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for file_rec in files:
            file_id = file_rec["file_id"]
            filename = file_rec["file_name"]
            size = file_rec.get("file_size", 0)
            f.write(f"{file_id}\t{filename}\t\t{size}\tlive\n")

    console.print(f"[green]✓[/green] Created manifest: {manifest_path}")
    return manifest_path


def create_metadata(files: list[dict[str, Any]], output_path: Path) -> Path:
    """
    Create metadata CSV with patient-slide mapping.

    Labels are derived from the barcode sample type code.
    """
    records: list[dict[str, Any]] = []

    for f in files:
        cases = f.get("cases", [{}])
        case_info = cases[0] if cases else {}

        submitter_id = case_info.get("submitter_id", "")
        case_id = case_info.get("case_id", "")

        diagnoses = case_info.get("diagnoses", [{}])
        diagnosis_info = diagnoses[0] if diagnoses else {}

        samples = case_info.get("samples", [{}])
        sample_info = samples[0] if samples else {}

        filename = f["file_name"]
        slide_id = filename.replace(".svs", "").replace(".SVS", "")

        bc = f.get("_barcode") or _tcga_barcode_tokens(filename)
        sample_code = bc.get("sample_code", "")
        portion_code = bc.get("portion_code", "")

        sample_meta = SAMPLE_CODE_META.get(sample_code, {"name": "Unknown", "label": -1})
        label = sample_meta["label"]
        sample_code_name = sample_meta["name"]

        records.append(
            {
                "file_id": f["file_id"],
                "slide_id": slide_id,
                "filename": filename,
                "patient_id": submitter_id,
                "case_id": case_id,
                "project_id": case_info.get("project", {}).get("project_id", ""),
                "sample_code": sample_code,
                "sample_code_name": sample_code_name,
                "portion_code": portion_code,
                "label": label,
                "sample_type_gdc": sample_info.get("sample_type", ""),
                "tissue_type_gdc": sample_info.get("tissue_type", ""),
                "primary_diagnosis": diagnosis_info.get("primary_diagnosis", ""),
                "tumor_stage": diagnosis_info.get("tumor_stage", ""),
                "experimental_strategy": f.get("experimental_strategy", ""),
                "data_format": f.get("data_format", ""),
                "file_size_bytes": f.get("file_size", 0),
                "data_type": f.get("data_type", ""),
            }
        )

    df = pd.DataFrame(records)
    df["file_size_gb"] = df["file_size_bytes"] / (1024**3)

    metadata_path = output_path / "metadata.csv"
    df.to_csv(metadata_path, index=False)

    console.print(f"\n[green]✓[/green] Created metadata: {metadata_path}")

    table = Table(title="TCGA Metadata Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total slides", str(len(df)))
    table.add_row("Unique patients", str(df["patient_id"].nunique()))
    table.add_row("Slides per patient (mean)", f"{len(df) / max(df['patient_id'].nunique(), 1):.2f}")
    table.add_row(
        "Slides per patient (max)",
        str(df.groupby("patient_id").size().max()) if len(df) > 0 else "0",
    )
    table.add_row("Primary tumor slides (01)", str((df["sample_code"] == "01").sum()))
    table.add_row("Metastatic slides (06)", str((df["sample_code"] == "06").sum()))
    table.add_row("Normal tissue slides (11)", str((df["sample_code"] == "11").sum()))
    table.add_row("Label=1 (tumor)", str((df["label"] == 1).sum()))
    table.add_row("Label=0 (normal)", str((df["label"] == 0).sum()))
    table.add_row("Label=-1 (unknown)", str((df["label"] == -1).sum()))
    table.add_row("Total size (GB)", f"{df['file_size_gb'].sum():.1f}")

    console.print(table)

    if len(df) > 0:
        slides_per_patient = df.groupby("patient_id").size()
        patients_with_multiple = (slides_per_patient > 1).sum()

        if patients_with_multiple > 0:
            console.print(
                f"\n[yellow]⚠  REPEATED MEASURES WARNING:[/yellow]\n"
                f"   {patients_with_multiple} patients have multiple slides.\n"
                f"   Always use --patient-id-column patient_id when calling create_splits.py!\n"
                f"   See: src/stain_benchmark/data/splitting.py"
            )

    unlabelled = (df["label"] == -1).sum()
    if unlabelled > 0:
        console.print(
            f"\n[red]⚠  {unlabelled} slides have label=-1 (unknown sample code).[/red]\n"
            f"   These rows MUST be excluded before splitting/training:\n"
            f"     df = df[df['label'].isin([0, 1])]"
        )

    return metadata_path


# =============================================================================
# Download and Organization
# =============================================================================


def download_with_gdc_client(
    manifest_path: Path,
    output_dir: Path,
    n_processes: int = 4,
    retry_amount: int = 3,
    token_path: Optional[Path] = None,
) -> bool:
    """Download files using gdc-client."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gdc-client",
        "download",
        "-m",
        str(manifest_path),
        "-d",
        str(output_dir),
        "-n",
        str(n_processes),
        "--retry-amount",
        str(retry_amount),
        "--log-file",
        str(output_dir / "gdc_download.log"),
    ]

    if token_path and token_path.exists():
        cmd.extend(["-t", str(token_path)])

    console.print(f"\n[bold cyan]Starting download with gdc-client[/bold cyan]")
    console.print(f"Command: {' '.join(cmd)}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Parallel processes: {n_processes}")
    console.print("\n[yellow]This may take several hours for large datasets...[/yellow]\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        process.wait()

        if process.returncode == 0:
            console.print("\n[bold green]✓ Download completed successfully![/bold green]")
            return True

        console.print(f"\n[bold red]✗ Download failed with code {process.returncode}[/bold red]")
        return False

    except subprocess.SubprocessError as e:
        console.print(f"\n[bold red]✗ Download error: {e}[/bold red]")
        return False


def organize_downloads(download_dir: Path, output_dir: Path) -> int:
    """Flatten GDC UUID directories into slides/ and return moved count."""
    console.print("[blue]Organizing downloaded files...[/blue]")

    slides_dir = output_dir / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped_existing = 0
    if download_dir.exists():
        for p in download_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() == ".svs":
                dest_path = slides_dir / p.name

                if dest_path.exists():
                    skipped_existing += 1
                    console.print(f"[yellow]Skipping (exists):[/yellow] {p.name}")
                    continue

                shutil.move(str(p), str(dest_path))
                moved += 1

        for uuid_dir in download_dir.iterdir():
            if uuid_dir.is_dir() and not list(uuid_dir.iterdir()):
                uuid_dir.rmdir()

    console.print(
        f"[green]✓[/green] Organized {moved} slide files to {slides_dir} "
        f"(skipped existing: {skipped_existing})"
    )
    return moved


def list_available_projects() -> None:
    """Print available TCGA projects."""
    table = Table(title="Available TCGA Projects for Stain Robustness Benchmark")
    table.add_column("Project ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Tissue", style="green")
    table.add_column("~Cases", style="yellow", justify="right")
    table.add_column("~Slides", style="yellow", justify="right")
    table.add_column("~Size (GB)", style="red", justify="right")

    for project_id, info in TCGA_PROJECTS.items():
        table.add_row(
            project_id,
            info["name"],
            info["tissue"],
            str(info["approx_cases"]),
            str(info["approx_slides"]),
            str(info["approx_size_gb"]),
        )

    console.print(table)
    console.print(
        "\n[dim]Note: Sizes are approximate. Actual sizes may vary.[/dim]"
        "\n[dim]For stain variation study, recommend: TCGA-BRCA, TCGA-COAD, TCGA-LUAD[/dim]"
    )


# =============================================================================
# CLI Interface
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TCGA slides using GDC Data Transfer Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available projects
  python scripts/download_tcga.py --list-projects

  # Download TCGA-BRCA (diagnostic-only default)
  python scripts/download_tcga.py --project TCGA-BRCA --output data/tcga

  # Download multiple projects
  python scripts/download_tcga.py --project TCGA-BRCA TCGA-COAD --output data/tcga

  # Download first 50 cases only (for testing)
  python scripts/download_tcga.py --project TCGA-BRCA --max-cases 50

  # Generate manifest only (no download)
  python scripts/download_tcga.py --project TCGA-BRCA --manifest-only

  # Generate broader inventory (01/06/11 sample codes, no API diagnostic restriction,
  # no DX-only barcode restriction)
  python scripts/download_tcga.py --project TCGA-BRCA --manifest-only --all-slides

  # Use auth token for controlled access data
  python scripts/download_tcga.py --project TCGA-BRCA --token ~/.gdc-token

IMPORTANT - Repeated Measures:
  TCGA has multiple slides per patient. When creating train/val/test splits,
  use patient_id_column='patient_id' to prevent data leakage!
        """,
    )

    parser.add_argument(
        "--project",
        "-p",
        type=str,
        nargs="+",
        help="TCGA project ID(s) to download (e.g., TCGA-BRCA TCGA-COAD)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/tcga",
        help="Output directory (default: data/tcga)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to download (for testing)",
    )
    parser.add_argument(
        "--n-processes",
        "-n",
        type=int,
        default=4,
        help="Number of parallel download processes (default: 4)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Generate manifest and metadata only, don't download",
    )
    parser.add_argument(
        "--token",
        "-t",
        type=str,
        default=None,
        help="Path to GDC auth token file (for controlled access data)",
    )
    parser.add_argument(
        "--no-organize",
        action="store_true",
        default=False,
        help="Do NOT flatten UUID dirs into slides/ (default: organize is ON)",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List available TCGA projects",
    )
    parser.add_argument(
        "--all-slides",
        action="store_true",
        default=False,
        help=(
            "Broader cohort mode: remove API Diagnostic Slide restriction and "
            "DX-only barcode restriction. Sample-code gate (01/06/11) still applies."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.list_projects:
        list_available_projects()
        return

    if not args.project:
        console.print("[red]Error:[/red] --project is required")
        console.print("Use --list-projects to see available projects")
        sys.exit(1)

    if not args.manifest_only:
        if not check_gdc_client():
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files: list[dict[str, Any]] = []

    for project_id in args.project:
        project_id = project_id.upper()

        if project_id not in TCGA_PROJECTS:
            console.print(f"[yellow]Warning: {project_id} not in known projects[/yellow]")

        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]Processing {project_id}[/bold cyan]")
        console.print("=" * 60)

        try:
            files = query_gdc_slides(
                project_id=project_id,
                max_cases=args.max_cases,
                dx_only=not args.all_slides,
                api_diagnostic_only=not args.all_slides,
            )
            all_files.extend(files)
        except requests.RequestException as e:
            console.print(f"[red]Error querying GDC API:[/red] {e}")
            continue

    if not all_files:
        console.print("[red]No files found to download![/red]")
        sys.exit(1)

    existing_filenames = collect_existing_slide_filenames(output_dir)
    existing_files, pending_files = split_files_by_existing(all_files, existing_filenames)

    full_manifest_path = create_manifest(all_files, output_dir, manifest_name="gdc_manifest_all.txt")
    pending_manifest_path = create_manifest(
        pending_files,
        output_dir,
        manifest_name="gdc_manifest_pending.txt",
    )
    compatibility_manifest_path = create_manifest(all_files, output_dir, manifest_name="gdc_manifest.txt")
    metadata_path = create_metadata(all_files, output_dir)

    print_download_plan(
        all_files=all_files,
        existing_files=existing_files,
        pending_files=pending_files,
        output_dir=output_dir,
    )

    if args.manifest_only:
        console.print("\n[green]✓ Manifests and metadata created. Skipping download.[/green]")
        console.print(f"  Full manifest: {full_manifest_path}")
        console.print(f"  Pending manifest: {pending_manifest_path}")
        console.print(f"  Compatibility manifest: {compatibility_manifest_path}")
        console.print(f"  Metadata: {metadata_path}")
        console.print(
            f"\nPending rows to download right now: {_manifest_row_count(pending_manifest_path)}"
        )
        console.print(f"\nTo download pending only, run:\n  gdc-client download -m {pending_manifest_path}")
        return

    token_path = Path(args.token) if args.token else None
    download_dir = output_dir / "raw"

    if not pending_files:
        console.print(
            "\n[bold yellow]Nothing new to download. All queried slides are already present locally.[/bold yellow]"
        )
        success = True
    else:
        success = download_with_gdc_client(
            manifest_path=pending_manifest_path,
            output_dir=download_dir,
            n_processes=args.n_processes,
            token_path=token_path,
        )

    if success and not args.no_organize:
        organize_downloads(download_dir, output_dir)

    console.print("\n" + "=" * 60)
    if success:
        console.print("[bold green]✓ TCGA download completed![/bold green]")
        console.print("\nNext steps:")
        console.print("  1. Create splits with patient-level grouping:")
        console.print("     python scripts/create_splits.py \\")
        console.print(f"       --metadata {metadata_path} \\")
        console.print("       --patient-id-column patient_id \\")
        console.print("       --output data/tcga/splits")
    else:
        console.print("[bold red]✗ Download incomplete. Check errors above.[/bold red]")
    console.print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()