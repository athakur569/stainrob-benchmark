#!/usr/bin/env python3
"""
Download Additional Datasets for Stain Robustness Benchmark

=============================================================================
This script provides download utilities for additional histopathology datasets
useful for stain robustness and domain adaptation research.

Supported Datasets:
    - PANDA: Prostate cancer grading (Radboud + Karolinska - multi-center)
    - MIDOG: Mitosis detection (multi-scanner)
    - NCT-CRC-HE: Colorectal tissue classification
    - BRACS: Breast carcinoma subtyping

Usage:
    # List available datasets
    python scripts/download_additional_datasets.py --list

    # Download NCT-CRC-HE (direct download)
    python scripts/download_additional_datasets.py --dataset nct-crc --output data/

    # Download MIDOG (requires Kaggle or manual)
    python scripts/download_additional_datasets.py --dataset midog --output data/
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASETS = {
    "nct-crc": {
        "name": "NCT-CRC-HE-100K",
        "description": "100K colorectal cancer patches (9 tissue classes)",
        "task": "Tissue classification",
        "size_gb": 7.0,
        "citation": "Kather et al., 2018",
        "license": "CC BY 4.0",
        "download_type": "zenodo",
        "zenodo_record": "1214456",
        "urls": {
            "normalized": "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip",
            "unnormalized": "https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K-NONORM.zip",
        },
        "has_repeated_measures": False,
        "notes": "Available in Macenko-normalized and unnormalized versions.",
    },
    "crc-val": {
        "name": "CRC-VAL-HE-7K",
        "description": "7K colorectal validation patches (external test set)",
        "task": "Tissue classification",
        "size_gb": 0.5,
        "citation": "Kather et al., 2018",
        "license": "CC BY 4.0",
        "download_type": "zenodo",
        "zenodo_record": "1214456",
        "urls": {
            "main": "https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip",
        },
        "has_repeated_measures": False,
        "notes": "External validation set from different patients.",
    },
    "panda": {
        "name": "PANDA Challenge",
        "description": "Prostate cancer grading (10,616 WSIs, 2 centers)",
        "task": "Gleason grading",
        "size_gb": 400,
        "citation": "Bulten et al., 2022 (Nature Medicine)",
        "license": "Challenge license",
        "download_type": "kaggle",
        "kaggle_competition": "prostate-cancer-grade-assessment",
        "has_repeated_measures": True,
        "notes": "Multi-center dataset with built-in domain shift. Requires Kaggle account.",
    },
    "midog": {
        "name": "MIDOG 2021",
        "description": "Mitosis detection across scanners (200 cases, 4 scanners)",
        "task": "Mitosis detection",
        "size_gb": 15,
        "citation": "Aubreville et al., 2023",
        "license": "CC BY-NC 4.0",
        "download_type": "manual",
        "url": "https://midog2021.grand-challenge.org/",
        "has_repeated_measures": False,
        "notes": "First challenge for domain generalization. Register at grand-challenge.org.",
    },
    "midog-plus": {
        "name": "MIDOG++",
        "description": "Extended mitosis dataset (503 cases, 7 tumor types)",
        "task": "Mitosis detection",
        "size_gb": 25,
        "citation": "Aubreville et al., 2023 (Scientific Data)",
        "license": "CC BY 4.0",
        "download_type": "figshare",
        "figshare_collection": "6615571",
        "url": "https://springernature.figshare.com/collections/MIDOG_A_Comprehensive_Multi-Domain_Dataset_for_Mitotic_Figure_Detection/6615571",
        "has_repeated_measures": False,
        "notes": "Extended version with multiple tumor types and scanners.",
    },
    "bracs": {
        "name": "BRACS",
        "description": "Breast carcinoma subtyping (547 WSIs, 4539 ROIs)",
        "task": "7-class classification",
        "size_gb": 50,
        "citation": "Brancati et al., 2022 (Database)",
        "license": "CC BY-NC-SA 4.0",
        "download_type": "manual",
        "url": "https://www.bracs.icar.cnr.it/",
        "has_repeated_measures": True,
        "notes": "Register at BRACS website for download access.",
    },
    "bach": {
        "name": "BACH Challenge",
        "description": "Breast cancer classification (400 images + 30 WSIs)",
        "task": "4-class classification",
        "size_gb": 8,
        "citation": "Aresta et al., 2019",
        "license": "Challenge license",
        "download_type": "manual",
        "url": "https://iciar2018-challenge.grand-challenge.org/",
        "has_repeated_measures": False,
        "notes": "ICIAR 2018 Grand Challenge dataset.",
    },
    "mitos-atypia": {
        "name": "MITOS-ATYPIA-14",
        "description": "Mitosis detection with 2 scanners (16 WSIs)",
        "task": "Mitosis detection",
        "size_gb": 5,
        "citation": "MITOS-ATYPIA 2014",
        "license": "Research only",
        "download_type": "manual",
        "url": "https://mitos-atypia-14.grand-challenge.org/",
        "has_repeated_measures": False,
        "notes": "Classic benchmark for multi-scanner evaluation (Aperio + Hamamatsu).",
    },
}


# =============================================================================
# Download Functions
# =============================================================================


def download_with_progress(
    url: str,
    output_path: Path,
    description: str = "Downloading",
    chunk_size: int = 8192,
) -> bool:
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(description, total=total_size)

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        return True

    except requests.RequestException as e:
        console.print(f"[red]Download error:[/red] {e}")
        return False


def extract_zip(archive_path: Path, output_dir: Path) -> bool:
    """Extract zip archive with progress."""
    console.print(f"[blue]Extracting:[/blue] {archive_path.name}")

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()

            with Progress(
                TextColumn("[bold blue]Extracting"),
                BarColumn(bar_width=40),
                TextColumn("{task.completed}/{task.total} files"),
                console=console,
            ) as progress:
                task = progress.add_task("", total=len(members))

                for member in members:
                    zf.extract(member, output_dir)
                    progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Extracted to {output_dir}")
        return True

    except zipfile.BadZipFile as e:
        console.print(f"[red]Extraction error:[/red] {e}")
        return False


def download_from_zenodo(
    record_id: str,
    urls: dict[str, str],
    output_dir: Path,
    version: str = "normalized",
    extract: bool = True,
) -> bool:
    """Download dataset from Zenodo."""
    if version not in urls:
        console.print(f"[red]Version '{version}' not available.[/red]")
        console.print(f"Available versions: {list(urls.keys())}")
        return False

    url = urls[version]
    filename = url.split("/")[-1]
    archive_path = output_dir / filename

    console.print(f"\n[bold cyan]Downloading from Zenodo[/bold cyan]")
    console.print(f"Record: {record_id}")
    console.print(f"Version: {version}")
    console.print(f"URL: {url}")

    if archive_path.exists():
        console.print(f"[yellow]File exists:[/yellow] {archive_path}")
        console.print("Delete file to re-download.")
    else:
        if not download_with_progress(url, archive_path, f"Downloading {filename}"):
            return False

    if extract and archive_path.exists():
        extract_dir = output_dir / archive_path.stem
        return extract_zip(archive_path, extract_dir)

    return True


def download_from_kaggle(
    competition: str,
    output_dir: Path,
) -> bool:
    """Download dataset from Kaggle competition."""
    console.print(f"\n[bold cyan]Downloading from Kaggle[/bold cyan]")
    console.print(f"Competition: {competition}")

    # Check for kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        console.print(
            "[red]Error:[/red] Kaggle CLI not found.\n"
            "\nInstall with:\n"
            "  pip install kaggle\n"
            "\nThen configure:\n"
            "  1. Create API token at https://www.kaggle.com/settings\n"
            "  2. Place kaggle.json in ~/.kaggle/\n"
            "  3. chmod 600 ~/.kaggle/kaggle.json"
        )
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c", competition,
        "-p", str(output_dir),
    ]

    console.print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        console.print(f"[green]✓[/green] Downloaded to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Kaggle download failed:[/red] {e}")
        return False


def print_manual_instructions(dataset_id: str, info: dict) -> None:
    """Print manual download instructions."""
    console.print(f"\n[bold yellow]Manual Download Required: {info['name']}[/bold yellow]")
    console.print(f"\nDescription: {info['description']}")
    console.print(f"Size: ~{info['size_gb']} GB")
    console.print(f"License: {info['license']}")
    console.print(f"\nDownload URL: {info['url']}")
    console.print(f"\nNotes: {info['notes']}")

    if info.get("has_repeated_measures"):
        console.print(
            "\n[yellow]⚠️  REPEATED MEASURES:[/yellow] "
            "This dataset may have multiple samples per patient.\n"
            "   Use patient_id_column when creating splits!"
        )


def list_datasets() -> None:
    """Print available datasets."""
    table = Table(title="Additional Datasets for Stain Robustness Benchmark")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Task", style="green")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Download", style="blue")
    table.add_column("Multi-Sample", style="red")

    for dataset_id, info in DATASETS.items():
        multi = "⚠️ Yes" if info.get("has_repeated_measures") else "No"
        table.add_row(
            dataset_id,
            info["name"],
            info["task"],
            f"{info['size_gb']} GB",
            info["download_type"],
            multi,
        )

    console.print(table)
    console.print(
        "\n[dim]Download types: zenodo (direct), kaggle (CLI), manual (registration)[/dim]"
        "\n[dim]Multi-Sample: Datasets with multiple slides per patient (use patient_id_column!)[/dim]"
    )


# =============================================================================
# Main Function
# =============================================================================


def download_dataset(
    dataset_id: str,
    output_dir: Path,
    version: str = "normalized",
    extract: bool = True,
) -> bool:
    """
    Download a dataset by ID.

    Args:
        dataset_id: Dataset identifier (e.g., 'nct-crc').
        output_dir: Output directory.
        version: Version to download (for datasets with multiple versions).
        extract: Whether to extract archives.

    Returns:
        True if download succeeded.
    """
    if dataset_id not in DATASETS:
        console.print(f"[red]Unknown dataset:[/red] {dataset_id}")
        console.print("Use --list to see available datasets.")
        return False

    info = DATASETS[dataset_id]
    dataset_dir = output_dir / dataset_id

    console.print("\n" + "=" * 60)
    console.print(f"[bold cyan]{info['name']}[/bold cyan]")
    console.print("=" * 60)
    console.print(f"Description: {info['description']}")
    console.print(f"Task: {info['task']}")
    console.print(f"Size: ~{info['size_gb']} GB")
    console.print(f"Citation: {info['citation']}")

    download_type = info["download_type"]

    if download_type == "zenodo":
        return download_from_zenodo(
            record_id=info["zenodo_record"],
            urls=info["urls"],
            output_dir=dataset_dir,
            version=version,
            extract=extract,
        )

    elif download_type == "kaggle":
        return download_from_kaggle(
            competition=info["kaggle_competition"],
            output_dir=dataset_dir,
        )

    elif download_type in ("manual", "figshare"):
        print_manual_instructions(dataset_id, info)
        return True

    else:
        console.print(f"[red]Unknown download type:[/red] {download_type}")
        return False


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download additional histopathology datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python scripts/download_additional_datasets.py --list

  # Download NCT-CRC-HE (normalized version)
  python scripts/download_additional_datasets.py --dataset nct-crc --output data/

  # Download unnormalized version
  python scripts/download_additional_datasets.py --dataset nct-crc --version unnormalized

  # Download PANDA from Kaggle
  python scripts/download_additional_datasets.py --dataset panda --output data/

  # Download multiple datasets
  python scripts/download_additional_datasets.py --dataset nct-crc crc-val --output data/

Recommended for Stain Robustness Benchmark:
  1. nct-crc      - Direct comparison of normalized vs unnormalized
  2. panda        - Multi-center domain shift (Kaggle)
  3. midog-plus   - Multi-scanner evaluation (Figshare)
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        nargs="+",
        help="Dataset ID(s) to download",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data",
        help="Output directory (default: data/)",
    )

    parser.add_argument(
        "--version",
        "-V",
        type=str,
        default="normalized",
        help="Version to download for datasets with multiple versions",
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract archives after download",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available datasets",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # List datasets
    if args.list:
        list_datasets()
        return

    # Check required arguments
    if not args.dataset:
        console.print("[red]Error:[/red] --dataset is required")
        console.print("Use --list to see available datasets")
        sys.exit(1)

    output_dir = Path(args.output)
    success = True

    for dataset_id in args.dataset:
        if not download_dataset(
            dataset_id=dataset_id.lower(),
            output_dir=output_dir,
            version=args.version,
            extract=not args.no_extract,
        ):
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
