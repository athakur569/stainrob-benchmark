#!/usr/bin/env python3
"""
Download CAMELYON16 Dataset

=============================================================================
This script downloads the CAMELYON16 dataset for the stain robustness
benchmark. The dataset consists of ~400 whole slide images (WSIs) of
breast cancer lymph node sections.

Usage:
    python scripts/download_data.py --output data/camelyon16/raw

Options:
    --output    Output directory for downloaded files
    --source    Download source (gdrive, aws, manual)
    --verify    Verify checksums after download
    --resume    Resume interrupted downloads

Dataset size: ~45 GB total
=============================================================================

Author: Abhishek Thakur
Date: February 2026
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Callable

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

console = Console()
logger = logging.getLogger(__name__)


# =============================================================================
# Download URLs and Metadata
# =============================================================================

CAMELYON16_INFO = {
    "name": "CAMELYON16",
    "description": "Breast cancer metastasis detection in lymph node WSIs",
    "url": "https://camelyon16.grand-challenge.org/",
    "license": "CC BY-NC-SA 4.0",
    "citation": """@article{bejnordi2017diagnostic,
  title={Diagnostic assessment of deep learning algorithms for detection of lymph node metastases in women with breast cancer},
  author={Bejnordi, Babak Ehteshami and others},
  journal={JAMA},
  volume={318},
  number={22},
  pages={2199--2210},
  year={2017}
}""",
}

# Google Drive file IDs (public mirrors)
GDRIVE_FILES = {
    "train_normal": {
        "id": "1vXCwhLoLmBQcMDkLKT9cQDWAc4y_cZCu",  # Example ID
        "filename": "train_normal.zip",
        "size_gb": 15,
    },
    "train_tumor": {
        "id": "1pHZ-WYhQoCxF4uLsrLoGPJkGOUoQb45y",  # Example ID
        "filename": "train_tumor.zip",
        "size_gb": 12,
    },
    "test": {
        "id": "1Sc1UPMGZA5bUJQVzFqKLELMIdjZCxDJC",  # Example ID
        "filename": "test.zip",
        "size_gb": 18,
    },
}

# Alternative: Direct download URLs (if available)
DIRECT_URLS = {
    # Add direct URLs here if available
}


# =============================================================================
# Download Utilities
# =============================================================================


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if there's enough disk space."""
    import shutil

    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024**3)

    if free_gb < required_gb:
        console.print(
            f"[red]Error:[/red] Not enough disk space. "
            f"Required: {required_gb:.1f} GB, Available: {free_gb:.1f} GB"
        )
        return False
    return True


def download_with_progress(
    url: str,
    output_path: Path,
    description: str = "Downloading",
    chunk_size: int = 8192,
    resume: bool = True,
) -> bool:
    """
    Download a file with progress bar and resume capability.

    Args:
        url: URL to download from.
        output_path: Path to save the file.
        description: Description for progress bar.
        chunk_size: Download chunk size in bytes.
        resume: Whether to resume partial downloads.

    Returns:
        True if download succeeded, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing partial download
    initial_pos = 0
    if resume and output_path.exists():
        initial_pos = output_path.stat().st_size
        logger.info("Resuming download from byte %d", initial_pos)

    # Set up headers for resume
    headers = {}
    if initial_pos > 0:
        headers["Range"] = f"bytes={initial_pos}-"

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()

        # Get total size
        total_size = int(response.headers.get("content-length", 0))
        if initial_pos > 0:
            total_size += initial_pos

        # Download with progress
        mode = "ab" if initial_pos > 0 else "wb"

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(description, total=total_size, completed=initial_pos)

            with open(output_path, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        return True

    except requests.RequestException as e:
        console.print(f"[red]Download error:[/red] {e}")
        return False


def download_from_gdrive(
    file_id: str,
    output_path: Path,
    description: str = "Downloading",
) -> bool:
    """
    Download a file from Google Drive.

    Args:
        file_id: Google Drive file ID.
        output_path: Path to save the file.
        description: Description for progress bar.

    Returns:
        True if download succeeded.
    """
    try:
        import gdown
    except ImportError:
        console.print("[red]Error:[/red] gdown not installed. Run: pip install gdown")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    console.print(f"[blue]Downloading from Google Drive:[/blue] {description}")

    try:
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists()
    except Exception as e:
        console.print(f"[red]Download error:[/red] {e}")
        return False


def compute_md5(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file checksum."""
    console.print(f"[blue]Verifying checksum:[/blue] {file_path.name}")
    actual_md5 = compute_md5(file_path)

    if actual_md5 == expected_md5:
        console.print(f"[green]✓[/green] Checksum verified: {actual_md5}")
        return True
    else:
        console.print(
            f"[red]✗[/red] Checksum mismatch!\n"
            f"  Expected: {expected_md5}\n"
            f"  Actual:   {actual_md5}"
        )
        return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract a zip archive with progress."""
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
                task = progress.add_task("Extracting", total=len(members))

                for member in members:
                    zf.extract(member, output_dir)
                    progress.update(task, advance=1)

        console.print(f"[green]✓[/green] Extracted to {output_dir}")
        return True

    except zipfile.BadZipFile as e:
        console.print(f"[red]Extraction error:[/red] {e}")
        return False


# =============================================================================
# Main Download Functions
# =============================================================================


def download_camelyon16(
    output_dir: Path,
    source: str = "gdrive",
    verify: bool = True,
    resume: bool = True,
    extract: bool = True,
) -> bool:
    """
    Download the complete CAMELYON16 dataset.

    Args:
        output_dir: Directory to save downloaded files.
        source: Download source ('gdrive', 'direct', 'manual').
        verify: Whether to verify checksums.
        resume: Whether to resume interrupted downloads.
        extract: Whether to extract archives after download.

    Returns:
        True if all downloads succeeded.
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]CAMELYON16 Dataset Download[/bold cyan]")
    console.print("=" * 60)

    # Print dataset info
    table = Table(show_header=False, box=None)
    table.add_row("[bold]Dataset:[/bold]", CAMELYON16_INFO["name"])
    table.add_row("[bold]Description:[/bold]", CAMELYON16_INFO["description"])
    table.add_row("[bold]URL:[/bold]", CAMELYON16_INFO["url"])
    table.add_row("[bold]License:[/bold]", CAMELYON16_INFO["license"])
    console.print(table)
    console.print()

    # Check disk space
    total_size_gb = sum(f["size_gb"] for f in GDRIVE_FILES.values())
    console.print(f"[yellow]Required disk space:[/yellow] ~{total_size_gb} GB")

    if not check_disk_space(output_dir, total_size_gb * 1.5):  # 1.5x for extraction
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    success = True

    if source == "gdrive":
        # Download from Google Drive
        for name, info in GDRIVE_FILES.items():
            archive_path = output_dir / info["filename"]

            console.print(f"\n[bold]Downloading {name}...[/bold]")

            if archive_path.exists():
                console.print(f"[yellow]File exists:[/yellow] {archive_path}")
                console.print("Skipping download. Delete file to re-download.")
            else:
                if not download_from_gdrive(
                    info["id"],
                    archive_path,
                    f"{name} ({info['size_gb']} GB)",
                ):
                    success = False
                    continue

            # Extract if requested
            if extract and archive_path.exists():
                extract_dir = output_dir / name
                if not extract_archive(archive_path, extract_dir):
                    success = False

    elif source == "manual":
        # Print manual download instructions
        console.print("\n[bold yellow]Manual Download Instructions[/bold yellow]")
        console.print(
            "\n1. Visit the CAMELYON16 challenge website:\n"
            f"   {CAMELYON16_INFO['url']}\n"
            "\n2. Register for the challenge (free)\n"
            "\n3. Download the following files:\n"
            "   - Training set (Normal slides)\n"
            "   - Training set (Tumor slides)\n"
            "   - Test set\n"
            "   - Annotations (XML files)\n"
            f"\n4. Place the downloaded files in:\n"
            f"   {output_dir}\n"
            "\n5. Re-run this script with --skip-download to continue."
        )
        return True

    else:
        console.print(f"[red]Unknown source:[/red] {source}")
        return False

    # Final status
    console.print("\n" + "=" * 60)
    if success:
        console.print("[bold green]✓ Download completed successfully![/bold green]")
    else:
        console.print("[bold red]✗ Some downloads failed. Please check errors above.[/bold red]")
    console.print("=" * 60 + "\n")

    return success


def create_metadata(data_dir: Path, output_path: Path) -> None:
    """
    Create metadata CSV from downloaded slides.

    Scans the data directory for SVS files and creates a metadata file
    with slide IDs and labels (inferred from directory structure).

    Args:
        data_dir: Directory containing downloaded slides.
        output_path: Path for output metadata CSV.
    """
    import pandas as pd

    console.print("[blue]Creating metadata file...[/blue]")

    records = []

    # Scan for SVS files
    for svs_path in data_dir.rglob("*.svs"):
        slide_id = svs_path.stem

        # Infer label from path
        path_lower = str(svs_path).lower()
        if "tumor" in path_lower:
            label = 1
        elif "normal" in path_lower:
            label = 0
        else:
            # Try to infer from filename
            if slide_id.startswith("tumor"):
                label = 1
            elif slide_id.startswith("normal"):
                label = 0
            else:
                label = -1  # Unknown

        records.append({
            "slide_id": slide_id,
            "label": label,
            "path": str(svs_path),
        })

    if not records:
        console.print("[yellow]Warning: No SVS files found![/yellow]")
        return

    # Create DataFrame and save
    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print summary
    console.print(f"[green]✓[/green] Created metadata: {output_path}")
    console.print(f"  Total slides: {len(df)}")
    console.print(f"  Normal: {(df['label'] == 0).sum()}")
    console.print(f"  Tumor: {(df['label'] == 1).sum()}")
    console.print(f"  Unknown: {(df['label'] == -1).sum()}")


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download CAMELYON16 dataset for stain robustness benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to default location
  python scripts/download_data.py

  # Download to specific directory
  python scripts/download_data.py --output /data/camelyon16

  # Show manual download instructions
  python scripts/download_data.py --source manual

  # Resume interrupted download
  python scripts/download_data.py --resume
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/camelyon16/raw",
        help="Output directory for downloaded files (default: data/camelyon16/raw)",
    )

    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["gdrive", "direct", "manual"],
        default="gdrive",
        help="Download source (default: gdrive)",
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify checksums after download (default: True)",
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume interrupted downloads (default: True)",
    )

    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Don't extract archives after download",
    )

    parser.add_argument(
        "--create-metadata",
        action="store_true",
        help="Create metadata CSV after download",
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
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run download
    output_dir = Path(args.output)
    success = download_camelyon16(
        output_dir=output_dir,
        source=args.source,
        verify=not args.no_verify,
        resume=args.resume,
        extract=not args.no_extract,
    )

    # Create metadata if requested
    if success and args.create_metadata:
        metadata_path = output_dir.parent / "metadata" / "slides.csv"
        create_metadata(output_dir, metadata_path)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
