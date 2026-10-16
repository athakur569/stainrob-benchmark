#!/usr/bin/env python3
"""
Download / register CAMELYON17 dataset and create metadata.

Design goals:
- same overall philosophy as the existing CAMELYON16 script
- manual-download friendly
- WSI-first (no pre-tiling)
- metadata ready for later local splits

Expected workflow:
1. Register/download CAMELYON17 manually from the official site
2. Place archives or extracted files under the chosen output directory
3. Run this script with --create-metadata
4. Later run your split script on the metadata

Example:
    python scripts/download_camelyon17.py --output /media/abhi/camelyon17/raw --source manual
    python scripts/download_camelyon17.py --output /media/abhi/camelyon17/raw --create-metadata
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

console = Console()
logger = logging.getLogger(__name__)

CAMELYON17_INFO = {
    "name": "CAMELYON17",
    "description": "Breast cancer lymph node metastasis detection in WSIs (multi-center)",
    "url": "https://camelyon17.grand-challenge.org/Data/",
    "license": "Challenge terms / official site",
    "notes": (
        "Official CAMELYON17 data include training and test WSIs, organized by patient. "
        "Use the official download source and keep raw data untouched."
    ),
}



def check_disk_space(path: Path, required_gb: float) -> bool:
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


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
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


def create_metadata(root: Path, output_path: Path) -> None:
    """Write the common-schema slide table (see stain_benchmark.data.metadata)."""
    from stain_benchmark.data.metadata import build_metadata

    console.print("[blue]Creating CAMELYON17 metadata from stages.csv and annotations...[/blue]")
    df = build_metadata("camelyon17", root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    console.print(f"[green]✓[/green] Created metadata: {output_path}")

    table = Table(title="CAMELYON17 Metadata Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total slides", str(len(df)))
    table.add_row("Patients", str(df["patient_id"].nunique()))
    table.add_row("Centers", str(df["center"].nunique()))
    for role, n in df["role"].value_counts().items():
        table.add_row(f"role={role}", str(n))
    console.print(table)


def print_manual_instructions(output_dir: Path) -> None:
    console.print("\n[bold yellow]Manual Download Instructions: CAMELYON17[/bold yellow]")
    console.print(
        f"\n1. Visit the official data page:\n   {CAMELYON17_INFO['url']}"
        "\n2. Register / accept terms if required"
        "\n3. Download the CAMELYON17 training and test data archives"
        f"\n4. Place the archives or extracted folders under:\n   {output_dir}"
        "\n5. Re-run with --extract-archives if needed"
        "\n6. Re-run with --create-metadata to build the metadata CSV"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare CAMELYON17 dataset for stain robustness benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_camelyon17.py --output /media/abhi/camelyon17/raw --source manual
  python scripts/download_camelyon17.py --output /media/abhi/camelyon17/raw --extract-archives
  python scripts/download_camelyon17.py --output /media/abhi/camelyon17/raw --create-metadata
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/camelyon17/raw",
        help="Output/raw directory (default: data/camelyon17/raw)",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["manual"],
        default="manual",
        help="Download source (currently manual only)",
    )
    parser.add_argument(
        "--extract-archives",
        action="store_true",
        help="Extract any .zip archives found in the output directory",
    )
    parser.add_argument(
        "--create-metadata",
        action="store_true",
        help="Create metadata CSV from discovered WSI files",
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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n" + "=" * 68)
    console.print("[bold cyan]CAMELYON17 Dataset Preparation[/bold cyan]")
    console.print("=" * 68)

    info = Table(show_header=False, box=None)
    info.add_row("[bold]Dataset:[/bold]", CAMELYON17_INFO["name"])
    info.add_row("[bold]Description:[/bold]", CAMELYON17_INFO["description"])
    info.add_row("[bold]URL:[/bold]", CAMELYON17_INFO["url"])
    info.add_row("[bold]Notes:[/bold]", CAMELYON17_INFO["notes"])
    console.print(info)

    # Rough safety check only
    check_disk_space(output_dir, required_gb=700.0)

    if args.source == "manual":
        print_manual_instructions(output_dir)

    if args.extract_archives:
        archives = sorted(output_dir.glob("*.zip"))
        if not archives:
            console.print("[yellow]No .zip archives found to extract.[/yellow]")
        else:
            for archive_path in archives:
                extract_dir = output_dir / archive_path.stem
                extract_archive(archive_path, extract_dir)

    if args.create_metadata:
        metadata_path = output_dir.parent / "metadata.csv"
        create_metadata(output_dir.parent, metadata_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())