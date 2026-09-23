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

VALID_WSI_SUFFIXES = {".tif", ".tiff", ".svs"}


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


def infer_split_from_path(path_str: str) -> str:
    p = path_str.lower()
    if "training" in p or "/train" in p:
        return "train_source"
    if "test" in p:
        return "test_source"
    if "validation" in p or "val" in p:
        return "val_source"
    return "unknown"


def infer_label_from_path(path_str: str) -> int:
    p = path_str.lower()
    # Conservative: only infer when obvious
    if "normal" in p:
        return 0
    if "tumor" in p or "lesion" in p or "metast" in p:
        return 1
    return -1


def parse_camelyon17_ids(stem: str) -> tuple[str, str]:
    """
    Try to derive patient_id and node_id from common CAMELYON17-style names.

    Examples this handles safely:
    - patient_025_node_4
    - center_0_patient_025_node_4
    - patient025_node4

    Returns:
        patient_id, node_id
    """
    tokens = stem.replace("-", "_").split("_")
    patient_id = ""
    node_id = ""

    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low == "patient" and i + 1 < len(tokens):
            patient_id = f"patient_{tokens[i + 1]}"
        elif low.startswith("patient") and len(low) > len("patient"):
            suffix = tok[len("patient"):]
            if suffix:
                patient_id = f"patient_{suffix}"

        if low == "node" and i + 1 < len(tokens):
            node_id = f"node_{tokens[i + 1]}"
        elif low.startswith("node") and len(low) > len("node"):
            suffix = tok[len("node"):]
            if suffix:
                node_id = f"node_{suffix}"

    if not patient_id:
        # Fallback: group by full slide stem to avoid accidental leakage
        patient_id = stem

    return patient_id, node_id


def parse_center_from_path_or_name(path_str: str, stem: str) -> str:
    path_lower = path_str.lower()
    stem_lower = stem.lower()

    for text in (path_lower, stem_lower):
        if "center_" in text:
            idx = text.find("center_")
            frag = text[idx:].split("/")[0].split("_")
            if len(frag) >= 2:
                return f"center_{frag[1]}"
    return ""


def create_metadata(data_dir: Path, output_path: Path) -> None:
    console.print("[blue]Creating CAMELYON17 metadata...[/blue]")

    records: list[dict[str, object]] = []

    for wsi_path in data_dir.rglob("*"):
        if not wsi_path.is_file():
            continue
        if wsi_path.suffix.lower() not in VALID_WSI_SUFFIXES:
            continue

        slide_id = wsi_path.stem
        patient_id, node_id = parse_camelyon17_ids(slide_id)
        path_str = str(wsi_path)
        split_source = infer_split_from_path(path_str)
        label = infer_label_from_path(path_str)
        center_id = parse_center_from_path_or_name(path_str, slide_id)

        records.append(
            {
                "dataset_id": "camelyon17",
                "project_id": "CAMELYON17",
                "slide_id": slide_id,
                "patient_id": patient_id,
                "node_id": node_id,
                "center_id": center_id,
                "label": label,
                "source_split": split_source,
                "path": path_str,
                "file_ext": wsi_path.suffix.lower(),
            }
        )

    if not records:
        console.print("[yellow]Warning: No WSI files found (.tif/.tiff/.svs).[/yellow]")
        return

    df = pd.DataFrame(records).sort_values(["patient_id", "slide_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    console.print(f"[green]✓[/green] Created metadata: {output_path}")

    table = Table(title="CAMELYON17 Metadata Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total slides", str(len(df)))
    table.add_row("Unique patients/groups", str(df["patient_id"].nunique()))
    table.add_row("Label=1", str((df["label"] == 1).sum()))
    table.add_row("Label=0", str((df["label"] == 0).sum()))
    table.add_row("Label=-1", str((df["label"] == -1).sum()))
    table.add_row("Source train", str((df["source_split"] == "train_source").sum()))
    table.add_row("Source test", str((df["source_split"] == "test_source").sum()))
    console.print(table)

    if (df["label"] == -1).any():
        console.print(
            "\n[yellow]Note:[/yellow] Some rows have label=-1 because labels were not "
            "safely inferable from filenames/paths alone. That is okay at ingest time; "
            "you can enrich labels later from official CSVs if needed."
        )


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
        metadata_path = output_dir.parent / "metadata" / "slides.csv"
        create_metadata(output_dir, metadata_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())