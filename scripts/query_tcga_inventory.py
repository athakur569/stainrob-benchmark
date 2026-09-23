#!/usr/bin/env python3
"""
Quick GDC slide inventory query tool.

Purpose:
- Query TCGA (or other GDC projects) and show how many slide/image files exist
- Summarize by:
    - experimental_strategy
    - sample_type
    - tissue_type
    - data_format
- Useful for checking whether BRCA has diagnostic vs tissue slides, tumor vs normal, etc.

Examples:
    python scripts/query_tcga_inventory.py --project TCGA-BRCA
    python scripts/query_tcga_inventory.py --project TCGA-BRCA TCGA-LUAD
    python scripts/query_tcga_inventory.py --project TCGA-BRCA --data-type "Slide Image"
    python scripts/query_tcga_inventory.py --project TCGA-BRCA --format SVS
    python scripts/query_tcga_inventory.py --project TCGA-BRCA --save-csv /tmp/brca_inventory.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

GDC_API_BASE = "https://api.gdc.cancer.gov"
GDC_FILES_ENDPOINT = f"{GDC_API_BASE}/files"


def build_filters(
    project_id: str,
    data_type: str | None = "Slide Image",
    data_format: str | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "op": "=",
            "content": {
                "field": "cases.project.project_id",
                "value": project_id,
            },
        }
    ]

    if data_type:
        content.append(
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": data_type,
                },
            }
        )

    if data_format:
        content.append(
            {
                "op": "=",
                "content": {
                    "field": "data_format",
                    "value": data_format,
                },
            }
        )

    return {"op": "and", "content": content}


def query_files(
    project_id: str,
    data_type: str | None = "Slide Image",
    data_format: str | None = None,
) -> list[dict[str, Any]]:
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
        "cases.samples.sample_type",
        "cases.samples.tissue_type",
        "cases.diagnoses.primary_diagnosis",
    ]

    filters = build_filters(project_id, data_type=data_type, data_format=data_format)

    all_hits: list[dict[str, Any]] = []
    page_size = 500
    offset = 0

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": page_size,
            "from": offset,
        }

        r = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()

        hits = payload.get("data", {}).get("hits", [])
        if not hits:
            break

        all_hits.extend(hits)
        offset += len(hits)

        total = payload.get("data", {}).get("pagination", {}).get("total", 0)
        if offset >= total:
            break

    return all_hits


def flatten_hits(hits: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for h in hits:
        cases = h.get("cases", [])
        case = cases[0] if cases else {}

        samples = case.get("samples", [])
        sample = samples[0] if samples else {}

        diagnoses = case.get("diagnoses", [])
        diagnosis = diagnoses[0] if diagnoses else {}

        rows.append(
            {
                "file_id": h.get("file_id", ""),
                "file_name": h.get("file_name", ""),
                "file_size_bytes": h.get("file_size", 0),
                "data_type": h.get("data_type", ""),
                "data_format": h.get("data_format", ""),
                "experimental_strategy": h.get("experimental_strategy", ""),
                "project_id": case.get("project", {}).get("project_id", ""),
                "case_id": case.get("case_id", ""),
                "patient_id": case.get("submitter_id", ""),
                "sample_type": sample.get("sample_type", ""),
                "tissue_type": sample.get("tissue_type", ""),
                "primary_diagnosis": diagnosis.get("primary_diagnosis", ""),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["file_size_gb"] = df["file_size_bytes"] / (1024**3)
    return df


def print_summary(df: pd.DataFrame, project_id: str) -> None:
    print("\n" + "=" * 80)
    print(f"PROJECT: {project_id}")
    print("=" * 80)

    if df.empty:
        print("No matching files found.")
        return

    n_files = len(df)
    n_patients = df["patient_id"].nunique() if "patient_id" in df.columns else 0
    n_cases = df["case_id"].nunique() if "case_id" in df.columns else 0
    total_gb = df["file_size_gb"].sum() if "file_size_gb" in df.columns else 0.0

    print(f"Total files:     {n_files}")
    print(f"Unique patients: {n_patients}")
    print(f"Unique cases:    {n_cases}")
    print(f"Total size (GB): {total_gb:.2f}")

    if n_patients:
        slides_per_patient = df.groupby("patient_id").size()
        print(f"Slides/patient:  min={slides_per_patient.min()}  "
              f"max={slides_per_patient.max()}  mean={slides_per_patient.mean():.2f}")

    def show_counts(col: str, top_n: int = 50) -> None:
        print(f"\n--- {col} ---")
        vc = df[col].fillna("").replace("", "<missing>").value_counts(dropna=False).head(top_n)
        for k, v in vc.items():
            print(f"{v:>6}  {k}")

    for col in [
        "experimental_strategy",
        "sample_type",
        "tissue_type",
        "data_format",
        "primary_diagnosis",
    ]:
        if col in df.columns:
            show_counts(col)

    print("\n--- sample_type x experimental_strategy ---")
    ctab = pd.crosstab(
        df["sample_type"].fillna("<missing>").replace("", "<missing>"),
        df["experimental_strategy"].fillna("<missing>").replace("", "<missing>"),
        dropna=False,
    )
    if ctab.empty:
        print("No data.")
    else:
        print(ctab.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Query GDC slide inventory by project.")
    parser.add_argument(
        "--project",
        "-p",
        nargs="+",
        required=True,
        help="One or more GDC project IDs, e.g. TCGA-BRCA TCGA-LUAD",
    )
    parser.add_argument(
        "--data-type",
        default="Slide Image",
        help='GDC data_type filter (default: "Slide Image"). Use "" to disable.',
    )
    parser.add_argument(
        "--format",
        default=None,
        help='Optional data_format filter, e.g. SVS',
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional output CSV path. If multiple projects are queried, project ID is appended.",
    )

    args = parser.parse_args()

    data_type = args.data_type if args.data_type != "" else None

    for project_id in args.project:
        hits = query_files(project_id=project_id, data_type=data_type, data_format=args.format)
        df = flatten_hits(hits)
        print_summary(df, project_id)

        if args.save_csv:
            out = Path(args.save_csv)
            if len(args.project) > 1:
                out = out.with_name(f"{out.stem}_{project_id}{out.suffix or '.csv'}")
            elif out.suffix == "":
                out = out.with_suffix(".csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"\nSaved CSV: {out}")


if __name__ == "__main__":
    main()