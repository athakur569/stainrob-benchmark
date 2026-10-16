"""Slide metadata adapters: one common schema for every dataset.

Every downstream step (tiling, splits, freeze, training) reads only these columns:

    dataset, project_id, slide_id, patient_id, center, label, role, file

    center  institution (CAMELYON17 center; TCGA tissue source site from the barcode)
    label   slide-level label: 1 tumour, 0 normal, -1 unusable
    role    'annotated'   -> tile labels come from outline polygons
            'slide_label' -> every tissue tile gets the slide's label
            'excluded'    -> not used
    file    image path relative to the dataset root

Dataset-specific extras are kept as additional columns (e.g. stage, slide_prep).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCHEMA = ["dataset", "project_id", "slide_id", "patient_id", "center", "label", "role", "file"]


def camelyon17_metadata(root: str | Path) -> pd.DataFrame:
    """CAMELYON17 from raw/stages.csv and raw/annotations/*.xml (root = /media/abhi/camelyon17).

    negative slides -> slide_label 0; outlined macro/micro slides -> annotated, label 1;
    ITC slides and macro/micro slides without outlines -> excluded (tumour present but not
    localised, so neither class can be trusted)."""
    root = Path(root)
    st = pd.read_csv(root / "raw" / "stages.csv")
    sl = st[st["patient"].str.endswith(".tif")].copy()
    sl["slide_id"] = sl["patient"].str.replace(".tif", "", regex=False)
    sl["patient_id"] = sl["slide_id"].str.extract(r"(patient_\d+)")[0]
    annotated = {p.stem for p in (root / "raw" / "annotations").glob("*.xml")}
    outlined = sl["slide_id"].isin(annotated)
    negative = sl["stage"] == "negative"
    tumour = sl["stage"].isin(["macro", "micro"]) & outlined
    # Assign each column separately: setting a mixed [str, int] list on two columns at once
    # silently turns `label` into strings under pandas 2.x.
    sl["role"] = "excluded"
    sl.loc[negative, "role"] = "slide_label"
    sl.loc[tumour, "role"] = "annotated"
    sl["label"] = -1
    sl.loc[negative, "label"] = 0
    sl.loc[tumour, "label"] = 1
    sl["label"] = sl["label"].astype(int)
    sl["dataset"], sl["project_id"] = "camelyon17", "CAMELYON17"
    sl["center"] = sl["center"].astype(int).astype(str)
    sl["file"] = "raw/images/" + sl["patient"]
    sl["has_outline"] = outlined
    return sl[SCHEMA + ["stage", "has_outline"]].sort_values("slide_id").reset_index(drop=True)


def tcga_metadata(root: str | Path) -> pd.DataFrame:
    """TCGA cohort from <root>/metadata.csv written by scripts/download_tcga.py.

    center = tissue source site (2nd barcode field); slide_prep = DX (FFPE diagnostic) or
    TS/BS/MS (frozen). Recorded because in BRCA/COAD/LUAD every normal slide is frozen."""
    root = Path(root)
    m = pd.read_csv(root / "metadata.csv")
    out = m.copy()
    out["dataset"] = "tcga"
    out["center"] = out["patient_id"].astype(str).str.split("-").str[1]
    out["slide_prep"] = out["filename"].astype(str).str.extract(r"-(DX|TS|BS|MS)[0-9A-Z]")[0].fillna("unknown")
    out["label"] = out["label"].astype(int)
    out["role"] = out["label"].isin([0, 1]).map({True: "slide_label", False: "excluded"})
    out["file"] = "slides/" + out["filename"].astype(str)
    extras = [c for c in out.columns if c not in SCHEMA]
    return out[SCHEMA + extras].sort_values("slide_id").reset_index(drop=True)


ADAPTERS = {"camelyon17": camelyon17_metadata, "tcga": tcga_metadata}


def build_metadata(dataset: str, root: str | Path) -> pd.DataFrame:
    if dataset not in ADAPTERS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(ADAPTERS)}")
    df = ADAPTERS[dataset](root)
    validate_metadata(df)
    return df


def validate_metadata(df: pd.DataFrame) -> None:
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"metadata missing columns {missing}")
    if not df["slide_id"].is_unique:
        raise ValueError("slide_id must be unique")
    bad_role = set(df["role"]) - {"annotated", "slide_label", "excluded"}
    if bad_role:
        raise ValueError(f"unknown roles {bad_role}")
    used = df[df["role"] != "excluded"]
    if not used["label"].isin([0, 1]).all():
        raise ValueError("used slides must have label 0 or 1")
    multi = df.groupby("patient_id")["center"].nunique()
    if (multi > 1).any():
        raise ValueError(f"patients in several centers: {list(multi[multi > 1].index[:5])}")
