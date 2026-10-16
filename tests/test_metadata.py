"""Tests for the per-dataset metadata adapters (common schema)."""

import pandas as pd
import pytest

from stain_benchmark.data.metadata import SCHEMA, build_metadata, validate_metadata


@pytest.fixture
def cam17_root(tmp_path):
    raw = tmp_path / "raw"; (raw / "annotations").mkdir(parents=True)
    rows = [("patient_000.zip", "pN1", 0)]
    stages = {0: "negative", 1: "macro", 2: "micro", 3: "itc", 4: "macro"}
    rows += [(f"patient_000_node_{i}.tif", s, 0) for i, s in stages.items()]
    pd.DataFrame(rows, columns=["patient", "stage", "center"]).to_csv(raw / "stages.csv", index=False)
    for i in (1, 2):  # nodes 1 and 2 outlined; node 4 is macro without outline
        (raw / "annotations" / f"patient_000_node_{i}.xml").write_text("<ASAP_Annotations/>")
    return tmp_path


def test_camelyon17_roles(cam17_root):
    raw = build_metadata("camelyon17", cam17_root)
    assert list(raw.columns[: len(SCHEMA)]) == SCHEMA
    m = raw.set_index("slide_id")
    assert m.loc["patient_000_node_0", "role"] == "slide_label" and m.loc["patient_000_node_0", "label"] == 0
    assert m.loc["patient_000_node_1", "role"] == "annotated" and m.loc["patient_000_node_1", "label"] == 1
    assert m.loc["patient_000_node_2", "role"] == "annotated" and m.loc["patient_000_node_2", "label"] == 1
    assert m.loc["patient_000_node_3", "role"] == "excluded"          # ITC
    assert m.loc["patient_000_node_4", "role"] == "excluded"          # macro, no outline
    assert m.loc["patient_000_node_0", "file"] == "raw/images/patient_000_node_0.tif"
    assert m.loc["patient_000_node_0", "center"] == "0"
    assert raw["label"].dtype.kind == "i"                              # integer labels in every pandas version


def test_tcga_center_prep_and_roles(tmp_path):
    pd.DataFrame({
        "slide_id": ["s1", "s2", "s3"],
        "patient_id": ["TCGA-E2-A1B5", "TCGA-E2-A1B5", "TCGA-A7-A0CE"],
        "filename": ["TCGA-E2-A1B5-01Z-00-DX1.x.svs", "TCGA-E2-A1B5-11A-01-TSA.y.svs", "TCGA-A7-A0CE-10A-01-BS1.z.svs"],
        "project_id": ["TCGA-BRCA"] * 3, "label": [1, 0, -1],
    }).to_csv(tmp_path / "metadata.csv", index=False)
    m = build_metadata("tcga", tmp_path).set_index("slide_id")
    assert list(m.center) == ["E2", "E2", "A7"]
    assert list(m.slide_prep) == ["DX", "TS", "BS"]
    assert list(m.role) == ["slide_label", "slide_label", "excluded"]
    assert m.loc["s1", "file"].startswith("slides/")


def test_validate_rejects_patient_in_two_centers():
    df = pd.DataFrame({"dataset": "x", "project_id": "x", "slide_id": ["a", "b"], "patient_id": ["p", "p"],
                       "center": ["0", "1"], "label": [0, 1], "role": "slide_label", "file": ["a", "b"]})
    with pytest.raises(ValueError, match="several centers"):
        validate_metadata(df)
