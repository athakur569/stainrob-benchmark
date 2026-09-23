#!/usr/bin/env bash
set -euo pipefail

# usage:
#   bash scripts/tcga_bootstrap.sh TCGA-BRCA /media/abhi/tcga_brca_test 2 2 v1
#   bash scripts/tcga_bootstrap.sh TCGA-BRCA /media/abhi/tcga_brca_full 10000 6 v1
#
# args:
#   1: project (e.g., TCGA-BRCA)
#   2: tcga_root (absolute path, e.g., /media/abhi/tcga_brca_full)
#   3: max_cases (integer; 2 for smoke test, 10000 for "all")
#   4: threads (gdc-client parallel threads)
#   5: split_version (string; bump when you intentionally change split protocol, e.g. v1, v2)
#
# behavior:
#   - always refresh full metadata from GDC query
#   - always build a pending-only manifest after checking what is already present locally
#   - only downloads missing slides
#   - always re-flattens raw/*.svs into slides/
#   - always refreshes splits from updated metadata while preserving frozen patient universe

PROJECT="${1:?project required}"
TCGA_ROOT="${2:?tcga_root required}"
MAX_CASES="${3:?max_cases required}"
THREADS="${4:?threads required}"
SPLIT_VERSION="${5:?split_version required}"

RAW_DIR="$TCGA_ROOT/raw"
SLIDES_DIR="$TCGA_ROOT/slides"
SPLITS_DIR="$TCGA_ROOT/splits"

FULL_MANIFEST="$TCGA_ROOT/gdc_manifest_all.txt"
PENDING_MANIFEST="$TCGA_ROOT/gdc_manifest_pending.txt"
METADATA="$TCGA_ROOT/metadata.csv"
LOGFILE="$RAW_DIR/gdc_download.log"
UNIVERSE_TXT="$SPLITS_DIR/patient_universe.txt"

mkdir -p "$RAW_DIR" "$SLIDES_DIR" "$SPLITS_DIR"

echo "============================================================"
echo "project       : $PROJECT"
echo "tcga_root     : $TCGA_ROOT"
echo "max_cases     : $MAX_CASES"
echo "threads       : $THREADS"
echo "split_version : $SPLIT_VERSION"
echo "mode          : broad cohort (--all-slides)"
echo "============================================================"

command -v gdc-client >/dev/null 2>&1 || {
  echo "error: gdc-client not found in PATH"
  exit 1
}

# ------------------------------------------------------------
# 0) pre-flatten any previously downloaded raw files
# ------------------------------------------------------------
echo "[0/4] pre-flattening any raw .svs into $SLIDES_DIR (skip existing)..."
find "$RAW_DIR" -type f \( -name "*.svs" -o -name "*.SVS" \) -print0 2>/dev/null \
  | xargs -0 -I {} mv -n "{}" "$SLIDES_DIR/" || true

# ------------------------------------------------------------
# 1) refresh full manifest + metadata + pending manifest
# ------------------------------------------------------------
echo "[1/4] refreshing metadata + manifests (full + pending-only)..."
python scripts/download_tcga.py \
  --project "$PROJECT" \
  --output "$TCGA_ROOT" \
  --max-cases "$MAX_CASES" \
  --manifest-only \
  --all-slides

test -f "$FULL_MANIFEST" || { echo "error: full manifest not created: $FULL_MANIFEST"; exit 1; }
test -f "$PENDING_MANIFEST" || { echo "error: pending manifest not created: $PENDING_MANIFEST"; exit 1; }
test -f "$METADATA" || { echo "error: metadata not created: $METADATA"; exit 1; }

FULL_ROWS=$(python - <<PY2
from pathlib import Path
p = Path("$FULL_MANIFEST")
print(max(sum(1 for _ in p.open()) - 1, 0))
PY2
)
PENDING_ROWS=$(python - <<PY2
from pathlib import Path
p = Path("$PENDING_MANIFEST")
print(max(sum(1 for _ in p.open()) - 1, 0))
PY2
)
SKIP_ROWS=$((FULL_ROWS - PENDING_ROWS))

echo "[1b/4] visual preflight check"
echo "  queried total : $FULL_ROWS"
echo "  skip existing : $SKIP_ROWS"
echo "  will download : $PENDING_ROWS"

echo "  examples:"
python - <<PY2
import csv
from pathlib import Path
full_p = Path("$FULL_MANIFEST")
pending_p = Path("$PENDING_MANIFEST")
def names(path):
    with path.open() as f:
        return [row["filename"] for row in csv.DictReader(f, delimiter="	")]
full = names(full_p)
pending = names(pending_p)
pending_set = set(pending)
skipped = [x for x in full if x not in pending_set][:5]
print("    skip:")
if skipped:
    for x in skipped:
        print(f"      - {x}")
else:
    print("      - <none>")
print("    download:")
if pending:
    for x in pending[:5]:
        print(f"      - {x}")
else:
    print("      - <none>")
PY2

# ------------------------------------------------------------
# 2) download only missing slides, then flatten again
# ------------------------------------------------------------
echo "[2/4] downloading only missing slides (if any)..."
if [ "$PENDING_ROWS" -le 0 ]; then
  echo "pending manifest is empty -> nothing new to download"
else
  gdc-client download     -m "$PENDING_MANIFEST"     -d "$RAW_DIR"     -n "$THREADS"     --retry-amount 50     --log-file "$LOGFILE" || {
      echo "warning: gdc-client returned non-zero (network hiccup or partial)."
      echo "rerun this script; it will resume and skip what is already present."
    }
fi

echo "[2b/4] flattening .svs into $SLIDES_DIR (skip existing)..."
find "$RAW_DIR" -type f \( -name "*.svs" -o -name "*.SVS" \) -print0 2>/dev/null \
  | xargs -0 -I {} mv -n "{}" "$SLIDES_DIR/" || true
SVS_COUNT="$(find "$SLIDES_DIR" -maxdepth 1 -type f \( -name "*.svs" -o -name "*.SVS" \) | wc -l | tr -d ' ')"
echo "slides in slides/ : $SVS_COUNT"

# ------------------------------------------------------------
# 3) freeze patient universe once, then recreate splits from current metadata
# ------------------------------------------------------------
echo "[3/4] ensuring frozen patient universe -> $UNIVERSE_TXT"
if [ -f "$UNIVERSE_TXT" ]; then
  echo "patient universe already exists (reusing)."
else
  python - <<PY
import pandas as pd
from pathlib import Path
meta = Path("${METADATA}")
out = Path("${UNIVERSE_TXT}")
df = pd.read_csv(meta)
if "patient_id" not in df.columns:
    raise SystemExit("metadata.csv missing required column: patient_id")
patients = sorted(df["patient_id"].dropna().astype(str).unique().tolist())
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(patients) + "\n")
print(f"wrote {len(patients)} patient_ids -> {out}")
PY
fi

echo "[4/4] recreating patient-level splits from refreshed metadata..."
python scripts/create_splits.py \
  --metadata "$METADATA" \
  --output "$SPLITS_DIR" \
  --slide-id-column slide_id \
  --patient-id-column patient_id \
  --patient-universe "$UNIVERSE_TXT" \
  --project-id "$PROJECT" \
  --split-version "$SPLIT_VERSION" \
  --force

echo "done."
