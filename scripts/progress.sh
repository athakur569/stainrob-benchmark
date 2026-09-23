#!/usr/bin/env bash

PROJECT=$1

if [ -z "$PROJECT" ]; then
  echo "Usage: ./progress.sh <project_name>"
  echo "Example: ./progress.sh tcga_luad"
  exit 1
fi

ROOT="/media/abhi/${PROJECT}"

watch -n 5 "
TOTAL=\$((\$(wc -l < ${ROOT}/gdc_manifest.txt) - 1));
DONE=\$(find ${ROOT}/raw -type f -name '*.svs' | wc -l);
LEFT=\$((TOTAL-DONE));
PCT=\$(awk \"BEGIN {printf \\\"%.1f\\\", (TOTAL>0 ? (DONE/TOTAL)*100 : 0)}\");
echo \"Project: ${PROJECT}\";
echo \"Downloaded: \$DONE / \$TOTAL\";
echo \"Remaining: \$LEFT\";
echo \"Progress: \$PCT%\";
"p