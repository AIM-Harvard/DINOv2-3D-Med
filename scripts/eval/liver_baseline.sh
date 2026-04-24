#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ ! -d "$nnUNet_raw/Dataset003_Liver" ]; then
  nnUNetv2_convert_MSD_dataset -i "$REPO/MSD/Task03_Liver" -overwrite_id 3
fi

run_baseline 3 "Dataset003_Liver" \
  "$PRED/liver_baseline" \
  "$LOG_DIR/liver_baseline.txt"
