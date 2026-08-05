#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

# One-time dataset setup (idempotent)
if [ ! -d "$nnUNet_raw/Dataset003_Liver" ]; then
  nnUNetv2_convert_MSD_dataset -i "$REPO/MSD/Task03_Liver" -overwrite_id 3
fi
nnUNetv2_plan_and_preprocess -d 3 --no_pp

run_ssl "DINOv2_PrimusM_exp2" \
  "$REPO/log/exported/model.pt" \
  3 "Dataset003_Liver" \
  "$PRED/liver_dinov2" \
  "$LOG_DIR/liver_dinov2.txt"
