#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ ! -d "$nnUNet_raw/Dataset003_Liver" ]; then
  nnUNetv2_convert_MSD_dataset -i "$REPO/MSD/Task03_Liver" -overwrite_id 3
fi
nnUNetv2_plan_and_preprocess -d 3 --no_pp

run_ssl "VoCoTrainer_BS8_ep150" \
  "$NNSSL_MODELS/VoCoTrainer_BS8_ep150__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth" \
  3 "Dataset003_Liver" \
  "$PRED/liver_voco_ep150" \
  "$LOG_DIR/liver_voco_ep150.txt"
