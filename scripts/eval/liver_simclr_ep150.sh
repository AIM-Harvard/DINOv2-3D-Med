#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ ! -d "$nnUNet_raw/Dataset003_Liver" ]; then
  nnUNetv2_convert_MSD_dataset -i "$REPO/MSD/Task03_Liver" -overwrite_id 3
fi
nnUNetv2_plan_and_preprocess -d 3 --no_pp

run_ssl "SimCLRTrainer_BS32_ep150" \
  "$NNSSL_MODELS/SimCLRTrainer_BS32_ep150__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth" \
  3 "Dataset003_Liver" \
  "$PRED/liver_simclr_ep150" \
  "$LOG_DIR/liver_simclr_ep150.txt"
