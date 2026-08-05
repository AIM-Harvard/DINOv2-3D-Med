#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_ssl "VoCoTrainer_BS8_ep150" \
  "$NNSSL_MODELS/VoCoTrainer_BS8_ep150__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth" \
  6 "Dataset006_Lung" \
  "$PRED/lung_voco_ep150" \
  "$LOG_DIR/lung_voco_ep150.txt"
