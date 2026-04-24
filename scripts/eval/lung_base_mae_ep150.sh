#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_ssl "BaseMAETrainer_BS8_ep150" \
  "$NNSSL_MODELS/BaseMAETrainer_BS8_ep150__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth" \
  6 "Dataset006_Lung" \
  "$PRED/lung_base_mae_ep150" \
  "$LOG_DIR/lung_base_mae_ep150.txt"
