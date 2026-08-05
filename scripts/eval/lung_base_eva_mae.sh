#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_ssl "BaseEvaMAETrainer_BS8" \
  "$NNSSL_MODELS/BaseEvaMAETrainer_BS8__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth" \
  6 "Dataset006_Lung" \
  "$PRED/lung_base_eva_mae" \
  "$LOG_DIR/lung_base_eva_mae.txt"
