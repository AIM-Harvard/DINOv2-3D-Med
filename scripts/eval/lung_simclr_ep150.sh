#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_ssl "SimCLRTrainer_BS32_ep150" \
  "$NNSSL_MODELS/SimCLRTrainer_BS32_ep150__nnsslPlans__onemmiso/fold_all/checkpoint_final.pth" \
  6 "Dataset006_Lung" \
  "$PRED/lung_simclr_ep150" \
  "$LOG_DIR/lung_simclr_ep150.txt"
