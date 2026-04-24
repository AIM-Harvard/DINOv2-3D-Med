#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_ssl "DINOv2_PrimusM_exp2" \
  "$REPO/log/exported/model.pt" \
  6 "Dataset006_Lung" \
  "$PRED/lung_dinov2" \
  "$LOG_DIR/lung_dinov2.txt"
