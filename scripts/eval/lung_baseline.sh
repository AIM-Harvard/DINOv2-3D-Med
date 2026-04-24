#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_baseline 6 "Dataset006_Lung" \
  "$PRED/lung_baseline" \
  "$LOG_DIR/lung_baseline.txt"
