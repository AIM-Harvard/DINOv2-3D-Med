#!/usr/bin/env bash
# Run all downstream evaluation experiments in order.
# Each subscript is self-contained and can also be run independently.
set -euo pipefail

DIR="$(dirname "$0")"

# ── Task06_Lung ───────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════"
echo " LUNG (Task06)"
echo "════════════════════════════════════════════"
# bash "$DIR/lung_baseline.sh"
bash "$DIR/lung_dinov2.sh"
# bash "$DIR/lung_base_eva_mae_ep150.sh"
# bash "$DIR/lung_base_eva_mae.sh"
# bash "$DIR/lung_base_mae_ep150.sh"
# bash "$DIR/lung_simclr_ep150.sh"
# bash "$DIR/lung_simclr_intrasample.sh"
# bash "$DIR/lung_voco_ep150.sh"

# # ── Task03_Liver ──────────────────────────────────────────────────────────────
# echo "════════════════════════════════════════════"
# echo " LIVER (Task03)"
# echo "════════════════════════════════════════════"
# bash "$DIR/liver_baseline.sh"
# bash "$DIR/liver_dinov2.sh"
# bash "$DIR/liver_base_eva_mae_ep150.sh"
# bash "$DIR/liver_base_eva_mae.sh"
# bash "$DIR/liver_base_mae_ep150.sh"
# bash "$DIR/liver_simclr_ep150.sh"
# bash "$DIR/liver_simclr_intrasample.sh"
# bash "$DIR/liver_voco_ep150.sh"

# echo ""
# echo "All experiments complete. Results in: $REPO/log/evaluation/"
