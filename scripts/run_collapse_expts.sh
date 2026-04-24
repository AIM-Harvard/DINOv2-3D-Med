#!/usr/bin/env bash
# Run collapse-diagnosis experiments sequentially.
# Each run is killed first to free GPU memory before the next starts.
# Usage: bash scripts/run_collapse_expts.sh [exp_ids...]
#   e.g. bash scripts/run_collapse_expts.sh 00 03 07   (run specific experiments)
#        bash scripts/run_collapse_expts.sh              (run all)

set -euo pipefail

BASE_CONFIGS="./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/idc_dump.yaml"
EXPTS_DIR="./configs/experiments"

# Ordered list of all experiment IDs
ALL_EXPTS=(01 02 03 04 05 06 07 08)

# If args provided, use those; otherwise run all
EXPTS=("${@:-${ALL_EXPTS[@]}}")

for id in "${EXPTS[@]}"; do
    cfg="${EXPTS_DIR}/exp${id}_"*.yaml
    cfg=$(ls ${cfg} 2>/dev/null | head -1)
    if [[ -z "$cfg" ]]; then
        echo "ERROR: no config found for experiment id=${id}" >&2
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "  Starting experiment ${id}: $(basename ${cfg})"
    echo "  Config: ${cfg}"
    echo "============================================================"

    pkill -f -9 "python -m scripts.run" 2>/dev/null || true
    sleep 2

    python -m scripts.run fit \
        --config_file="${BASE_CONFIGS},${cfg}"

    echo "  Experiment ${id} done."
done

echo ""
echo "All experiments complete."
