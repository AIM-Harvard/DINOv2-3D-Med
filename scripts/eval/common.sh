#!/usr/bin/env bash
# Shared env vars and helper functions — sourced by all eval scripts

REPO=/home/suraj/Repositories/DINOv2_3D
NNSSL_MODELS=/mnt/data1/CT_FM/nnssl/nnssl_results/Dataset043_IDC
PRED=$REPO/nnunet/predictions
LOG_DIR=$REPO/log/evaluation

export nnUNet_raw=$REPO/nnunet/raw
export nnUNet_preprocessed=$REPO/nnunet/preprocessed
export nnUNet_results=$REPO/nnunet/results

mkdir -p "$LOG_DIR"

# Batch size for all runs — override with e.g. BATCH_SIZE=16 bash scripts/eval/lung_dinov2.sh
BATCH_SIZE=${BATCH_SIZE:-8}

# Build the plan name produced by nnUNetv2_preprocess_like_nnssl for a given
# pretrain name (all models use 1mm isotropic + ZScore → same data_identifier)
plan_name() { echo "ptPlans__${1}____Spacing__1.00_1.00_1.00___Norm__Z"; }

# Patch batch_size in a plans JSON in-place
set_batch_size() {
  local plans_json=$1
  python3 -c "
import json, sys
p = json.load(open('$plans_json'))
p['configurations']['3d_fullres']['batch_size'] = $BATCH_SIZE
json.dump(p, open('$plans_json', 'w'), indent=2)
print('  batch_size set to $BATCH_SIZE in', '$plans_json')
"
}

# ── SSL model: preprocess + train + infer + eval ──────────────────────────────
# Args: pretrain_name  checkpoint_path  dataset_id  dataset_folder  pred_dir  eval_log
run_ssl() {
  local name=$1 ckpt=$2 did=$3 dset=$4 pred=$5 log=$6
  local plan; plan=$(plan_name "$name")
  local plans_json="$nnUNet_preprocessed/$dset/${plan}.json"

  echo ">>> [$name / $dset] Preprocessing"
  nnUNetv2_preprocess_like_nnssl -d "$did" -n "$name" -pc "$ckpt" -am "like_pretrained" -np 2

  echo ">>> [$name / $dset] Training fold 0"
  nnUNetv2_train_pretrained "$did" 3d_fullres 0 -p "$plan"

  echo ">>> [$name / $dset] Inference"
  mkdir -p "$pred"
  nnUNetv2_predict \
    -d "$did" -i "$nnUNet_raw/$dset/imagesTr" -o "$pred" \
    -f 0 -c 3d_fullres -p "$plan" --save_probabilities

  echo ">>> [$name / $dset] Evaluation"
  nnUNetv2_evaluate_folder \
    "$nnUNet_raw/$dset/labelsTr" "$pred" \
    -djfile "$nnUNet_raw/$dset/dataset.json" \
    -pfile  "$nnUNet_preprocessed/$dset/gt_segmentations" \
    | tee "$log"
}

# ── Baseline: train + infer + eval (plan+preprocess assumed already done) ─────
# Args: dataset_id  dataset_folder  pred_dir  eval_log
run_baseline() {
  local did=$1 dset=$2 pred=$3 log=$4

  echo ">>> [baseline / $dset] plan_and_preprocess"
  nnUNetv2_plan_and_preprocess -d "$did"

  echo ">>> [baseline / $dset] Training fold 0"
  nnUNetv2_train "$did" 3d_fullres 0

  echo ">>> [baseline / $dset] Inference"
  mkdir -p "$pred"
  nnUNetv2_predict \
    -d "$did" -i "$nnUNet_raw/$dset/imagesTr" -o "$pred" \
    -f 0 -c 3d_fullres

  echo ">>> [baseline / $dset] Evaluation"
  nnUNetv2_evaluate_folder \
    "$nnUNet_raw/$dset/labelsTr" "$pred" \
    -djfile "$nnUNet_raw/$dset/dataset.json" \
    | tee "$log"
}
