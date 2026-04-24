#!/usr/bin/env bash
# nnUNet SSL Evaluation Pipeline
# Runs SSL fine-tuning vs baseline for Task06_Lung and Task03_Liver

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO=/home/suraj/Repositories/DINOv2_3D
MSD=$REPO/MSD
CKPT=$REPO/log/exported/model.pt
PRED=$REPO/nnunet/predictions

export nnUNet_raw=$REPO/nnunet/raw
export nnUNet_preprocessed=$REPO/nnunet/preprocessed
export nnUNet_results=$REPO/nnunet/results

PLAN_NAME="ptPlans__DINOv2_PrimusM_exp2____Spacing__1.00_1.00_1.00___Norm__Z"
LOG_DIR=$REPO/log/evaluation
mkdir -p "$LOG_DIR" "$PRED/ssl_lung_fold0" "$PRED/baseline_lung_fold0" \
         "$PRED/ssl_liver_fold0" "$PRED/baseline_liver_fold0"

# ── Task06_Lung ───────────────────────────────────────────────────────────────
echo "=== Task06_Lung: SSL fine-tuning fold 0 ==="
nnUNetv2_train_pretrained 6 3d_fullres 0 -p "$PLAN_NAME"

echo "=== Task06_Lung: SSL inference ==="
nnUNetv2_predict \
  -d 6 \
  -i "$nnUNet_raw/Dataset006_Lung/imagesTr" \
  -o "$PRED/ssl_lung_fold0" \
  -f 0 -c 3d_fullres -p "$PLAN_NAME" \
  --save_probabilities

echo "=== Task06_Lung: SSL evaluation ==="
nnUNetv2_evaluate_folder \
  "$nnUNet_raw/Dataset006_Lung/labelsTr" \
  "$PRED/ssl_lung_fold0" \
  -djfile "$nnUNet_raw/Dataset006_Lung/dataset.json" \
  -pfile "$nnUNet_preprocessed/Dataset006_Lung/gt_segmentations" \
  | tee "$LOG_DIR/ssl_lung_fold0_eval.txt"

echo "=== Task06_Lung: Baseline plan+preprocess ==="
nnUNetv2_plan_and_preprocess -d 6

echo "=== Task06_Lung: Baseline training fold 0 ==="
nnUNetv2_train 6 3d_fullres 0

echo "=== Task06_Lung: Baseline inference ==="
nnUNetv2_predict \
  -d 6 \
  -i "$nnUNet_raw/Dataset006_Lung/imagesTr" \
  -o "$PRED/baseline_lung_fold0" \
  -f 0 -c 3d_fullres

echo "=== Task06_Lung: Baseline evaluation ==="
nnUNetv2_evaluate_folder \
  "$nnUNet_raw/Dataset006_Lung/labelsTr" \
  "$PRED/baseline_lung_fold0" \
  -djfile "$nnUNet_raw/Dataset006_Lung/dataset.json" \
  | tee "$LOG_DIR/baseline_lung_fold0_eval.txt"

# ── Task03_Liver ──────────────────────────────────────────────────────────────
echo "=== Task03_Liver: Convert MSD dataset ==="
nnUNetv2_convert_MSD_dataset -i "$MSD/Task03_Liver" -overwrite_id 3

echo "=== Task03_Liver: SSL plan+preprocess ==="
nnUNetv2_plan_and_preprocess -d 3 --no_pp

echo "=== Task03_Liver: SSL preprocessing ==="
nnUNetv2_preprocess_like_nnssl \
  -d 3 \
  -n DINOv2_PrimusM_exp2 \
  -pc "$CKPT" \
  -am "like_pretrained" \
  -np 2

echo "=== Task03_Liver: SSL fine-tuning fold 0 ==="
nnUNetv2_train_pretrained 3 3d_fullres 0 -p "$PLAN_NAME"

echo "=== Task03_Liver: SSL inference ==="
nnUNetv2_predict \
  -d 3 \
  -i "$nnUNet_raw/Dataset003_Liver/imagesTr" \
  -o "$PRED/ssl_liver_fold0" \
  -f 0 -c 3d_fullres -p "$PLAN_NAME" \
  --save_probabilities

echo "=== Task03_Liver: SSL evaluation ==="
nnUNetv2_evaluate_folder \
  "$nnUNet_raw/Dataset003_Liver/labelsTr" \
  "$PRED/ssl_liver_fold0" \
  -djfile "$nnUNet_raw/Dataset003_Liver/dataset.json" \
  -pfile "$nnUNet_preprocessed/Dataset003_Liver/gt_segmentations" \
  | tee "$LOG_DIR/ssl_liver_fold0_eval.txt"

echo "=== Task03_Liver: Baseline plan+preprocess ==="
nnUNetv2_plan_and_preprocess -d 3

echo "=== Task03_Liver: Baseline training fold 0 ==="
nnUNetv2_train 3 3d_fullres 0

echo "=== Task03_Liver: Baseline inference ==="
nnUNetv2_predict \
  -d 3 \
  -i "$nnUNet_raw/Dataset003_Liver/imagesTr" \
  -o "$PRED/baseline_liver_fold0" \
  -f 0 -c 3d_fullres

echo "=== Task03_Liver: Baseline evaluation ==="
nnUNetv2_evaluate_folder \
  "$nnUNet_raw/Dataset003_Liver/labelsTr" \
  "$PRED/baseline_liver_fold0" \
  -djfile "$nnUNet_raw/Dataset003_Liver/dataset.json" \
  | tee "$LOG_DIR/baseline_liver_fold0_eval.txt"

echo ""
echo "All experiments complete. Results saved to $LOG_DIR/"
