# DINOv2-3D: Self-Supervised 3D Vision Transformer Pretraining

A configuration-first (and therefore easily understandable and trackable) repository for a 3D implementation od DINOv2. Based on the implementations from Lightly (Thank you!) and integrated with Pytorch Lightning. 3D capabilities of this implementation are largely through MONAI's functionalities

## What you can do with this Repo
- Train your own 3D Dinov2 on CT, MRI, PET data, etc. with very little configuration other than whats been provided. 
- Use state of the art PRIMUS transformer in medical segmentation to pretrain your DINOV2
- Make a baseline for DinoV2 to improve and build on.
- Change elements of the framework through modular extensions. 

## Features
- DINOv2-style self-supervised learning with teacher-student models
- Block masking for 3D volumes 
- Flexible 3D augmentations (global/local views) courtesy of MONAI
- PyTorch Lightning training loop 
- YAML-based experiment configuration that is explainable at a glance due to its abstraction!


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AIM-Harvard/DINOv2-3D-Med.git
   cd DINOv2_3D
   ```
2. Create a virtual environment with UV(recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```

If you do not want to use uv, you could just as easily do a `pip install -e .` in the repo directory

## Usage
### Training
Run the training script with the default training config:
```bash
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/amos.yaml
```

Here the train.yaml contains most of the heart of the configuration. primus.yaml provides the backbone to use for DINOv2 and amos.yaml provides the path to the dataset to be used.


### Configuration
- All experiment settings (model, trainer, data) are defined in YAML configs.
- `configs/train.yaml`: Main training configuration with complete setup
- `configs/predict.yaml`: Configuration for inference/prediction tasks

### Gram Matrix Debugging
This repo includes `project.callbacks.GramMatrixCallback` (configured in `configs/train.yaml`) to monitor representation collapse and logging health during training.

Debug metrics logged to Weights & Biases:
- `gram_matrix_debug/offdiag_mean`: Mean off-diagonal cosine similarity (higher can indicate collapse).
- `gram_matrix_debug/offdiag_std`: Spread of off-diagonal similarities (very low can indicate near-constant representations).
- `gram_matrix_debug/sample_variance`: Mean per-dimension variance of monitored features.
- `gram_matrix_debug/fallback_used`: `1.0` when fallback backbone features were used after saturation detection.
- `gram_matrix_debug/root_cause_framework`: `1.0` when fallback resolves saturation (likely feature/key/pipeline issue).
- `gram_matrix_debug/root_cause_model`: `1.0` when saturation persists on primary and fallback features.
- `gram_matrix_debug/root_cause_unknown`: `1.0` when saturation is detected but no fallback feature is available.
Stage diagnostics also logged:
- `gram_matrix_debug/stage_feature_extraction_offdiag_mean|std|sample_variance`
- `gram_matrix_debug/stage_normalization_offdiag_mean|std|sample_variance`
- `gram_matrix_debug/stage_sampling_offdiag_mean|std|sample_variance`
- `gram_matrix_debug/stage_gather_offdiag_mean|std|sample_variance`

Key callback knobs (`configs/train.yaml`):
- `feature_key`: Feature source under model `pred` outputs (default here: `student_cls_token`).
- `saturation_offdiag_threshold` and `saturation_offdiag_std_threshold`: Collapse gate thresholds.
- `saturation_sample_var_threshold`: Optional extra gate. Set to `null` to disable variance gating.
- `auto_fallback_to_backbone_on_saturation`: Automatically retry using backbone feature keys.
- `max_samples`: Caps gram-matrix sample count to reduce memory/time overhead.

Common debugging checklist:
1. If no gram matrix appears in W&B, verify `WANDB_DISABLED` is not set and `WANDB_MODE` is not `disabled`.
2. If warnings report missing feature keys, confirm `feature_key` exists in `model_outputs["pred"]`.
3. If you see repeated saturation with `root_cause_model=1`, inspect augmentation, learning rate, and collapse-sensitive hyperparameters.
4. If saturation disappears only when fallback is used, inspect head/projection-space behavior and keep fallback diagnostics enabled.
5. If training is slowed by callback overhead, lower logging frequency (`log_every_n_steps`) or reduce `max_samples`.

## Data Preparation

For now, to run a straightforward DINOv2 pipeline, all you need to do is setup your data paths in a JSON in the MONAI format. 

It looks something like this

```json
{
   "training": [
      {"image": <path_to_image>},
      ....
   ]
}
```
If you'd like to do more complex manipulations like sample based on a mask and so on, you can easily extend this json to include a "label" in addition to the image and use MONAI transforms to sample as you like.

## References
- [Lightly](https://github.com/lightly-ai/lightly)
- [DINOv2 (Facebook Research)](https://github.com/facebookresearch/dinov2)
- [MONAI (Medical Open Network for AI)](https://github.com/Project-MONAI/MONAI)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)


## License
Copyright &copy; 2025 Suraj Pai, Vasco Prudente

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
