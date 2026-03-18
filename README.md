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

### Collapse Monitoring Callbacks

Three callbacks (configured in `configs/train.yaml`) track representation health during training. All log to Weights & Biases under the `collapse_monitor/` and `gram_matrix_debug/` namespaces.

#### GramMatrixCallback
Visualizes pairwise cosine similarity across a batch as a heatmap. Detects representational collapse by checking whether all features become nearly identical.

W&B metrics:
- `gram_matrix`: heatmap image logged every `log_every_n_steps`
- `gram_matrix_debug/offdiag_mean`: mean off-diagonal cosine similarity — rising toward 1.0 signals collapse
- `gram_matrix_debug/offdiag_std`: spread of off-diagonal values — near zero means near-constant representations
- `gram_matrix_debug/fallback_used`: `1.0` when backbone features replaced saturated projected features

When `fallback_used=1.0`, a `logging.warning` fires identifying whether the collapse is framework-path (fallback healthy) or model-side (fallback also saturated). Check training logs for the exact message.

Key knobs:
- `feature_key`: which feature to monitor (`student_cls_token` by default — changes faster than EMA teacher)
- `saturation_offdiag_threshold` / `saturation_offdiag_std_threshold`: gate thresholds for saturation
- `saturation_sample_var_threshold`: optional third gate, `null` to rely on off-diagonal criteria only
- `auto_fallback_to_backbone_on_saturation`: retry with backbone keys to isolate collapse source
- `max_samples`: caps gram matrix size (memory/speed trade-off)

Troubleshooting:
1. No heatmap in W&B → check `WANDB_DISABLED` is unset and `WANDB_MODE` is not `disabled`
2. Warning about missing feature key → confirm `feature_key` exists in `model_outputs["pred"]`
3. `fallback_used` stays 0 but `offdiag_mean` is high → model-side collapse; check LR, augmentation, centering
4. `fallback_used` flips to 1 → projection-head saturation, backbone is still healthy (often recovers)
5. Callback overhead → increase `log_every_n_steps` or reduce `max_samples`

#### TeacherEntropyCallback
Monitors the Shannon entropy of the teacher's DINO-centered softmax distribution — the distribution the loss actually trains against.

W&B metrics:
- `collapse_monitor/teacher_entropy`: entropy in nats
- `collapse_monitor/teacher_entropy_normalized`: entropy / log(output_dim), in [0, 1]

Interpretation:
- **Healthy**: normalized entropy ≈ 0.7–1.0 (teacher assigns mass across many prototypes)
- **Collapse warning**: normalized entropy < 0.1 (teacher peaked on a few dimensions)

Troubleshooting:
1. Entropy drops suddenly → check centering (`dino_loss_fn.center`) and teacher temperature schedule
2. Warning about `criterion.dino_loss_fn` not found → loss must expose `dino_loss_fn` attribute
3. Entropy near 0 from the start → teacher temperature may be too low; check `teacher_temp_min`

#### EffectiveRankCallback
Computes the Roy & Vetterli (2007) effective rank of CLS token embeddings via SVD, measuring how many dimensions the model actively uses.

W&B metrics:
- `collapse_monitor/effective_rank`: effective rank (≥ 1.0)
- `collapse_monitor/num_svd_samples`: number of samples used for the SVD estimate

Interpretation:
- **Healthy**: effective rank ≈ 50–500+ (representations spread across many dimensions)
- **Soft collapse warning**: effective rank < 10
- **Hard collapse**: effective rank < 2 (near-rank-1 — all samples map to a line)

Troubleshooting:
1. Effective rank near 1 from epoch 1 → model may not be learning; check loss is decreasing
2. Effective rank drops mid-training → potential collapse; cross-reference with teacher entropy and gram matrix
3. `num_svd_samples` is very low → increase `max_buffer_samples` or check DDP gather is working

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
