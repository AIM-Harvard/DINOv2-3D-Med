# DINOv2-3D: Self-Supervised 3D Vision Transformer Pretraining

> ⚠️ **Warning:** This implementation has **not** been extensively benchmarked and you should **not** expect it to work out of the box for your specific task. It is primarily a reimplementation codebase aimed at providing easier understanding and cleaner interfaces for DINOv2 in 3D medical imaging. If you need a well-tuned configuration for a specific medical imaging task, finding and validating that config is on you — although this repo includes plentiful debugging tools (see [Collapse Monitoring Callbacks](#collapse-monitoring-callbacks)) to help you get there. Feel free to [open an issue](https://github.com/AIM-Harvard/DINOv2-3D-Med/issues) and I'm happy to chat!

A configuration-first (and therefore easily understandable and trackable) repository for a 3D implementation of DINOv2. Based on the implementations from Lightly (Thank you!) and integrated with PyTorch Lightning. 3D capabilities of this implementation are largely through MONAI's functionalities.

## What you can do with this Repo
- Train your own 3D DINOv2 on CT, MRI, PET data, etc. with very little configuration other than what's been provided.
- Use state of the art PRIMUS transformer in medical segmentation to pretrain your DINOv2.
- Swap in different backbones (PRIMUS, EVA, MONAI ViT) via config files — no code changes needed.
- Extend to DINOv3-style Gram anchoring for improved spatial coherence.
- Run multimodal (vision + text) pretraining with a CLIP-based text encoder.
- Export pretrained checkpoints to nnUNet format for downstream segmentation tasks.
- Monitor training health with three built-in collapse detection callbacks.
- Make a baseline for DINOv2 to improve and build on.
- Change elements of the framework through modular extensions.

## Features
- DINOv2-style self-supervised learning with teacher-student EMA framework
- DINOv3-style Gram anchoring loss for spatial coherence (stage-aware training)
- Multiple 3D backbone architectures: PRIMUS, EVA, and MONAI ViT
- Block masking for 3D volumes (iBOT-style masked patch prediction)
- Flexible 3D augmentations (global/local views) courtesy of MONAI
- FP16-safe projection heads (fixes known numerical instability in mixed precision)
- Layer-wise learning rate decay and weight decay scheduling
- Multimodal pretraining support (vision + text alignment)
- Three collapse monitoring callbacks (Gram matrix, entropy, effective rank)
- PyTorch Lightning training loop with DDP support
- YAML-based experiment configuration that is explainable at a glance due to its abstraction!
- Checkpoint export to nnUNet format

## Repository Structure

```
DINOv2-3D-Med/
├── callbacks/               # Collapse monitoring callbacks (W&B integration)
│   ├── gram_matrix_logger.py
│   ├── entropy_logger.py
│   └── effective_rank_logger.py
├── configs/                 # Composable YAML configuration files
│   ├── train.yaml           # Main training config
│   ├── predict.yaml         # Inference config
│   ├── dinotxt_stage.yaml   # Multimodal training config
│   ├── models/              # Backbone configs (primus.yaml, vit.yaml)
│   └── datasets/            # Dataset configs (amos.yaml, idc_dump.yaml)
├── losses/                  # Loss functions
│   ├── dino.py              # DINOv2 loss (DINO + iBOT + KoLeo)
│   ├── dinov3.py            # DINOv3 loss (+ Gram anchoring)
│   ├── ibot_patch_3d.py     # 3D masked patch prediction loss
│   └── image_text_alignment.py  # Multimodal alignment loss
├── models/                  # Model architectures
│   ├── meta_arch.py         # Teacher-student meta-architecture
│   ├── multimodal_meta_arch.py  # Multimodal meta-architecture
│   ├── dynamic_utils.py     # Dynamic backbone utilities
│   ├── rope.py              # 3D Rotary position embeddings
│   └── backbones/           # Vision & text backbone implementations
│       ├── primus.py        # PRIMUS lightweight 3D transformer
│       ├── eva.py           # EVA large-scale transformer
│       ├── masked_vit_wrapper.py  # MONAI ViT wrapper with masking
│       ├── vision_enc_wrapper.py  # Generic vision encoder wrapper
│       └── text_encoder.py  # CLIP-based text encoder
├── training/                # PyTorch Lightning modules
│   ├── dinov2_lightning_module.py  # DINOv2 LightningModule
│   ├── dinotxt_lightning_module.py # Multimodal LightningModule
│   └── data_module.py       # DataModule for train/val/test/predict
├── transforms/              # Data augmentation pipelines
│   ├── dinov2_aug.py        # 3D global/local view augmentation
│   ├── blockmask.py         # 3D random block masking for iBOT
│   └── random_resized_crop.py  # 3D random resized cropping
├── utils/                   # Utility functions
│   ├── imports.py           # Dynamic module loading
│   └── safe_dataset.py      # Error-resilient dataset wrapper
├── scripts/                 # Entry points & utilities
│   ├── run.py               # Main CLI (training & prediction)
│   └── utility/
│       └── export_ckpt_to_nnunet.py  # Checkpoint export to nnUNet
└── tests/                   # Unit tests
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AIM-Harvard/DINOv2-3D-Med.git
   cd DINOv2-3D-Med
   ```
2. Create a virtual environment with UV (recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```

If you do not want to use uv, you could just as easily do a `pip install -e .` in the repo directory.

## Usage

### Training
Run the training script with the default training config:
```bash
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/amos.yaml
```

Here the `train.yaml` contains most of the heart of the configuration. `primus.yaml` provides the backbone to use for DINOv2 and `amos.yaml` provides the path to the dataset to be used.

You can override any config value from the command line:
```bash
python -m scripts.run fit \
  --config_file=./configs/train.yaml,./configs/models/primus.yaml,./configs/datasets/amos.yaml \
  --trainer#max_epochs=300 \
  --lightning_module#base_lr=0.002
```

### Prediction / Inference
Run inference with a trained checkpoint using the prediction config:
```bash
python -m scripts.run predict --config_file=./configs/predict.yaml,./configs/models/primus.yaml,./configs/datasets/amos.yaml
```

The `predict.yaml` config sets up the model in inference mode using `model.encode()` to extract CLS token features from input volumes.

### Configuration
All experiment settings (model, trainer, data) are defined in composable YAML configs using MONAI's `ConfigParser`:

| Config | Purpose |
|--------|---------|
| `configs/train.yaml` | Main training setup (trainer, LR, loss, callbacks, augmentations) |
| `configs/predict.yaml` | Inference / feature extraction setup |
| `configs/dinotxt_stage.yaml` | Multimodal (vision + text) training |
| `configs/models/primus.yaml` | PRIMUS backbone configuration |
| `configs/models/vit.yaml` | MONAI ViT backbone configuration |
| `configs/datasets/amos.yaml` | AMOS dataset paths |
| `configs/datasets/idc_dump.yaml` | IDC dataset paths |

Configs are merged left-to-right: later files override earlier ones. This lets you mix and match backbones and datasets without editing the main training config.

## Architecture

### Teacher-Student Framework
The core architecture (`models/meta_arch.py`) follows the DINOv2 self-supervised paradigm:

- **Student** backbone is trained via gradient descent
- **Teacher** backbone is updated via Exponential Moving Average (EMA) with a cosine momentum schedule (0.992 → 1.0)
- Both share the same architecture but have separate DINO and iBOT projection heads
- The projection heads use **FP16-safe normalization** (`eps=1e-6` instead of `1e-12`) to prevent `inf`/`NaN` during mixed precision training

### Backbones

Three interchangeable 3D vision transformer backbones are available:

| Backbone | Config | Description |
|----------|--------|-------------|
| **PRIMUS** | `models/primus.yaml` | Lightweight 3D transformer built on Dynamic Network Architectures with EVA attention blocks, rotary position embeddings (RoPE), SwiGLU MLPs, and configurable drop path / patch dropout. Default choice. |
| **EVA** | (used internally by PRIMUS) | Large-scale vision transformer with combined absolute + rotary position embeddings, SwiGLU MLPs, and support for up to 24 layers / 16 heads. |
| **MONAI ViT** | `models/vit.yaml` | Wrapped MONAI `ViT` with masking support for iBOT training. Good for simpler setups or when using existing MONAI-based pipelines. |

Switch backbones by changing which model config file you pass:
```bash
# Use PRIMUS (default)
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/primus.yaml,...

# Use MONAI ViT
python -m scripts.run fit --config_file=./configs/train.yaml,./configs/models/vit.yaml,...
```

### Loss Functions

| Loss | Module | Description |
|------|--------|-------------|
| **DINOv2Loss** | `losses/dino.py` | Combined DINO (CLS token contrastive) + iBOT (masked patch prediction) + KoLeo (diversity regularization) with dynamic teacher temperature warmup. |
| **DINOv3Loss** | `losses/dinov3.py` | Extends DINOv2Loss with Gram anchoring for spatial coherence. Stage-aware: pretrain → gram_anchor → high_res. Based on [arXiv:2508.10104](https://arxiv.org/abs/2508.10104). |
| **IBOTPatchLoss3D** | `losses/ibot_patch_3d.py` | Patch-level masked prediction loss adapted for 3D volumes. |
| **ImageTextAlignmentLoss** | `losses/image_text_alignment.py` | Contrastive alignment loss for multimodal (vision + text) pretraining. |

To use DINOv3Loss instead of DINOv2Loss, specify it as the criterion in your config:
```yaml
lightning_module:
  criterion:
    _target_: project.losses.dinov3.DINOv3Loss
    gram_loss_weight: 0.1
    training_stage: "gram_anchor"
    # ... other DINOv2Loss params carry over
```

### Training Details

The Lightning module (`training/dinov2_lightning_module.py`) implements several training techniques from the DINOv2 paper:

- **LR scaling**: Base LR is scaled by `sqrt(effective_batch_size / 1024)` where effective batch = per-device batch × devices × gradient accumulation steps
- **Layer-wise LR decay**: Each transformer block gets a decayed LR (`layer_decay^(num_layers + 1 - layer_idx)`), with `layer_decay=0.9` by default
- **Weight decay scheduling**: Cosine schedule from 0.04 → 0.4 over training
- **Last layer LR zeroing**: During warmup, the last projection layer's LR is set to zero (DINOv2 mechanism, preserves optimizer state unlike gradient-zeroing)
- **Gradient clipping**: Norm-based clipping at 3.0
- **Cosine warmup scheduler**: 10-epoch warmup → cosine annealing to `min_lr`

### Multimodal Training

For vision-language pretraining, use `configs/dinotxt_stage.yaml` with the `DINOtxt_LightningModule`:

```bash
python -m scripts.run fit --config_file=./configs/dinotxt_stage.yaml,./configs/datasets/amos.yaml
```

This combines the PRIMUS vision backbone with a CLIP-based text encoder (`openai/clip-vit-base-patch32`) for image-text alignment learning. The text encoder can be frozen or fine-tuned.

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

It looks something like this:

```json
{
   "training": [
      {"image": "<path_to_image>"},
      ...
   ]
}
```
If you'd like to do more complex manipulations like sample based on a mask and so on, you can easily extend this JSON to include a `"label"` in addition to the image and use MONAI transforms to sample as you like.

The default training pipeline applies the following MONAI transforms in sequence:
`LoadImaged` → `EnsureChannelFirstd` → `Orientationd` → `Spacingd` → `CropForegroundd` → `SpatialPadd` → `ScaleIntensityRanged` → `RandSpatialCropd` → `DINOv2Augmentation3D`

## Exporting Checkpoints to nnUNet

A utility script converts Lightning checkpoints to the format expected by nnUNet for downstream segmentation:

```bash
python scripts/utility/export_ckpt_to_nnunet.py \
  path/to/checkpoint.ckpt \
  path/to/output.pt \
  --arch-class-name PrimusM \
  --pretrain-patch-size 96 96 96 \
  --pretrain-spacing 1.0 1.0 1.0
```

This extracts the student backbone weights, renames keys from `vit` → `eva` naming, optionally removes the CLS token from positional embeddings, and wraps everything in the `nnssl_adaptation_plan` metadata that nnUNet expects. See `--help` for all options.

## References
- [Lightly](https://github.com/lightly-ai/lightly)
- [DINOv2 (Facebook Research)](https://github.com/facebookresearch/dinov2)
- [DINOv3 (arXiv:2508.10104)](https://arxiv.org/abs/2508.10104)
- [MONAI (Medical Open Network for AI)](https://github.com/Project-MONAI/MONAI)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Dynamic Network Architectures / PRIMUS](https://github.com/MIC-DKFZ/dynamic-network-architectures)
- [nnUNet](https://github.com/MIC-DKFZ/nnUNet)

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
