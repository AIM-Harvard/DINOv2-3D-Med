# DINOv2-3D: Self-Supervised 3D Vision Transformer Pretraining

A configuration-driven repository for 3D DINOv2 self-supervised learning. Built with [Lighter](https://github.com/project-lighter/lighter), PyTorch Lightning, and MONAI.

## What You Can Do with This Repo
- Train your own 3D DINOv2 on CT, MRI, PET data, etc. with minimal configuration
- Use state-of-the-art PRIMUS transformer for medical imaging pretraining
- Make a baseline for DINOv2 to improve and build on
- Change elements of the framework through modular extensions

## Features
- DINOv2-style self-supervised learning with teacher-student models
- Block masking for 3D volumes
- Flexible 3D augmentations (global/local views) courtesy of MONAI
- PyTorch Lightning training loop
- YAML-based experiment configuration powered by Lighter

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

If you do not want to use uv, you can use `pip install -e .` instead.

## Usage

### Training

Run training with the default configuration:
```bash
lighter fit configs/train.yaml configs/models/primus.yaml configs/datasets/amos.yaml
```

Override parameters directly from the CLI:
```bash
lighter fit configs/train.yaml configs/models/primus.yaml configs/datasets/amos.yaml \
  trainer::max_epochs=50 \
  model::base_lr=0.0005 \
  data::batch_size=4
```

### Prediction

```bash
lighter predict configs/predict.yaml
```

### Configuration

Lighter uses YAML configs with powerful features:

- **Variable references**: `%vars::hidden_size` - reference shared variables
- **Cross-section references**: `%trainer::max_epochs` - reference other config sections
- **Python expressions**: `$int(%trainer::max_epochs * 0.03)` - compute values dynamically
- **Object instantiation**: `_target_: module.ClassName` - create objects from config

#### Config Structure

```
configs/
├── train.yaml           # Main training configuration
├── predict.yaml         # Inference configuration
├── dinotxt_stage.yaml   # Image-text alignment training
├── models/
│   ├── primus.yaml      # PRIMUS backbone
│   └── vit.yaml         # MONAI ViT backbone
└── datasets/
    ├── amos.yaml        # AMOS dataset
    └── idc_dump.yaml    # IDC dataset
```

Configs are composable - pass multiple files and they merge in order:
```bash
lighter fit base.yaml model.yaml dataset.yaml  # Later files override earlier ones
```

## Path Configuration

Each config file defines its paths in the `vars:` section at the top for easy customization:

| Config | Variable | Description |
|--------|----------|-------------|
| `train.yaml` | `experiments_dir` | Output directory for checkpoints and logs |
| `dinotxt_stage.yaml` | `experiments_dir` | Output directory for checkpoints and logs |
| `predict.yaml` | `amos_dataset` | Path to AMOS dataset |
| `datasets/amos.yaml` | `amos_dataset` | Path to AMOS dataset |
| `datasets/idc_dump.yaml` | `idc_dataset` | Path to IDC dataset |

Override paths from the CLI:
```bash
lighter fit configs/train.yaml configs/models/primus.yaml configs/datasets/amos.yaml \
    vars::experiments_dir=/your/output/path

lighter fit configs/train.yaml configs/models/primus.yaml configs/datasets/idc_dump.yaml \
    vars::idc_dataset=/your/idc/data/path
```

## Data Preparation

Create a JSON file in MONAI format:

```json
{
   "training": [
      {"image": "/path/to/image1.nii.gz"},
      {"image": "/path/to/image2.nii.gz"}
   ]
}
```

If you need more complex data loading (e.g., with labels for sampling), extend the JSON:

```json
{
   "training": [
      {"image": "/path/to/image.nii.gz", "label": "/path/to/label.nii.gz"}
   ]
}
```

Then update your dataset config or override from CLI:
```bash
lighter fit configs/train.yaml \
  "data::train_dataset::dataset::data=\$monai.auto3dseg.datafold_read('/path/to/dataset.json', basedir='/path/to/data', key='training')[0]"
```

## Project Structure

```
DINOv2-3D-Med/
├── __lighter__.py           # Lighter marker (enables project.* imports)
├── configs/                 # YAML configurations
├── models/                  # Model architectures
│   ├── meta_arch.py         # DINOv2 teacher-student architecture
│   └── backbones/           # PRIMUS, ViT, EVA backbones
├── training/                # Lightning modules
│   ├── dinov2_lightning_module.py
│   ├── dinotxt_lightning_module.py
│   └── data_module.py
├── transforms/              # Data augmentations
│   ├── dinov2_aug.py        # DINOv2 3D augmentations
│   └── blockmask.py         # Block masking for iBOT
├── losses/                  # Loss functions
│   └── dino.py              # DINOv2 + iBOT + KoLeo losses
└── utils/                   # Utilities
```

## References
- [Lighter](https://github.com/project-lighter/lighter)
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
