# DINOv2-3D: Self-Supervised 3D Vision Transformer Pretraining

This repository provides a 3D implementation of DINOv2 for self-supervised pretraining on volumetric (3D) medical images using Lightly, MONAI, and Pytorch Lightning!

## Features
- 3D Vision Transformer (ViT) backbone (MONAI-based)
- DINOv2-style self-supervised learning with teacher-student models
- Block masking for 3D volumes
- Flexible 3D augmentations (global/local views)
- PyTorch Lightning training loop
- YAML-based experiment configuration
- Custom DINO loss implementation
- Utility functions for seamless imports

## Directory Structure
```
.
├── scripts/
│   └── train_dinov2.py         # Main training script
├── models/
│   ├── model.py                # DINOv2_3D model implementation
│   ├── lightning_module.py     # LightningModule trainer
│   ├── masked_vit_wrapper.py   # Masked ViT wrapper
│   └── __init__.py
├── transforms/
│   ├── dinov2_aug.py           # 3D augmentations
│   ├── blockmask.py            # 3D block masking
│   └── random_resized_crop.py  # 3D random resized crop
├── losses/
│   ├── dino.py                 # DINO-specific loss functions
│   └── __init__.py
├── utils/
│   ├── imports.py              # Import utilities
│   └── __init__.py
├── configs/
│   ├── train.yaml              # Training configuration
│   └── predict.yaml            # Prediction configuration
├── requirements.txt
└── README.md
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd DINOv2_3D
   ```
2. Create a virtual environment (recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```


## Usage
### Training
Run the training script with the default training config:
```bash
python -m scripts.train --config_file configs/train.yaml
```

Override config parameters via CLI using MONAI and fire syntax:
```bash
python -m scripts.train --config_file configs/train.yaml --num_workers=4 --trainer#max_epochs=100
```

### Configuration
- All experiment settings (model, trainer, data) are defined in YAML configs.
- `configs/train.yaml`: Main training configuration with complete setup
- `configs/predict.yaml`: Configuration for inference/prediction tasks
- The configs support advanced features like:
  - Multi-GPU training with DDP
  - Mixed precision training (16-bit)
  - WandB logging
  - Flexible data augmentation pipelines
  - MONAI-based medical image preprocessing

### Key Configuration Parameters
- **Model**: ViT-Small/4 architecture by default (768 hidden size, 8 layers, 12 heads)
- **Training**: 500 epochs, gradient clipping, layer decay, temperature scheduling
- **Data**: 128³ volumes with 16³ patches, supports AMOS22 dataset format
- **Augmentation**: Global views (128³) + local views (64³) with scale variations

## Model Architecture
The implementation includes:
- **DINOv2_3D_LightningModule**: Main training module with teacher-student architecture
- **Masked ViT Wrapper**: Supports block masking for self-supervised learning
- **3D Augmentations**: Specialized transforms for volumetric medical data
- **DINO Loss**: Custom implementation of DINO loss with temperature scheduling

## References
- [Lightly](https://github.com/lightly-ai/lightly)
- [DINOv2 (Facebook Research)](https://github.com/facebookresearch/dinov2)
- [MONAI (Medical Open Network for AI)](https://github.com/Project-MONAI/MONAI)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)


## License
This project is for research purposes. See individual file headers for third-party code references. 
