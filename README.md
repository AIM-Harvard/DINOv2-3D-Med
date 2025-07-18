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
This project is provided under MIT License. See individual file headers for third-party code references. 
