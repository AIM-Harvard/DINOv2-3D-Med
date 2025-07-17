"""
Script to run 3D DINOv2 training or prediction using a YAML config.
- Uses PyTorch Lightning, MONAI, and custom project modules.
- Accepts config file and overrides via CLI (with fire).
"""

import os
from torch.utils.data import DataLoader
from monai.bundle import ConfigParser

import fire
from utils.imports import import_module_from_path
import torch
import monai

torch.serialization.safe_globals([monai.data.meta_tensor.MetaTensor])


def run(mode, config_file: str, **config_overrides):
    """
    Run training or prediction based on the mode parameter.

    Args:
        config_file (str): Path to the configuration file (YAML)
        mode (str): Either "train" or "predict"
        **config_overrides: Additional configuration overrides (key=value)
    """

    assert mode in ["train", "predict"], "Invalid mode"

    parser = ConfigParser()
    parser.read_config(config_file)
    parser.parse()

    parser.update(config_overrides)

    project_path = parser.get("project")
    import_module_from_path("project", project_path)

    trainer = parser.get_parsed_content("trainer")
    model = parser.get_parsed_content("system#model")

    if mode == "train":
        # Training mode
        train_dataset = parser.get_parsed_content("system#datasets#train")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=model.batch_size_per_device,
            shuffle=True,
            num_workers=parser.get("num_workers"),
            pin_memory=parser.get("pin_memory"),
            drop_last=parser.get("drop_last"),
        )

        trainer.fit(model, train_dataloader)

    elif mode == "predict":
        # Prediction mode
        predict_dataset = parser.get_parsed_content("system#datasets#predict")

        predict_dataloader = DataLoader(
            predict_dataset,
            batch_size=model.batch_size_per_device,
            shuffle=False,  # Don't shuffle for prediction
            num_workers=parser.get("num_workers"),
            pin_memory=parser.get("pin_memory"),
            drop_last=False,  # Don't drop last batch for prediction
        )

        trainer.predict(model, predict_dataloader)

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'train' or 'predict'")


if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run training or prediction
    fire.Fire(run)
