"""
Script to run 3D DINOv2 training or prediction using a YAML config.
- Uses PyTorch Lightning, MONAI, and custom project modules.
- Accepts config file and overrides via CLI (with fire).
"""

import os
from pathlib import Path
from monai.bundle import ConfigParser

import fire
from utils.imports import import_module_from_path
import torch
import monai

torch.serialization.safe_globals([monai.data.meta_tensor.MetaTensor])

REQUIRED_CONFIG_SECTIONS = ("trainer", "lightning_module", "data_module")


def _parse_config_files(config_file: str) -> list[str]:
    """Parse and validate a comma-separated config file argument."""
    if not isinstance(config_file, str) or not config_file.strip():
        raise ValueError(
            "Invalid --config_file value. Provide a comma-separated list of YAML files. "
            "Example: --config_file=./configs/train.yaml,./configs/models/primus.yaml"
        )

    parts = [part.strip() for part in config_file.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(
            "Invalid --config_file format: expected a comma-separated list without empty entries. "
            "Example: --config_file=./configs/train.yaml,./configs/models/primus.yaml"
        )

    return parts


def _validate_config_paths(config_paths: list[str]) -> None:
    """Ensure all referenced config files exist before parser work begins."""
    missing_paths = [path for path in config_paths if not Path(path).is_file()]
    if missing_paths:
        missing = ", ".join(missing_paths)
        raise FileNotFoundError(
            "Config file not found: "
            f"{missing}. Fix the --config_file paths and try again."
        )


def _validate_required_sections(parser: ConfigParser) -> None:
    """Verify required top-level sections are present in the loaded config."""
    missing_sections = []
    for key in REQUIRED_CONFIG_SECTIONS:
        try:
            section = parser.get(key)
            if section is None:
                missing_sections.append(key)
        except Exception:
            missing_sections.append(key)

    if missing_sections:
        raise ValueError(
            "Missing required config sections: "
            f"{', '.join(missing_sections)}. "
            "Add these sections to your config files before running."
        )


def _load_parsed_component(parser: ConfigParser, key: str, remediation: str):
    """Load instantiated component with actionable error context."""
    try:
        return parser.get_parsed_content(key)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Failed to initialize '{key}': {exc}. {remediation}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Failed to initialize '{key}': {exc}. {remediation}"
        ) from exc


def run(mode, config_file: str, **config_overrides):
    """
    Run training or prediction based on the mode parameter.

    Args:
        config_file (str): Comma-separated paths to configuration files (YAML)
        mode (str): Either "fit" or "predict"
        **config_overrides: Additional configuration overrides (key=value)
    """

    if mode not in ["fit", "predict"]:
        raise ValueError("Unsupported mode. Use 'fit' or 'predict'.")

    config_paths = _parse_config_files(config_file)
    _validate_config_paths(config_paths)

    parser = ConfigParser()
    try:
        parser.read_config(config_paths)
        parser.parse()
        parser.update(config_overrides)
    except FileNotFoundError as exc:
        missing_path = exc.filename or "unknown file"
        raise FileNotFoundError(
            f"Missing file referenced during config parse: {missing_path}. "
            "Check config paths and any referenced dataset/artifact files."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Failed to parse configuration files ({', '.join(config_paths)}): {exc}. "
            "Check YAML syntax and MONAI interpolation references."
        ) from exc

    _validate_required_sections(parser)

    try:
        project_path = parser.get("project")
    except Exception as exc:
        raise ValueError(
            "Missing required 'project' path in config. "
            "Set 'project' to your repository root path."
        ) from exc

    try:
        import_module_from_path("project", project_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to load project module from '{project_path}': {exc}. "
            "Ensure the project path exists and contains __init__.py."
        ) from exc

    trainer = _load_parsed_component(
        parser,
        "trainer",
        "Validate trainer configuration under the 'trainer' section.",
    )
    lightning_module = _load_parsed_component(
        parser,
        "lightning_module",
        "Validate model wiring under the 'lightning_module' section.",
    )
    data_module = _load_parsed_component(
        parser,
        "data_module",
        "Check dataset path and dataloader settings under the 'data_module' section.",
    )

    try:
        getattr(trainer, mode)(lightning_module, data_module)
    except Exception as exc:
        raise ValueError(
            f"Training launch failed during trainer.{mode}(...): {exc}. "
            "Check dataset path validity and runtime trainer/module compatibility."
        ) from exc


if __name__ == "__main__":
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run training or prediction
    fire.Fire(run)
