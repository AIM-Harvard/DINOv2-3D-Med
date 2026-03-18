#!/usr/bin/env python3
"""
Checkpoint Conversion Script
Converts PyTorch Lightning checkpoints to nnUNet format
"""

import argparse
import torch
import os
import sys
from typing import Dict, Any

# Ensure project root is on sys.path so 'utils' and project modules resolve
# when script is run directly (e.g. python scripts/utility/export_ckpt_to_nnunet.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utils.imports import import_module_from_path


def modify_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract student backbone weights and convert vit -> eva naming"""
    # RoPE buffers are computed on-the-fly in the downstream network and must not
    # be included in the exported weights (causes strict load_state_dict failures).
    SKIP_SUFFIXES = ('rope.periods',)

    modified_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith('model.student_backbone.'):
            continue
        new_key = key.replace('model.student_backbone.', '').replace('vit', 'eva')
        if any(new_key.endswith(s) for s in SKIP_SUFFIXES):
            continue
        modified_state_dict[new_key] = value

    return modified_state_dict


def process_checkpoint(input_path: str, output_path: str,
                      remove_cls_token: bool = True,
                      arch_class_name: str = 'PrimusM',
                      pretrain_spacing: list = None,
                      pretrain_patch_size: list = None,
                      pretrain_num_input_channels: int = 1,
                      downstream_patch_size: list = None) -> None:
    """Process checkpoint file and save in nnUNet format"""

    if pretrain_spacing is None:
        pretrain_spacing = [1.0, 1.0, 1.0]
    if pretrain_patch_size is None:
        pretrain_patch_size = [96, 96, 96]
    if downstream_patch_size is None:
        downstream_patch_size = pretrain_patch_size

    # Import project module to handle checkpoint dependencies
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    import_module_from_path("project", project_path)

    print(f"Loading checkpoint from: {input_path}")
    try:
        ckpt = torch.load(input_path, weights_only=False, map_location='cpu')
    except ModuleNotFoundError as e:
        print(f"Warning: Missing dependency {e}. Attempting to load with weights_only=True")
        ckpt = torch.load(input_path, weights_only=True, map_location='cpu')

    # Extract and modify state dict
    modified_state_dict = modify_state_dict(ckpt['state_dict'])

    # Remove CLS token from positional embeddings if requested
    if remove_cls_token and 'eva.pos_embed' in modified_state_dict:
        original_shape = modified_state_dict['eva.pos_embed'].shape
        modified_state_dict['eva.pos_embed'] = modified_state_dict['eva.pos_embed'][:, 1:, :]
        new_shape = modified_state_dict['eva.pos_embed'].shape
        print(f"Removed CLS token from positional embeddings: {original_shape} -> {new_shape}")
    elif not remove_cls_token and 'eva.pos_embed' in modified_state_dict:
        print(f"Keeping CLS token in positional embeddings: {modified_state_dict['eva.pos_embed'].shape}")
    else:
        print("No positional embeddings found in checkpoint")

    # Build full nnssl_adaptation_plan as required by TaWald nnUNet fork
    weights = {
        "network_weights": modified_state_dict,
        "nnssl_adaptation_plan": {
            "pretrain_plan": {
                "configurations": {
                    "3d_fullres": {
                        "spacing": pretrain_spacing,
                        "patch_size": pretrain_patch_size,
                    }
                }
            },
            "architecture_plans": {
                "arch_class_name": arch_class_name,
                "arch_kwargs": None,
                "arch_kwargs_requiring_import": None,
            },
            "key_to_encoder": "eva",
            "key_to_stem": "down_projection",
            "keys_to_in_proj": ["down_projection.proj"],
            "key_to_lpe": "eva.pos_embed",
            "pretrain_num_input_channels": pretrain_num_input_channels,
            "recommended_downstream_patchsize": downstream_patch_size,
        }
    }
    
    print(f"Saving processed checkpoint to: {output_path}")
    torch.save(weights, output_path)
    print(f"Successfully saved {len(modified_state_dict)} parameters")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Lightning checkpoints to nnUNet format"
    )
    
    parser.add_argument("input_path", help="Path to input checkpoint file (.ckpt)")
    parser.add_argument("output_path", help="Path to output file (.pt or .pth)")
    parser.add_argument("--arch-class-name", default="PrimusM",
                       help="Architecture class name (default: PrimusM)")
    parser.add_argument("--keep-cls-token", action="store_true",
                       help="Keep CLS token in positional embeddings")
    parser.add_argument("--pretrain-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                       metavar=("Z", "Y", "X"),
                       help="Voxel spacing used during pretraining in mm (default: 1.0 1.0 1.0)")
    parser.add_argument("--pretrain-patch-size", type=int, nargs=3, default=[96, 96, 96],
                       metavar=("Z", "Y", "X"),
                       help="Input patch size used during pretraining (default: 96 96 96)")
    parser.add_argument("--downstream-patch-size", type=int, nargs=3, default=None,
                       metavar=("Z", "Y", "X"),
                       help="Recommended downstream patch size (default: same as pretrain-patch-size)")
    parser.add_argument("--pretrain-num-input-channels", type=int, default=1,
                       help="Number of input channels during pretraining (default: 1)")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} does not exist")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    try:
        downstream_patch_size = args.downstream_patch_size or args.pretrain_patch_size
        process_checkpoint(
            input_path=args.input_path,
            output_path=args.output_path,
            remove_cls_token=not args.keep_cls_token,
            arch_class_name=args.arch_class_name,
            pretrain_spacing=args.pretrain_spacing,
            pretrain_patch_size=args.pretrain_patch_size,
            pretrain_num_input_channels=args.pretrain_num_input_channels,
            downstream_patch_size=downstream_patch_size,
        )
        print("Checkpoint conversion completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()