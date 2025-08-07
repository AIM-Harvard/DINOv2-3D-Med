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

from utils.imports import import_module_from_path


def modify_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract student backbone weights and convert vit -> eva naming"""
    modified_state_dict = {}
    
    for key, value in state_dict.items():
        if not key.startswith('model.student_backbone.'):
            continue
            
        # Remove prefix and convert vit -> eva
        new_key = key.replace('model.student_backbone.', '').replace('vit', 'eva')
        modified_state_dict[new_key] = value
    
    return modified_state_dict


def process_checkpoint(input_path: str, output_path: str, 
                      remove_cls_token: bool = True,
                      arch_class_name: str = 'PrimusM') -> None:
    """Process checkpoint file and save in nnUNet format"""
    
    # Import project module to handle checkpoint dependencies
    project_path = "/home/suraj/Repositories/DINOv2_3D"
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
    
    # Create final weights dictionary
    weights = {
        "network_weights": modified_state_dict,
        "nnssl_adaptation_plan": {
            'architecture_plans': {
                'arch_class_name': arch_class_name,
                'arch_kwargs': None,
                'arch_kwargs_requiring_import': None
            }
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} does not exist")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        process_checkpoint(
            input_path=args.input_path,
            output_path=args.output_path,
            remove_cls_token=not args.keep_cls_token,
            arch_class_name=args.arch_class_name
        )
        print("Checkpoint conversion completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()