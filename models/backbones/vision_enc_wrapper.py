"""
Vision Blocks for DINO.txt implementation.
Adds two Eva blocks on top of the backbone for image-text alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dynamic_network_architectures.building_blocks.eva import Eva
from timm.layers import trunc_normal_
from enum import Enum


class SigmaPool(Enum):
    """
    Sigma pool type.
    """

    CLS = "cls"
    AVG = "avg"
    MAX = "max"
    CLS_MAX = "cls_max"
    CLS_AVG = "cls_avg"


class VisionEncoder_w_Blocks(nn.Module):
    """
    Enhanced Vision Encoder with two additional Eva blocks.
    Processes backbone features through additional Eva transformer layers.
    """

    def __init__(
        self,
        backbone: nn.Module,
        vision_block: nn.Module,
        sigma_pool: SigmaPool = SigmaPool.CLS_AVG,
    ):
        """
        Initialize enhanced vision encoder with Eva blocks.

        Args:
            backbone: Vision transformer backbone
            vision_block: Eva blocks for additional processing
            sigma_pool: Pooling strategy for features
        """
        super().__init__()

        self.backbone = backbone
        self.sigma_pool = sigma_pool
        self.vision_block = vision_block

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_patch_average(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute average of patch tokens, excluding CLS token.

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]

        Returns:
            Patch average [batch_size, embed_dim]
        """
        start_idx = 1  # Skip CLS token
        patch_tokens = x[:, start_idx:]
        return patch_tokens.mean(dim=1)

    def get_patch_max(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute max of patch tokens, excluding CLS token.

        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]

        Returns:
            Patch max [batch_size, embed_dim]
        """
        start_idx = 1  # Skip CLS token
        patch_tokens = x[:, start_idx:]
        return patch_tokens.max(dim=1)[0]

    def apply_sigma_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply the specified sigma pooling strategy.

        Args:
            features: Input features [batch_size, seq_len, embed_dim]

        Returns:
            Pooled features [batch_size, embed_dim or 2*embed_dim]
        """
        cls_token = features[:, 0]  # [batch_size, embed_dim]

        if self.sigma_pool == SigmaPool.CLS:
            return cls_token
        elif self.sigma_pool == SigmaPool.AVG:
            return self.get_patch_average(features)
        elif self.sigma_pool == SigmaPool.MAX:
            return self.get_patch_max(features)
        elif self.sigma_pool == SigmaPool.CLS_AVG:
            patch_avg = self.get_patch_average(features)
            return torch.cat([cls_token, patch_avg], dim=1)
        elif self.sigma_pool == SigmaPool.CLS_MAX:
            patch_max = self.get_patch_max(features)
            return torch.cat([cls_token, patch_max], dim=1)
        else:
            raise ValueError(f"Unknown sigma pooling strategy: {self.sigma_pool}")

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through enhanced vision encoder.

        Args:
            x: Input tensor [batch_size, channels, depth, height, width]
            mask: Optional mask for backbone

        Returns:
            Vision features [batch_size, embed_dim or 2*embed_dim]
        """
        # Forward through backbone
        if mask is not None:
            backbone_features = self.backbone(x, mask=mask)
        else:
            backbone_features = self.backbone(x)

        # Get rotary position embeddings if available
        rot_pos_embed = None
        if hasattr(self.vision_block, "rope") and self.vision_block.rope is not None:
            rot_pos_embed = self.vision_block.rope.get_embed()

        # Pass through additional Eva blocks
        features = backbone_features
        for blk in self.vision_block.blocks:
            if self.vision_block.grad_checkpointing and not torch.jit.is_scripting():
                from torch.utils.checkpoint import checkpoint

                features = checkpoint(blk, features, rope=rot_pos_embed)
            else:
                features = blk(features, rope=rot_pos_embed)

        # Apply normalization
        features = self.vision_block.norm(features)

        # Apply sigma pooling
        output_features = self.apply_sigma_pooling(features)
        return output_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward method for compatibility."""
        return self.forward(x)
