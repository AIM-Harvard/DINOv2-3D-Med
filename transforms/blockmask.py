"""
RandomBlockMask3D: 3D block masking for self-supervised learning (DINOv2-style).
Supports simple and advanced block masking for volumetric data.
"""

from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor


class RandomBlockMask3D(nn.Module):
    """
    A class for generating 3D random block masks for self-supervised learning.
    Supports both simple and advanced (multi-block, target ratio) masking.
    """

    def __init__(
        self,
        ratio_min: float = 0.1,
        ratio_max: float = 0.5,
        min_block_size: int = 1,
        max_block_size: Optional[int] = None,
        aspect_ratio_range: Tuple[float, float] = (0.75, 1.25),
        num_masking_patches: Optional[int] = None,
        min_num_patches: int = 4,
        max_num_patches: Optional[int] = None,
    ):
        """
        Initialize the 3D random block masking.
        Args: see class docstring for details.
        """
        super().__init__()
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.aspect_ratio_range = aspect_ratio_range
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.mode = "advanced"

    def forward(
        self,
        size: Tuple[int, int, int, int],
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        """
        Generate 3D random block masks for a batch.
        Args:
            size: Tensor size as (batch_size, depth, height, width)
            device: Device to create tensors on
        Returns:
            Boolean tensor of shape (batch_size, depth, height, width) where True indicates masked patches
        """
        if self.mode == "advanced":
            # Use advanced block masking strategy
            return self.advanced_block_mask(size, mask_ratio=0.75, device=device)
        else:
            # Default to simple block masking
            return self.simple_block_mask(size, device)

    def _calculate_block_sizes(self, D, H, W):
        """Calculate block sizes for all dimensions efficiently."""
        ratio = torch.empty(1).uniform_(self.ratio_min, self.ratio_max).item()
        sizes = [max(self.min_block_size, int(dim * ratio)) for dim in [D, H, W]]

        # Apply constraints
        if self.max_block_size is not None:
            sizes = [min(size, self.max_block_size) for size in sizes]

        # Ensure sizes don't exceed grid dimensions
        return [min(size, dim) for size, dim in zip(sizes, [D, H, W])]

    def simple_block_mask(
        self,
        size: Tuple[int, int, int, int],
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        """Generate simple, single, 3D block masks using ratio-based sizing."""
        B, D, H, W = size

        # Calculate block sizes efficiently
        block_sizes = self._calculate_block_sizes(D, H, W)

        # Calculate valid placement ranges and generate random starts
        valid_ranges = [
            max(1, dim - block_size + 1)
            for dim, block_size in zip([D, H, W], block_sizes)
        ]
        starts = [
            torch.randint(0, valid_range, (B,), device=device)
            for valid_range in valid_ranges
        ]

        # Create masks
        mask = torch.zeros(size, dtype=torch.bool, device=device)
        for i in range(B):
            slices = [
                slice(start[i], start[i] + block_size)
                for start, block_size in zip(starts, block_sizes)
            ]
            mask[i, slices[0], slices[1], slices[2]] = True

        return mask

    def advanced_block_mask(
        self,
        size: Tuple[int, int, int, int],
        mask_ratio: float = 0.75,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tensor:
        """
        Generate advanced 3D block masks with target masking ratio.

        This method creates multiple blocks per image (volume patches positions) to reach a target
        masking ratio, similar to DINOv2's strategy but adapted for 3D.

        Args:
            size: Tensor size as (batch_size, depth, height, width)
            mask_ratio: Target ratio of patches to mask
            device: Device to create tensors on

        Returns:
            Boolean tensor where True indicates masked patches
        """
        B, D, H, W = size
        total_patches = D * H * W
        target_masked = int(total_patches * mask_ratio)

        masks = []
        for batch_idx in range(B):
            mask = torch.zeros((D, H, W), dtype=torch.bool, device=device)
            current_masked = 0

            for _ in range(50):  # Max attempts to prevent infinite loops
                if current_masked >= target_masked:
                    break

                remaining = target_masked - current_masked
                target_patches = self._get_target_patches(remaining)

                # Calculate 3D block dimensions from target patches
                block_dims = self._calculate_3d_block_dims(target_patches)
                block_dims = self._apply_size_constraints(block_dims, D, H, W)

                # Try to place block
                if self._try_place_block(mask, block_dims, D, H, W):
                    new_masked = block_dims[0] * block_dims[1] * block_dims[2]
                    current_masked += new_masked

            masks.append(mask)

        return torch.stack(masks)

    def _get_target_patches(self, remaining):
        """Get target number of patches for next block."""
        if self.num_masking_patches is not None:
            return min(remaining, self.num_masking_patches)
        return max(
            self.min_num_patches, min(remaining, self.max_num_patches or remaining)
        )

    def _calculate_3d_block_dims(self, target_patches):
        """Calculate 3D block dimensions from target patch count."""
        import random

        base_size = max(1, int(target_patches ** (1 / 3)))
        aspect_ratio = random.uniform(*self.aspect_ratio_range)

        return [
            max(1, target_patches // (base_size * base_size)),  # depth
            max(1, int(base_size * aspect_ratio)),  # height
            max(1, int(base_size / aspect_ratio)),  # width
        ]

    def _apply_size_constraints(self, block_dims, D, H, W):
        """Apply size constraints to block dimensions."""
        dims = [D, H, W]

        # Apply minimum size constraint
        block_dims = [max(self.min_block_size, dim) for dim in block_dims]

        # Apply maximum size constraint
        if self.max_block_size is not None:
            block_dims = [min(dim, self.max_block_size) for dim in block_dims]

        # Ensure blocks fit within volume
        return [min(block_dim, vol_dim) for block_dim, vol_dim in zip(block_dims, dims)]

    def _try_place_block(self, mask, block_dims, D, H, W):
        """Try to place a block in the mask. Returns True if successful."""
        import random

        # Calculate valid placement ranges
        valid_ranges = [
            max(1, vol_dim - block_dim + 1)
            for vol_dim, block_dim in zip([D, H, W], block_dims)
        ]

        if any(valid_range <= 0 for valid_range in valid_ranges):
            return False

        # Random placement
        starts = [random.randint(0, valid_range - 1) for valid_range in valid_ranges]
        ends = [start + block_dim for start, block_dim in zip(starts, block_dims)]

        # Check if placement adds new masked patches
        slices = [slice(start, end) for start, end in zip(starts, ends)]
        block_region = mask[slices[0], slices[1], slices[2]]

        if (~block_region).sum().item() > 0:
            mask[slices[0], slices[1], slices[2]] = True
            return True

        return False
