"""
DINOv2Augmentation3D: 3D data augmentation pipeline for DINOv2 self-supervised learning.
Includes global/local views, affine, histogram, smoothing, and cropping.
"""

from typing import List, Optional, Union, Tuple
import torch
from torchvision.transforms import Compose
from monai.transforms import (
    EnsureChannelFirst,
    RandAffine,
    RandHistogramShift,
    RandGaussianSmooth,
    SpatialPad,
)
from .random_resized_crop import RandomResizedCrop3D
from copy import deepcopy
import torch
import torch.nn as nn


class DINOv2Augmentation3D(nn.Module):
    """
    3D data augmentation for DINOv2: global and local views, affine, histogram, smoothing, cropping.
    """

    def __init__(
        self,
        global_view_scale: Optional[List[float]] = None,
        global_view_size: Union[int, Tuple[int, int, int]] = 48,
        local_view_scale: Optional[List[float]] = None,
        local_view_size: Union[int, Tuple[int, int, int]] = 24,
        num_local_views: int = 2,
    ):
        """
        Initialize 3D DINOv2 augmentation pipeline.
        Args:
            global_view_scale: Scale range for global crops
            global_view_size: Output size for global crops
            local_view_scale: Scale range for local crops
            local_view_size: Output size for local crops
            num_local_views: Number of local views
        """
        super().__init__()
        if global_view_scale is None:
            global_view_scale = [0.3, 1.0]
        if local_view_scale is None:
            local_view_scale = [0.1, 0.3]
        self.global_view_scale = global_view_scale
        self.global_view_size = global_view_size
        self.local_view_scale = local_view_scale
        self.local_view_size = local_view_size
        self.num_local_views = num_local_views

        if self.num_local_views == 0 and min(self.global_view_scale) > 0.4:
            # TODO: implement a warning
            message = f"MultiView is disabled because num_local_views is set to {num_local_views} and \
            global_view_scale lower bound is > 0.4, i.e. {global_view_scale}. Setting global_view_scale \
            lower bound to 0.2 to ensure local learning is possible."
            self.global_view_scale[0] = (
                sum(self.local_view_scale) / 2 if self.local_view_scale else 0.25
            )

        self.global_aug = Compose(
            [
                RandAffine(
                    prob=0.5,
                    rotate_range=(22 / 7) / 180 * 10,
                    shear_range=0.1,
                    padding_mode="zeros",
                ),
                RandomResizedCrop3D(
                    prob=1, size=self.global_view_size, scale=self.global_view_scale
                ),
                RandHistogramShift(prob=0.5),
                RandGaussianSmooth(prob=0.5),
                SpatialPad(spatial_size=self.global_view_size),
            ]
        )
        if self.num_local_views > 0:
            self.local_aug = Compose(
                [
                    RandAffine(
                        prob=0.5,
                        rotate_range=(22 / 7) / 180 * 10,
                        shear_range=0.1,
                        padding_mode="zeros",
                    ),
                    RandomResizedCrop3D(
                        prob=1, size=self.local_view_size, scale=self.local_view_scale
                    ),
                    RandHistogramShift(prob=0.5),
                    RandGaussianSmooth(prob=0.5),
                    SpatialPad(spatial_size=self.local_view_size),
                ]
            )
        else:
            self.local_aug = None
            # TODO: implement a warning ("No local views will be generated as num_local_views is set to 0. "
            # "Consider setting num_local_views > 0 for better performance.")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply global and local augmentations to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W).
        Returns:
            List[torch.Tensor]: List containing N augmented tensors for global (2) and local (N-2) views.
        """
        views = []
        global_views = [self.global_aug(deepcopy(x)) for _ in range(2)]
        views.extend(global_views)
        if self.local_aug is not None:
            local_views = [self.local_aug(x) for _ in range(self.num_local_views)]
            views.extend(local_views)
        return views
