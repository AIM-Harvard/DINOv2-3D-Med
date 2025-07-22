"""
DINOv2_3D model definition for 3D self-supervised learning.
Includes teacher/student ViT, DINO/iBOT heads, and update/cancel logic.
"""

import copy
import torch
from torch import nn

from lightly.models.modules import DINOProjectionHead
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import update_momentum
from transforms.blockmask import RandomBlockMask3D

# References:
# Thanks to the following repositories that provided the structure and necessary components for this implementation:
#     https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers
#     https://github.com/Project-MONAI/VISTA
#     https://github.com/lightly-ai/lightly/blob/master/benchmarks/imagenet/vitb16/dinov2.py


def freeze_eval_module(module: nn.Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


# Wrappers to ensure the param groups are named differently for the DINO and iBOT heads
class DINOHead(nn.Module):
    """A wrapper for the DINO projection head."""

    def __init__(self, dino_head: nn.Module) -> None:
        super().__init__()
        self._dino_head = dino_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._dino_head(x)

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        self._dino_head.cancel_last_layer_gradients(current_epoch)


class IBOTHead(nn.Module):
    """A wrapper for the iBOT projection head."""

    def __init__(self, ibot_head: nn.Module) -> None:
        super().__init__()
        self._ibot_head = ibot_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ibot_head(x)

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        self._ibot_head.cancel_last_layer_gradients(current_epoch)


class DINOv2_3D_Meta_Architecture(nn.Module):
    """
    3D DINOv2 model with teacher/student ViT backbones and DINO/iBOT heads.
    Supports block masking and teacher-student EMA updates.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        norm_last_layer: bool = False,
        ibot_separate_head: bool = True,
        freeze_last_layer: int = -1,
        projection_dim: int = 65536,
        backbone: nn.Module = None,
    ):
        """
        Initialize DINOv2_3D model.
        Args: see model config for details.
        """
        super().__init__()

        self.norm_last_layer = norm_last_layer
        self.ibot_separate_head = ibot_separate_head

        self.hidden_size = hidden_size
        self.teacher_backbone = backbone
        # Freeze teacher backbone
        freeze_eval_module(self.teacher_backbone)

        # DINO head
        teacher_dino_head = DINOProjectionHead(
            input_dim=self.hidden_size,
            output_dim=projection_dim,
            norm_last_layer=norm_last_layer,
        )
        self.teacher_dino_head = DINOHead(teacher_dino_head)
        freeze_eval_module(self.teacher_dino_head)

        # Student components
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        # Unfreeze student
        for param in self.student_backbone.parameters():
            param.requires_grad = True
        self.student_backbone.train()

        student_dino_head = DINOProjectionHead(
            input_dim=self.hidden_size,
            output_dim=projection_dim,
            freeze_last_layer=freeze_last_layer,
            norm_last_layer=norm_last_layer,
        )
        self.student_dino_head = DINOHead(student_dino_head)

        # iBOT heads - separate or shared
        if ibot_separate_head:
            teacher_ibot_head = DINOProjectionHead(
                input_dim=self.hidden_size,
                output_dim=projection_dim,
                freeze_last_layer=freeze_last_layer,
                norm_last_layer=norm_last_layer,
            )
            self.teacher_ibot_head = IBOTHead(teacher_ibot_head)
            freeze_eval_module(self.teacher_ibot_head)

            student_ibot_head = DINOProjectionHead(
                input_dim=self.hidden_size,
                output_dim=projection_dim,
                freeze_last_layer=freeze_last_layer,
                norm_last_layer=norm_last_layer,
            )
            self.student_ibot_head = IBOTHead(student_ibot_head)
        else:
            self.teacher_ibot_head = self.teacher_dino_head
            self.student_ibot_head = self.student_dino_head

    def forward_teacher(self, x, mask=None):
        features = self.teacher_backbone(x)
        cls_token = features[:, 0]
        features = features if mask is None else features[mask]
        return cls_token, features

    def forward_student(self, x, mask=None):
        features = self.student_backbone(x, mask=mask)
        cls_tokens = features[:, 0]
        features = features if mask is None else features[mask]
        return cls_tokens, features

    def update_teacher(self, global_step: int, max_steps: int) -> None:
        """Update teacher using EMA with cosine momentum schedule."""
        momentum = cosine_schedule(
            step=global_step, max_steps=max_steps, start_value=0.992, end_value=1.0
        )

        # Remove problematic device movement logic
        # In DDP, parameters should already be on correct devices
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_dino_head, self.teacher_dino_head, m=momentum)
        if self.ibot_separate_head:
            update_momentum(self.student_ibot_head, self.teacher_ibot_head, m=momentum)

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel gradients in the last layer during warmup."""
        self.student_dino_head.cancel_last_layer_gradients(current_epoch)
        if self.ibot_separate_head:
            self.student_ibot_head.cancel_last_layer_gradients(current_epoch)

    def forward(self, views: list[torch.Tensor]):
        """
        Forward pass for DINOv2 3D with block masking and multi-view augmentations.
        Args:
            views: List of augmented 3D tensors (global/local views)
        Returns:
            Dict of teacher/student outputs for loss computation
        """
        device = views[0].device

        global_views = torch.cat(views[:2])

        if len(views) > 2:
            local_views = torch.cat(views[2:])
        else:
            local_views = None

        # Masking
        B = len(global_views)
        sequence_length = self.student_backbone.sequence_length
        mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)
        H, W, D = self.student_backbone.grid_size
        assert H == W == D, "Patch size must be cubic for 3D input"
        assert H * W * D == sequence_length - 1, (
            f"Unexpected grid size {H * W * D} ({H}, {W}, {D}) does not match sequence length {sequence_length - 1}"
        )

        block_masker = RandomBlockMask3D(max_block_size=3)
        block_mask = block_masker(size=(B, D, H, W), device=device)
        mask[:, 1:] = block_mask.flatten(start_dim=1)

        # Teacher forward
        with torch.no_grad():
            teacher_cls_token, teacher_patch_tokens = self.forward_teacher(
                global_views, mask=mask
            )
            teacher_cls_token = self.teacher_dino_head(teacher_cls_token)
            teacher_patch_tokens = self.teacher_ibot_head(teacher_patch_tokens)

        # Student forward
        student_global_cls_token, student_global_patch_tokens = self.forward_student(
            global_views, mask=mask
        )
        student_global_cls_token = self.student_dino_head(student_global_cls_token)
        student_global_patch_tokens = self.student_ibot_head(
            student_global_patch_tokens
        )

        # Local views
        if local_views is not None:
            student_local_cls_token, _ = self.forward_student(local_views, mask=None)
            student_local_cls_token = self.student_dino_head(student_local_cls_token)
            student_cls_token = torch.cat(
                [student_global_cls_token, student_local_cls_token], dim=0
            )
        else:
            student_cls_token = student_global_cls_token

        out = {
            "teacher_cls_token": teacher_cls_token,
            "student_cls_token": student_cls_token,
            "teacher_patch_tokens": teacher_patch_tokens,
            "student_patch_tokens": student_global_patch_tokens,
            "student_glob_cls_token": student_global_cls_token,
            "mask": block_mask,
            "n_local_views": torch.tensor(len(views) - 2, device=device),
        }

        return {"pred": out}

    def encode(self, x: torch.Tensor):
        """
        Simple encoding method that returns the raw CLS token features.
        Useful for feature extraction or as input to downstream tasks.
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        Returns:
            CLS token features of shape (B, hidden_size)
        """
        backbone_features = self.student_backbone(x, mask=None)
        return backbone_features[:, 0]  # Return CLS token only
