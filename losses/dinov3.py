"""
DINOv3 Loss implementation for 3D extension.
Based on the Gram anchoring technique from DINOv3 paper (arXiv:2508.10104).

This extends the DINOv2Loss with Gram anchoring for improved spatial coherence
and prevention of feature collapse during long training schedules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .dino import DINOv2Loss


class GramLoss(nn.Module):
    """
    Gram Loss for DINOv3-style Gram anchoring in 3D.

    This loss encourages the student's patch-level feature similarities (Gram matrix)
    to match those of a teacher network, helping preserve spatial coherence.
    """

    def __init__(
        self,
        normalize: bool = True,
        temperature: float = 1.0,
        loss_weight: float = 1.0,
    ):
        """
        Initialize Gram Loss.

        Args:
            normalize: Whether to normalize the Gram matrices
            temperature: Temperature for softmax normalization
            loss_weight: Weight for this loss component
        """
        super().__init__()
        self.normalize = normalize
        self.temperature = temperature
        self.loss_weight = loss_weight

    def compute_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix for a batch of features.

        Args:
            features: Feature tensor of shape (B, N, D) where:
                     B = batch size, N = number of patches, D = feature dimension

        Returns:
            Gram matrix of shape (B, N, N)
        """
        # Normalize features if requested
        if self.normalize:
            features = F.normalize(features, p=2, dim=-1)

        # Compute Gram matrix: G_ij = f_i · f_j
        gram = torch.bmm(features, features.transpose(1, 2))

        # Apply temperature scaling
        if self.temperature != 1.0:
            gram = gram / self.temperature

        return gram

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Gram loss between student and teacher features.

        Args:
            student_features: Student patch features (B, N, D)
            teacher_features: Teacher patch features (B, N, D)
            mask: Optional mask for valid patches (B, N)

        Returns:
            Gram loss value
        """
        # Ensure features have the same shape
        assert student_features.shape == teacher_features.shape, (
            f"Feature shapes must match: {student_features.shape} vs {teacher_features.shape}"
        )

        # Remove CLS token if present (assume it's the first token)
        if student_features.size(1) > 1:
            student_patches = student_features[:, 1:]  # Skip CLS token
            teacher_patches = teacher_features[:, 1:]  # Skip CLS token
            if mask is not None:
                mask = mask[:, 1:]  # Skip CLS token mask
        else:
            student_patches = student_features
            teacher_patches = teacher_features

        # Apply masking if provided - process each sample individually
        if mask is not None:
            batch_losses = []
            B = mask.shape[0]

            for b in range(B):
                valid_mask = mask[b]
                if valid_mask.sum() > 1:  # Need at least 2 patches for Gram matrix
                    student_valid = student_patches[b][valid_mask]
                    teacher_valid = teacher_patches[b][valid_mask]

                    # Compute Gram matrices for this sample
                    student_gram = self.compute_gram_matrix(student_valid.unsqueeze(0))
                    teacher_gram = self.compute_gram_matrix(teacher_valid.unsqueeze(0))

                    # Compute loss for this sample
                    sample_loss = F.mse_loss(student_gram, teacher_gram)
                    batch_losses.append(sample_loss)

            if batch_losses:
                gram_loss = torch.stack(batch_losses).mean()
            else:
                return torch.tensor(
                    0.0, device=student_features.device, requires_grad=True
                )
        else:
            # No masking - compute Gram matrices for all patches
            student_gram = self.compute_gram_matrix(student_patches)
            teacher_gram = self.compute_gram_matrix(teacher_patches)
            gram_loss = F.mse_loss(student_gram, teacher_gram)

        return self.loss_weight * gram_loss


class DINOv3Loss(DINOv2Loss):
    """
    DINOv3 Loss extending DINOv2Loss with Gram anchoring.

    Implements the complete DINOv3 training pipeline with stage-aware loss computation:
    - Stage 1 (Pretrain): DINO + iBOT + KoLeo
    - Stage 2 (Gram Anchor): DINO + iBOT + KoLeo + Gram
    - Stage 3 (High-res): DINO + iBOT + KoLeo + Gram
    """

    def __init__(
        self,
        student_temp: float = 0.1,
        teacher_temp_min: float = 0.04,
        teacher_temp_max: float = 0.07,
        teacher_temp_warmup_epochs: int = 30,
        center_momentum: float = 0.9,
        output_dim: int = 65536,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        gram_loss_weight: float = 0.1,
        gram_temperature: float = 1.0,
        max_steps: int = 1000,
        max_epochs: int = 500,
        training_stage: str = "pretrain",
    ):
        """
        Initialize DINOv3 Loss.

        Args:
            student_temp: Student temperature for DINO loss
            teacher_temp_min: Minimum teacher temperature
            teacher_temp_max: Maximum teacher temperature
            teacher_temp_warmup_epochs: Warmup epochs for teacher temperature
            center_momentum: Momentum for centering
            output_dim: Output dimension for projection heads
            ibot_loss_weight: Weight for iBOT loss
            koleo_loss_weight: Weight for KoLeo loss
            gram_loss_weight: Weight for Gram loss
            gram_temperature: Temperature for Gram matrix computation
            max_steps: Maximum training steps
            max_epochs: Maximum training epochs
            training_stage: Current training stage ("pretrain", "gram_anchor", "high_res")
        """
        # Initialize parent DINOv2Loss
        super().__init__(
            student_temp=student_temp,
            teacher_temp_min=teacher_temp_min,
            teacher_temp_max=teacher_temp_max,
            teacher_temp_warmup_epochs=teacher_temp_warmup_epochs,
            center_momentum=center_momentum,
            output_dim=output_dim,
            ibot_loss_weight=ibot_loss_weight,
            koleo_loss_weight=koleo_loss_weight,
            max_steps=max_steps,
            max_epochs=max_epochs,
        )

        # Additional DINOv3 parameters
        self.w_gram = gram_loss_weight
        self.training_stage = training_stage

        # Initialize Gram loss
        self.gram_loss_fn = GramLoss(
            normalize=True,
            temperature=gram_temperature,
            loss_weight=self.w_gram,
        )

    def set_training_stage(self, stage: str):
        """
        Set the current training stage.

        Args:
            stage: Training stage ("pretrain", "gram_anchor", "high_res")
        """
        assert stage in ["pretrain", "gram_anchor", "high_res"], (
            f"Invalid stage: {stage}. Must be one of: pretrain, gram_anchor, high_res"
        )
        self.training_stage = stage

    def should_use_gram_loss(self) -> bool:
        """Check if Gram loss should be used in current stage."""
        return self.training_stage in ["gram_anchor", "high_res"]

    def forward(self, input_dict, global_step: int = 0):
        """
        Compute DINOv3 loss based on current training stage.

        Args:
            input_dict: Dictionary containing model outputs
            global_step: Current training step

        Returns:
            Dictionary containing loss components
        """
        # Get base DINOv2 losses
        losses = super().forward(input_dict, global_step)

        # Add Gram loss if in appropriate stage
        gram_loss = 0
        if self.should_use_gram_loss():
            # Extract patch features for Gram loss
            teacher_patch_features = input_dict.get("teacher_patch_features", None)
            student_patch_features = input_dict.get("student_patch_features", None)
            mask = input_dict.get("mask", None)

            if (
                teacher_patch_features is not None
                and student_patch_features is not None
            ):
                gram_loss = self.gram_loss_fn(
                    student_features=student_patch_features,
                    teacher_features=teacher_patch_features,
                    mask=mask,
                )

                # Update total loss
                losses["total_loss"] = losses["total_loss"] + gram_loss

        # Add Gram loss to output dictionary
        losses["gram_loss"] = gram_loss
        losses["training_stage"] = self.training_stage

        return losses
