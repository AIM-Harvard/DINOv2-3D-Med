"""
Image-Text Alignment Loss for DINO.txt implementation.
Based on LiT (Locked-Image Tuning) strategy with modifications for DINOv2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ImageTextAlignmentLoss(nn.Module):
    """
    Image-Text Alignment Loss based on DINO.txt approach.
    Implements contrastive learning between image and text representations.
    """

    def __init__(
        self,
        logit_scale_init: float = 2.6592,  # ln(1/0.07)
        learnable_temperature: bool = True,
        reduction: str = "mean",
    ):
        """
        Initialize Image-Text Alignment Loss.

        Args:
            temperature: Initial temperature for contrastive loss
            logit_scale_init: Initial value for learnable logit scale
            learnable_temperature: Whether to make temperature learnable
            reduction: Reduction method for loss
        """
        super().__init__()

        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))
        else:
            self.register_buffer("logit_scale", torch.tensor(logit_scale_init))

        self.reduction = reduction

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> dict:
        """
        Compute image-text alignment loss.

        Args:
            image_features: Image representations [batch_size, embed_dim]
            text_features: Text representations [batch_size, embed_dim]
            mask: Optional mask for valid pairs [batch_size, batch_size]
            global_step: Global step for temperature scaling

        Returns:
            Dictionary containing loss components
        """
        batch_size = image_features.shape[0]
        device = image_features.device

        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute logits
        logit_scale = torch.clamp(self.logit_scale, max=100)  # Prevent overflow
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Create labels for contrastive learning
        labels = torch.arange(batch_size, device=device, dtype=torch.long)

        # Apply mask if provided
        if mask is not None:
            logits_per_image = logits_per_image.masked_fill(~mask, float("-inf"))
            logits_per_text = logits_per_text.masked_fill(~mask.T, float("-inf"))

        # Compute cross-entropy losses
        image_loss = F.cross_entropy(logits_per_image, labels, reduction=self.reduction)
        text_loss = F.cross_entropy(logits_per_text, labels, reduction=self.reduction)

        # Total contrastive loss
        contrastive_loss = (image_loss + text_loss) / 2
        return contrastive_loss
