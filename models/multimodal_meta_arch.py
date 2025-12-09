"""
Multimodal DINOv2 Meta Architecture for image-text alignment.
This architecture only does text alignment on top of a pretrained DINOv2 model.
"""

import torch
from torch import nn
from typing import Optional, Dict, Any, Union, List

from project.losses.image_text_alignment import ImageTextAlignmentLoss


class DINOtxtMetaArchitecture(nn.Module):
    """
    Multimodal DINOv2 Meta Architecture for image-text alignment only.
    Takes a pretrained DINOv2 backbone and adds text alignment capabilities.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        """
        Initialize Multimodal DINOv2 Meta Architecture.

        Args:
            vision_encoder: Vision encoder
            text_encoder: Text encoder
            temperature: Temperature for contrastive loss
            learnable_temperature: Whether temperature is learnable
        """
        super().__init__()

        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        self.alignment_loss = ImageTextAlignmentLoss(
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )

    def forward_vision(
        self, images: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through vision encoder."""
        return self.vision_encoder(images, mask=mask)

    def forward_text(
        self, texts: Union[List[str], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Forward pass through text encoder with tokenization support."""
        if isinstance(texts, list):
            # Tokenize text inputs
            tokenized = self.text_encoder.tokenize(texts)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
        else:
            # Already tokenized
            input_ids = texts["input_ids"]
            attention_mask = texts.get("attention_mask", None)

        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        texts: Dict[str, torch.Tensor],
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal architecture.

        Args:
            images: Input images [batch_size, channels, depth, height, width]
            texts: Text inputs {"input_ids": tensor, "attention_mask": tensor}
            return_features: Whether to return extracted features

        Returns:
            Dictionary containing loss components and optionally features
        """
        # Extract image features
        image_features = self.forward_vision(images)

        # Extract text features
        text_features = self.forward_text(texts)

        return {"pred": (image_features, text_features)}

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature representations."""
        return self.forward_vision(images)

    def encode_text(
        self, texts: Union[List[str], Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Encode texts to feature representations."""
        return self.forward_text(texts)

    def get_similarity(
        self, images: torch.Tensor, texts: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute similarity between images and texts.

        Returns:
            Similarity matrix [batch_size, batch_size]
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # Compute cosine similarity
        similarity = torch.matmul(image_features, text_features.T)
        return similarity
