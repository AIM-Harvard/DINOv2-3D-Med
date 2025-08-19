"""
Text Encoder for DINO.txt implementation using pretrained HuggingFace models.
Supports CLIP, BERT, and other transformer-based text encoders.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import (
    AutoModel, 
    AutoTokenizer, 
)


class TextEncoder(nn.Module):
    """
    Wrapper for HuggingFace pretrained text encoders.
    Supports various text models with optional projection heads.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 512,
        freeze_encoder: bool = False,
        use_projection: bool = True,
        pooling_strategy: str = "cls",  # "cls", "mean", "eos"
        max_length: int = 77,
    ):
        """
        Initialize HuggingFace text encoder.
        
        Args:
            model_name: HuggingFace model identifier
            output_dim: Output feature dimension
            freeze_encoder: Whether to freeze the base encoder
            use_projection: Whether to add projection head
            pooling_strategy: How to pool sequence outputs ("cls", "mean", "eos")
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.encoder.config.hidden_size
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Optional projection head
        if use_projection and self.embed_dim != output_dim:
            self.projection = nn.Linear(self.embed_dim, output_dim)
        else:
            self.projection = None
    
    def tokenize(
        self, 
        texts: list, 
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Tokenize input texts."""
        return self.tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length
        )
    
    def pool_sequence_outputs(
        self, 
        sequence_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool sequence outputs to get sentence representation."""
        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return sequence_output[:, 0]
        elif self.pooling_strategy == "mean":
            # Mean pooling over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size())
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == "eos":
            # Use last valid token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(sequence_output.size(0))
            return sequence_output[batch_indices, seq_lengths]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through text encoder.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return dict with additional info
            
        Returns:
            Text features [batch_size, output_dim]
        """
        # Get encoder outputs
        if "clip" in self.model_name.lower():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # CLIP already pools to sentence representation
            text_features = outputs.pooler_output
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Pool sequence outputs
            text_features = self.pool_sequence_outputs(
                outputs.last_hidden_state, attention_mask
            )
        
        # Apply projection if available
        if self.projection is not None:
            text_features = self.projection(text_features)
        
        if return_dict:
            return {
                'text_features': text_features,
                'last_hidden_state': outputs.last_hidden_state,
                'pooler_output': getattr(outputs, 'pooler_output', None)
            }
        
        return text_features

