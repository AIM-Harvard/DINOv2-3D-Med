"""
Callback to log gram matrix of features to wandb.

This callback computes and logs a heatmap showing the similarity between
normalized features for each sample in a batch (after DDP aggregation).
"""

import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import matplotlib.pyplot as plt
import numpy as np
from typing import Any


class GramMatrixCallback(Callback):
    """
    Callback that logs a gram matrix visualization to wandb.
    
    The gram matrix shows the similarity between normalized features
    for all samples in a batch, computed after DDP aggregation.
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 100,
        feature_key: str = "teacher_cls_token",
    ):
        """
        Initialize the GramMatrixCallback.
        
        Args:
            log_every_n_steps: Log the gram matrix every N training steps
            feature_key: Key to extract features from model outputs 
                        (default: "teacher_cls_token")
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.feature_key = feature_key
    
    def on_train_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """
        Called when the training batch ends.
        
        Computes and logs the gram matrix of features if conditions are met.
        """
        # Only log at specified intervals
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        # Get the last batch outputs from the model
        # We need to do a forward pass to get features
        with torch.no_grad():
            views = batch[0]
            model_outputs = pl_module.model(views)
            features = model_outputs["pred"].get(self.feature_key, None)
            
            if features is None:
                return
            
            # Normalize features
            features_normalized = F.normalize(features, dim=-1, p=2)
            
            # Gather features across all DDP ranks
            if trainer.world_size > 1:
                # Gather from all ranks
                gathered_features = [
                    torch.zeros_like(features_normalized) 
                    for _ in range(trainer.world_size)
                ]
                torch.distributed.all_gather(gathered_features, features_normalized)
                features_normalized = torch.cat(gathered_features, dim=0)
            
            # Compute gram matrix (similarity matrix)
            # Shape: [batch_size, batch_size]
            gram_matrix = torch.matmul(
                features_normalized, features_normalized.T
            )
            
            # Log to wandb (only on rank 0)
            self._log_gram_matrix(trainer, gram_matrix)
    
    @rank_zero_only
    def _log_gram_matrix(self, trainer, gram_matrix: torch.Tensor) -> None:
        """
        Log the gram matrix as a heatmap to wandb.
        
        Args:
            trainer: PyTorch Lightning trainer
            gram_matrix: The computed gram matrix [N, N]
        """
        # Convert to numpy for plotting
        gram_np = gram_matrix.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        im = ax.imshow(gram_np, cmap='viridis', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sample Index')
        ax.set_title(f'Feature Gram Matrix (Step {trainer.global_step})')
        
        # Add grid for better readability
        ax.grid(False)
        
        # Log to wandb
        if trainer.logger is not None:
            try:
                # Try to log as wandb Image
                import wandb
                trainer.logger.experiment.log({
                    "gram_matrix": wandb.Image(fig),
                    "global_step": trainer.global_step,
                })
            except (ImportError, AttributeError):
                # Fallback: just close the figure
                pass
        
        # Close figure to free memory
        plt.close(fig)
