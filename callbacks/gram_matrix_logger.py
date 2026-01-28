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
    
    Note: This callback performs an additional forward pass to extract features,
    which allows it to remain decoupled from the training step implementation.
    """
    
    def __init__(
        self,
        log_every_n_steps: int = 100,
        feature_key: str = "teacher_cls_token",
        max_samples: int = 128,
    ):
        """
        Initialize the GramMatrixCallback.
        
        Args:
            log_every_n_steps: Log the gram matrix every N training steps
            feature_key: Key to extract features from model outputs under the "pred" key
                        (default: "teacher_cls_token")
            max_samples: Maximum number of samples to include in gram matrix
                        to avoid memory issues (default: 128)
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.feature_key = feature_key
        self.max_samples = max_samples
    
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
        # Only log at specified intervals (skip step 0)
        if trainer.global_step == 0 or trainer.global_step % self.log_every_n_steps != 0:
            return
        
        # Get the last batch outputs from the model
        # We need to do a forward pass to get features
        features = None
        error_occurred = False
        
        try:
            # Validate batch structure
            if not isinstance(batch, (list, tuple)) or len(batch) < 1:
                if trainer.global_rank == 0:
                    trainer.logger.experiment.log({
                        "gram_matrix_error": f"Invalid batch structure at step {trainer.global_step}"
                    })
                error_occurred = True
            else:
                with torch.no_grad():
                    views = batch[0]
                    model_outputs = pl_module.model(views)
                    
                    # Safe nested dictionary access
                    pred_dict = model_outputs.get("pred", {})
                    features = pred_dict.get(self.feature_key, None)
                    
                    if features is None:
                        if trainer.global_rank == 0:
                            trainer.logger.experiment.log({
                                "gram_matrix_error": f"Feature key '{self.feature_key}' not found at step {trainer.global_step}"
                            })
                        error_occurred = True
                    else:
                        # Validate feature shape (should be [batch_size, feature_dim])
                        if features.dim() != 2:
                            if trainer.global_rank == 0:
                                trainer.logger.experiment.log({
                                    "gram_matrix_error": f"Expected 2D features, got shape {features.shape} at step {trainer.global_step}"
                                })
                            features = None
                            error_occurred = True
        except Exception as e:
            # Ensure all ranks know an error occurred
            error_occurred = True
            features = None
            if trainer.global_rank == 0:
                import logging
                logging.warning(f"Failed to extract features for gram matrix at step {trainer.global_step}: {e}")
        
        # Ensure all DDP ranks proceed together to avoid hangs
        # Only continue if features were successfully extracted
        if error_occurred or features is None:
            return
        
        try:
            # Normalize features
            features_normalized = F.normalize(features, dim=-1, p=2)
            
            # Gather features across all DDP ranks
            if trainer.world_size > 1:
                # All ranks must participate in gather
                gathered_features = [
                    torch.zeros_like(features_normalized) 
                    for _ in range(trainer.world_size)
                ]
                torch.distributed.all_gather(gathered_features, features_normalized)
                features_normalized = torch.cat(gathered_features, dim=0)
                
                # Clean up intermediate tensors to save memory
                del gathered_features
            
            # Limit number of samples to avoid memory issues
            if features_normalized.shape[0] > self.max_samples:
                # Sample uniformly
                indices = torch.linspace(
                    0, features_normalized.shape[0] - 1, self.max_samples, dtype=torch.long
                )
                features_normalized = features_normalized[indices]
            
            # Compute gram matrix (similarity matrix)
            # Shape: [N, N] where N <= max_samples
            gram_matrix = torch.matmul(
                features_normalized, features_normalized.T
            )
            
            # Log to wandb (only on rank 0)
            self._log_gram_matrix(trainer, gram_matrix)
            
        except Exception as e:
            # Gracefully handle errors to avoid interrupting training
            if trainer.global_rank == 0:
                import logging
                logging.warning(f"Failed to compute/log gram matrix at step {trainer.global_step}: {e}")
        finally:
            # Clean up to prevent memory leaks
            if features is not None:
                del features
            if 'features_normalized' in locals():
                del features_normalized
            if 'gram_matrix' in locals():
                del gram_matrix
    
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
        
        # Log to wandb if using WandbLogger
        if trainer.logger is not None:
            try:
                # Check if using WandbLogger
                from pytorch_lightning.loggers import WandbLogger
                if isinstance(trainer.logger, WandbLogger):
                    import wandb
                    trainer.logger.experiment.log({
                        "gram_matrix": wandb.Image(fig),
                    })
            except (ImportError, AttributeError) as e:
                # Fallback: log a warning if wandb is not available
                import logging
                logging.warning(f"Could not log gram matrix to wandb: {e}")
        
        # Close figure to free memory
        plt.close(fig)
