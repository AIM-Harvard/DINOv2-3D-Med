"""
Callback to log gram matrix of features to wandb.

This callback computes and logs a heatmap showing the similarity between
normalized features for each sample in a batch (after DDP aggregation).
"""

import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import torch.distributed as dist
import matplotlib.pyplot as plt
from typing import Any
import logging


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
        saturation_offdiag_threshold: float = 0.995,
        saturation_offdiag_std_threshold: float = 0.01,
        saturation_sample_var_threshold: float = 1e-4,
        auto_fallback_to_backbone_on_saturation: bool = True,
    ):
        """
        Initialize the GramMatrixCallback.
        
        Args:
            log_every_n_steps: Log the gram matrix every N training steps
            feature_key: Key to extract features from model outputs under the "pred" key
                        (default: "teacher_cls_token")
            max_samples: Maximum number of samples to include in gram matrix
                        to avoid memory issues (default: 128)
            saturation_offdiag_threshold: Threshold used to flag near-constant off-diagonal
                        cosine similarities.
            saturation_offdiag_std_threshold: Maximum off-diagonal std to treat as
                        suspiciously constant.
            saturation_sample_var_threshold: Maximum sample variance threshold to treat
                        representation as collapsed.
            auto_fallback_to_backbone_on_saturation: If true, fallback to backbone
                        feature keys when projected-head features look framework-saturated.
        """
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be > 0")
        self.log_every_n_steps = log_every_n_steps
        self.feature_key = feature_key
        self.max_samples = max_samples
        self.saturation_offdiag_threshold = saturation_offdiag_threshold
        self.saturation_offdiag_std_threshold = saturation_offdiag_std_threshold
        self.saturation_sample_var_threshold = saturation_sample_var_threshold
        self.auto_fallback_to_backbone_on_saturation = auto_fallback_to_backbone_on_saturation
        self._warned_no_wandb = False
        self._warned_wandb_disabled = False
    
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
        fallback_features = None
        error_occurred = False
        
        try:
            # Validate batch structure
            if not isinstance(batch, (list, tuple)) or len(batch) < 1:
                self._report_extraction_issue(
                    trainer, f"Invalid batch structure at step {trainer.global_step}"
                )
                error_occurred = True
            else:
                with torch.no_grad():
                    views = batch[0]
                    model_outputs = pl_module.model(views)
                    
                    # Safe nested dictionary access
                    pred_dict = model_outputs.get("pred", {})
                    features = pred_dict.get(self.feature_key, None)
                    fallback_key = self._fallback_feature_key_for(self.feature_key)
                    if fallback_key is not None:
                        fallback_features = pred_dict.get(fallback_key, None)
                    
                    if features is None:
                        self._report_extraction_issue(
                            trainer,
                            f"Feature key '{self.feature_key}' not found at step {trainer.global_step}",
                        )
                        error_occurred = True
                    else:
                        # Validate feature shape (should be [batch_size, feature_dim])
                        if features.dim() != 2:
                            self._report_extraction_issue(
                                trainer,
                                f"Expected 2D features, got shape {features.shape} at step {trainer.global_step}",
                            )
                            features = None
                            error_occurred = True
                        elif (
                            fallback_features is not None
                            and hasattr(fallback_features, "dim")
                            and fallback_features.dim() != 2
                        ):
                            fallback_features = None
        except Exception as e:
            # Ensure all ranks know an error occurred
            error_occurred = True
            features = None
            if trainer.global_rank == 0:
                logging.warning(f"Failed to extract features for gram matrix at step {trainer.global_step}: {e}")
        
        # Ensure all DDP ranks proceed together to avoid hangs
        # Only continue if features were successfully extracted
        has_features = 0 if error_occurred or features is None else 1
        device = features.device if features is not None else pl_module.device
        has_features_tensor = torch.tensor(has_features, device=device, dtype=torch.int)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(has_features_tensor, op=dist.ReduceOp.MIN)
        if has_features_tensor.item() == 0:
            return
        
        try:
            stage_metrics = self._build_stage_metrics(
                stage_name="feature_extraction", features=features
            )
            primary_normalized = self._prepare_features_for_gram(features, trainer)
            stage_metrics.update(
                self._build_stage_metrics(
                    stage_name="normalization", features=F.normalize(features, dim=-1, p=2)
                )
            )
            stage_metrics.update(
                self._build_stage_metrics(
                    stage_name="sampling",
                    features=self._sample_features_for_stage(
                        F.normalize(features, dim=-1, p=2),
                        trainer=trainer,
                    ),
                )
            )
            stage_metrics.update(
                self._build_stage_metrics(
                    stage_name="gather", features=primary_normalized
                )
            )
            gram_matrix = torch.matmul(primary_normalized, primary_normalized.T)
            primary_stats = self._compute_saturation_stats(gram_matrix, primary_normalized)

            fallback_used = 0.0
            root_cause_framework = 0.0
            root_cause_model = 0.0
            root_cause_unknown = 0.0

            if self._is_saturated(primary_stats):
                fallback_key = self._fallback_feature_key_for(self.feature_key)
                if (
                    self.auto_fallback_to_backbone_on_saturation
                    and fallback_key is not None
                    and fallback_features is not None
                ):
                    fallback_normalized = self._prepare_features_for_gram(
                        fallback_features, trainer
                    )
                    fallback_gram = torch.matmul(
                        fallback_normalized, fallback_normalized.T
                    )
                    fallback_stats = self._compute_saturation_stats(
                        fallback_gram, fallback_normalized
                    )

                    if not self._is_saturated(fallback_stats):
                        gram_matrix = fallback_gram
                        fallback_used = 1.0
                        root_cause_framework = 1.0
                        if trainer.global_rank == 0:
                            logging.warning(
                                "Gram-matrix saturation detected on '%s'; using '%s' fallback.",
                                self.feature_key,
                                fallback_key,
                            )
                    else:
                        root_cause_model = 1.0
                        if trainer.global_rank == 0:
                            logging.warning(
                                "Gram-matrix saturation persists on both '%s' and '%s'; likely model-side collapse.",
                                self.feature_key,
                                fallback_key,
                            )
                else:
                    root_cause_unknown = 1.0
                    if trainer.global_rank == 0:
                        logging.warning(
                            "Gram-matrix saturation detected on '%s' with no available backbone fallback.",
                            self.feature_key,
                        )

            self._report_debug_diagnostics(
                trainer,
                {
                    "gram_matrix_debug/offdiag_mean": primary_stats["offdiag_mean"],
                    "gram_matrix_debug/offdiag_std": primary_stats["offdiag_std"],
                    "gram_matrix_debug/sample_variance": primary_stats["sample_variance"],
                    "gram_matrix_debug/fallback_used": fallback_used,
                    "gram_matrix_debug/root_cause_framework": root_cause_framework,
                    "gram_matrix_debug/root_cause_model": root_cause_model,
                    "gram_matrix_debug/root_cause_unknown": root_cause_unknown,
                    **stage_metrics,
                },
            )

            # Log visualization to wandb (only on rank 0)
            self._log_gram_matrix(trainer, gram_matrix)
            
        except Exception as e:
            # Gracefully handle errors to avoid interrupting training
            if trainer.global_rank == 0:
                logging.warning(f"Failed to compute/log gram matrix at step {trainer.global_step}: {e}")
        finally:
            # Clean up to prevent memory leaks
            if features is not None:
                del features
            if fallback_features is not None:
                del fallback_features
            if 'primary_normalized' in locals():
                del primary_normalized
            if 'fallback_normalized' in locals():
                del fallback_normalized
            if 'fallback_gram' in locals():
                del fallback_gram
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
        fig, ax = plt.subplots(figsize=(10, 8))
        try:
            # Convert to numpy for plotting
            gram_np = gram_matrix.cpu().numpy()

            # Use adaptive colormap range so structure is visible even when all
            # cosines are in a narrow high-similarity band (e.g. 0.95-1.0).
            # Fixed vmin=-1 collapses the entire range into a single yellow band.
            n = gram_np.shape[0]
            if n > 1:
                offdiag_mask = ~np.eye(n, dtype=bool)
                offdiag_vals = gram_np[offdiag_mask]
                offdiag_mean = float(offdiag_vals.mean())
                offdiag_std = float(offdiag_vals.std())
                vmin = float(offdiag_vals.min())
                # Pad slightly so off-diagonal minimum isn't clipped to identical colour
                vmin = max(-1.0, vmin - 0.05 * abs(1.0 - vmin))
            else:
                offdiag_mean = 0.0
                offdiag_std = 0.0
                vmin = -1.0
            vmax = 1.0

            # Plot heatmap
            im = ax.imshow(gram_np, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)

            # Labels and title — include actual statistics so the researcher can
            # distinguish "all yellow due to collapsed range" from "all yellow = truly 1".
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Sample Index')
            ax.set_title(
                f'Feature Gram Matrix (Step {trainer.global_step})\n'
                f'off-diag mean={offdiag_mean:.4f}  std={offdiag_std:.4f}  '
                f'cmap=[{vmin:.3f}, {vmax:.3f}]'
            )

            # Add grid for better readability
            ax.grid(False)

            # Log to wandb if using WandbLogger
            if trainer.logger is not None:
                try:
                    # Check if using WandbLogger
                    from pytorch_lightning.loggers import WandbLogger
                    import wandb

                    if (
                        os.environ.get("WANDB_DISABLED") in {"1", "true", "True"}
                        or os.environ.get("WANDB_MODE") == "disabled"
                    ):
                        if not self._warned_wandb_disabled:
                            logging.warning(
                                "WANDB is disabled via environment variables; gram matrix will not be logged."
                            )
                            self._warned_wandb_disabled = True
                        return

                    wandb_logger = self._get_wandb_logger(trainer, WandbLogger)
                    if wandb_logger is None:
                        if not self._warned_no_wandb:
                            logging.warning(
                                "WandbLogger not found on trainer; gram matrix will not be logged."
                            )
                            self._warned_no_wandb = True
                        return

                    wandb_logger.experiment.log(
                        {"gram_matrix": wandb.Image(fig)},
                        step=trainer.global_step,
                    )
                except (ImportError, AttributeError) as e:
                    # Fallback: log a warning if wandb is not available
                    logging.warning(f"Could not log gram matrix to wandb: {e}")
        finally:
            # Close figure to free memory even on early returns.
            plt.close(fig)

    def _report_extraction_issue(self, trainer, message: str) -> None:
        """Warn and best-effort log a structured extraction issue metric on rank 0."""
        if trainer.global_rank != 0:
            return

        logging.warning(message)
        if self._is_wandb_disabled():
            self._warn_wandb_disabled_once()
            return
        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        log_fn = getattr(experiment, "log", None)
        if callable(log_fn):
            try:
                log_fn({"gram_matrix_error": message})
            except Exception as err:  # pragma: no cover - defensive telemetry path
                logging.warning(f"Failed to record gram_matrix_error metric: {err}")

    def _prepare_features_for_gram(self, features: torch.Tensor, trainer) -> torch.Tensor:
        """Normalize and gather feature tensors in a DDP-safe way."""
        features_normalized = F.normalize(features, dim=-1, p=2)

        world_size = trainer.world_size if dist.is_available() and dist.is_initialized() else 1
        if world_size > 1:
            per_rank_max = max(1, math.ceil(self.max_samples / world_size))
            if features_normalized.shape[0] > per_rank_max:
                indices = torch.linspace(
                    0,
                    features_normalized.shape[0] - 1,
                    per_rank_max,
                    dtype=torch.long,
                    device=features_normalized.device,
                )
                features_normalized = features_normalized[indices]

            local_count = torch.tensor(
                [features_normalized.shape[0]],
                dtype=torch.long,
                device=features_normalized.device,
            )
            gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
            dist.all_gather(gathered_counts, local_count)
            counts = [int(c.item()) for c in gathered_counts]
            max_count = max(counts)
            feature_dim = features_normalized.shape[1]

            if features_normalized.shape[0] < max_count:
                pad_rows = max_count - features_normalized.shape[0]
                padding = torch.zeros(
                    (pad_rows, feature_dim),
                    device=features_normalized.device,
                    dtype=features_normalized.dtype,
                )
                padded = torch.cat([features_normalized, padding], dim=0)
            else:
                padded = features_normalized

            gathered_padded = [
                torch.zeros(
                    (max_count, feature_dim),
                    device=features_normalized.device,
                    dtype=features_normalized.dtype,
                )
                for _ in range(world_size)
            ]
            dist.all_gather(gathered_padded, padded)
            gathered_features = [
                tensor[:count] for tensor, count in zip(gathered_padded, counts) if count > 0
            ]
            if not gathered_features:
                return features_normalized
            features_normalized = torch.cat(gathered_features, dim=0)

            del gathered_counts
            del gathered_padded
            del gathered_features
            del local_count
            if "padding" in locals():
                del padding
            del padded

        if features_normalized.shape[0] > self.max_samples:
            indices = torch.linspace(
                0,
                features_normalized.shape[0] - 1,
                self.max_samples,
                dtype=torch.long,
                device=features_normalized.device,
            )
            features_normalized = features_normalized[indices]
        return features_normalized

    def _sample_features_for_stage(self, features_normalized: torch.Tensor, trainer) -> torch.Tensor:
        """Apply pre-gather and global sample capping without distributed collectives."""
        sampled = features_normalized
        world_size = trainer.world_size if dist.is_available() and dist.is_initialized() else 1
        if world_size > 1:
            per_rank_max = max(1, math.ceil(self.max_samples / world_size))
            if sampled.shape[0] > per_rank_max:
                indices = torch.linspace(
                    0,
                    sampled.shape[0] - 1,
                    per_rank_max,
                    dtype=torch.long,
                    device=sampled.device,
                )
                sampled = sampled[indices]
        if sampled.shape[0] > self.max_samples:
            indices = torch.linspace(
                0,
                sampled.shape[0] - 1,
                self.max_samples,
                dtype=torch.long,
                device=sampled.device,
            )
            sampled = sampled[indices]
        return sampled

    def _compute_saturation_stats(self, gram_matrix: torch.Tensor, features: torch.Tensor):
        """Compute robust saturation statistics for root-cause diagnostics."""
        n = gram_matrix.shape[0]
        if n <= 1:
            return {
                "offdiag_mean": 0.0,
                "offdiag_std": 0.0,
                "sample_variance": 0.0,
            }

        offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=gram_matrix.device)
        offdiag_vals = gram_matrix[offdiag_mask]

        sample_var = features.float().var(dim=0, unbiased=False).mean().item()
        return {
            "offdiag_mean": offdiag_vals.mean().item(),
            "offdiag_std": offdiag_vals.std(unbiased=False).item(),
            "sample_variance": sample_var,
        }

    def _is_saturated(self, stats) -> bool:
        return (
            stats["offdiag_mean"] >= self.saturation_offdiag_threshold
            and stats["offdiag_std"] <= self.saturation_offdiag_std_threshold
            and stats["sample_variance"] <= self.saturation_sample_var_threshold
        )

    def _fallback_feature_key_for(self, feature_key: str):
        fallback_map = {
            "teacher_cls_token": "teacher_cls_token_backbone",
            "student_glob_cls_token": "student_glob_cls_token_backbone",
            "student_cls_token": "student_cls_token_backbone",
        }
        return fallback_map.get(feature_key)

    def _report_debug_diagnostics(self, trainer, metrics: dict) -> None:
        """Best-effort debug telemetry emission on rank 0 only."""
        if trainer.global_rank != 0:
            return
        if self._is_wandb_disabled():
            self._warn_wandb_disabled_once()
            return
        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        log_fn = getattr(experiment, "log", None)
        if callable(log_fn):
            try:
                log_fn(metrics, step=trainer.global_step)
            except Exception as err:  # pragma: no cover - defensive telemetry path
                logging.warning(f"Failed to record gram_matrix_debug metrics: {err}")

    def _get_wandb_logger(self, trainer, wandb_logger_type):
        loggers = []
        if hasattr(trainer, "loggers") and trainer.loggers is not None:
            loggers = list(trainer.loggers)
        elif hasattr(trainer.logger, "loggers"):
            loggers = list(trainer.logger.loggers)
        else:
            loggers = [trainer.logger]

        for logger in loggers:
            if isinstance(logger, wandb_logger_type):
                return logger
        return None

    def _build_stage_metrics(self, stage_name: str, features: torch.Tensor) -> dict:
        """Build stable stage-level telemetry metrics from features."""
        if features is None or not hasattr(features, "dim") or features.dim() != 2:
            stats = {
                "offdiag_mean": 0.0,
                "offdiag_std": 0.0,
                "sample_variance": 0.0,
            }
        else:
            normalized = F.normalize(features, dim=-1, p=2)
            gram = torch.matmul(normalized, normalized.T)
            stats = self._compute_saturation_stats(gram, normalized)
        return {
            f"gram_matrix_debug/stage_{stage_name}_offdiag_mean": stats["offdiag_mean"],
            f"gram_matrix_debug/stage_{stage_name}_offdiag_std": stats["offdiag_std"],
            f"gram_matrix_debug/stage_{stage_name}_sample_variance": stats["sample_variance"],
        }

    def _is_wandb_disabled(self) -> bool:
        disabled = str(os.environ.get("WANDB_DISABLED", "")).strip().lower()
        mode = str(os.environ.get("WANDB_MODE", "")).strip().lower()
        return disabled in {"1", "true", "yes", "on"} or mode == "disabled"

    def _warn_wandb_disabled_once(self) -> None:
        if not self._warned_wandb_disabled:
            logging.warning(
                "WANDB is disabled via environment variables; gram matrix will not be logged."
            )
            self._warned_wandb_disabled = True
