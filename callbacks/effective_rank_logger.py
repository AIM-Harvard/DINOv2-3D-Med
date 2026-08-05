"""
Callback to log the effective rank of learned representations to wandb.

Accumulates CLS token embeddings from the current training batch, computes
SVD on mean-centred features, and derives effective rank using the Roy &
Vetterli (2007) definition:

    effective_rank = exp(H(σ / ‖σ‖₁))

where H is the Shannon entropy of the normalised singular value distribution.

Effective rank near 1 signals dimensional collapse; high values signal that
the model is using many dimensions of its representation space.

Logs:
    collapse_monitor/effective_rank   — Roy & Vetterli effective rank (≥ 1.0)
    collapse_monitor/num_svd_samples  — number of samples used for SVD
"""

import math
import os
import logging
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import torch.distributed as dist
from typing import Any, Dict, List, Optional


class EffectiveRankCallback(Callback):
    """
    Callback that logs SVD-based effective rank of CLS token embeddings.

    Every `log_every_n_steps` steps, performs an extra forward pass to extract
    features, gathers them across DDP ranks (rank 0 only), mean-centres them,
    and computes effective rank via singular value decomposition.

    Interpretation:
        - Healthy: effective_rank ≈ 50–500+
        - Soft collapse warning: effective_rank < 10
        - Hard collapse: effective_rank < 2
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        feature_key: str = "student_cls_token",
        max_buffer_samples: int = 512,
    ):
        """
        Args:
            log_every_n_steps: Compute and log effective rank every N training steps.
            feature_key: Key to extract from model pred dict. Same convention as
                GramMatrixCallback (default: "student_cls_token").
            max_buffer_samples: Maximum samples to gather per SVD computation. Caps
                memory and SVD cost. At 512 × 65536 × float32 ≈ 134 MB on rank 0.
        """
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be > 0")
        self.log_every_n_steps = log_every_n_steps
        self.feature_key = feature_key
        self.max_buffer_samples = max_buffer_samples
        self._warned_no_wandb = False
        self._warned_wandb_disabled = False

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Cadence guard — skip step 0 and non-interval steps
        if trainer.global_step == 0 or trainer.global_step % self.log_every_n_steps != 0:
            return

        features = None
        error_occurred = False

        try:
            if not isinstance(batch, (list, tuple)) or len(batch) < 1:
                if trainer.global_rank == 0:
                    logging.warning(
                        "EffectiveRankCallback: invalid batch structure at step %d",
                        trainer.global_step,
                    )
                error_occurred = True
            else:
                model = getattr(pl_module, "model", None)
                model_training_states: Dict[torch.nn.Module, bool] = {}
                if model is not None:
                    model_training_states = self._snapshot_training_states(model)
                    model.eval()
                try:
                    with torch.no_grad():
                        views_container = batch[0]
                        if not isinstance(views_container, (list, tuple)) or len(views_container) == 0:
                            if trainer.global_rank == 0:
                                logging.warning(
                                    "EffectiveRankCallback: no views in batch at step %d",
                                    trainer.global_step,
                                )
                            error_occurred = True
                        else:
                            views = views_container[: min(2, len(views_container))]
                            model_outputs = pl_module.model(views)
                            pred_dict = model_outputs
                            features = pred_dict.get(self.feature_key, None)
                            if features is None or features.dim() != 2:
                                if trainer.global_rank == 0:
                                    logging.warning(
                                        "EffectiveRankCallback: feature key '%s' not found "
                                        "or wrong shape at step %d",
                                        self.feature_key,
                                        trainer.global_step,
                                    )
                                features = None
                                error_occurred = True
                finally:
                    if model is not None and model_training_states:
                        self._restore_training_states(model_training_states)
        except Exception as e:
            error_occurred = True
            features = None
            if trainer.global_rank == 0:
                logging.warning(
                    "EffectiveRankCallback: feature extraction failed at step %d: %s",
                    trainer.global_step,
                    e,
                )

        # DDP all_reduce gate — propagate any rank failure to all ranks
        has_data = 0 if (error_occurred or features is None) else 1
        device = features.device if features is not None else pl_module.device
        gate = torch.tensor(has_data, device=device, dtype=torch.int)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(gate, op=dist.ReduceOp.MIN)
        if gate.item() == 0:
            return

        try:
            # F6: gather to rank 0 only — non-rank-0 workers send data but don't receive
            gathered = self._gather_features(features, trainer)
            # gathered is the full set on rank 0; local features on non-rank-0 (unused)
            self._compute_and_log_rank(trainer, gathered.cpu())
        finally:
            if features is not None:
                del features

    @rank_zero_only
    def _compute_and_log_rank(self, trainer, features: torch.Tensor) -> None:
        try:
            n = features.shape[0]
            if n < 2:
                self._log_metrics(
                    trainer,
                    {
                        "collapse_monitor/effective_rank": 1.0,
                        "collapse_monitor/num_svd_samples": n,
                    },
                )
                return

            # F2: mean-centre to remove DC offset before SVD; otherwise the first
            # singular value is dominated by the mean and effective rank ≈ 1 always.
            features = features - features.mean(dim=0, keepdim=True)

            # SVD in float32 to avoid precision issues
            sigma = torch.linalg.svdvals(features.float())  # [min(N, D)]
            sigma = sigma[sigma > 0]  # drop numerical zeros
            if sigma.numel() == 0:
                self._log_metrics(
                    trainer,
                    {
                        "collapse_monitor/effective_rank": 1.0,
                        "collapse_monitor/num_svd_samples": n,
                    },
                )
                return

            # Roy & Vetterli (2007): effective_rank = exp(H(σ / ‖σ‖₁))
            p = sigma / sigma.sum()
            eps = 1e-9
            effective_rank = torch.exp(-(p * (p + eps).log()).sum()).item()

            self._log_metrics(
                trainer,
                {
                    "collapse_monitor/effective_rank": effective_rank,
                    "collapse_monitor/num_svd_samples": n,
                },
            )
        except Exception as e:
            logging.warning(
                "EffectiveRankCallback: SVD failed at step %d: %s",
                trainer.global_step,
                e,
            )

    def _gather_features(self, features: torch.Tensor, trainer) -> torch.Tensor:
        """Gather features to rank 0 only using dist.gather (not all_gather).

        Non-rank-0 workers send their features to rank 0 but do not receive
        the full gathered matrix, saving memory proportional to world_size.
        """
        world_size = (
            trainer.world_size
            if (dist.is_available() and dist.is_initialized())
            else 1
        )
        if world_size == 1:
            return features

        # F11: random per-rank subsample (avoids deterministic stride bias)
        per_rank_max = max(1, math.ceil(self.max_buffer_samples / world_size))
        if features.shape[0] > per_rank_max:
            indices = torch.randperm(
                features.shape[0], device=features.device
            )[:per_rank_max]
            features = features[indices]

        # Gather counts from all ranks so rank 0 knows each rank's contribution
        local_count = torch.tensor(
            [features.shape[0]], dtype=torch.long, device=features.device
        )
        gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(c.item()) for c in gathered_counts]
        max_count = max(counts)
        feature_dim = features.shape[1]

        # Pad local tensor to max_count so gather buffers are uniform size
        if features.shape[0] < max_count:
            padding = torch.zeros(
                (max_count - features.shape[0], feature_dim),
                device=features.device,
                dtype=features.dtype,
            )
            padded = torch.cat([features, padding], dim=0)
        else:
            padded = features

        # F6: dist.gather to rank 0 only — non-root ranks don't allocate gather_list
        gather_list = None
        if trainer.global_rank == 0:
            gather_list = [
                torch.zeros(
                    (max_count, feature_dim),
                    device=features.device,
                    dtype=features.dtype,
                )
                for _ in range(world_size)
            ]
        dist.gather(padded, gather_list, dst=0)

        if trainer.global_rank == 0:
            gathered = [t[:c] for t, c in zip(gather_list, counts) if c > 0]
            result = torch.cat(gathered, dim=0) if gathered else features
            # Final global cap
            if result.shape[0] > self.max_buffer_samples:
                indices = torch.randperm(result.shape[0])[: self.max_buffer_samples]
                result = result[indices]
            return result

        # Non-rank-0: return local features (caller uses @rank_zero_only methods only)
        return features

    @rank_zero_only
    def _log_metrics(self, trainer, metrics: dict) -> None:
        if self._is_wandb_disabled():
            if not self._warned_wandb_disabled:
                logging.warning(
                    "WANDB is disabled via environment variables; "
                    "effective rank will not be logged."
                )
                self._warned_wandb_disabled = True
            return
        try:
            from pytorch_lightning.loggers import WandbLogger

            wandb_logger = self._get_wandb_logger(trainer, WandbLogger)
            if wandb_logger is None:
                if not self._warned_no_wandb:
                    logging.warning(
                        "WandbLogger not found on trainer; effective rank will not be logged."
                    )
                    self._warned_no_wandb = True
                return
            wandb_logger.experiment.log(metrics, step=trainer.global_step)
        except (ImportError, AttributeError) as e:
            logging.warning(
                "EffectiveRankCallback: could not log metrics: %s", e
            )

    def _snapshot_training_states(
        self, module: torch.nn.Module
    ) -> Dict[torch.nn.Module, bool]:
        return {submodule: submodule.training for submodule in module.modules()}

    def _restore_training_states(
        self, states: Dict[torch.nn.Module, bool]
    ) -> None:
        for submodule, was_training in states.items():
            submodule.train(was_training)

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

    def _is_wandb_disabled(self) -> bool:
        disabled = str(os.environ.get("WANDB_DISABLED", "")).strip().lower()
        mode = str(os.environ.get("WANDB_MODE", "")).strip().lower()
        return disabled in {"1", "true", "yes", "on"} or mode == "disabled"
