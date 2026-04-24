"""
Callback to log the entropy of the teacher's output distribution to wandb.

Monitors collapse by tracking how uniform the teacher's DINO-centered softmax
distribution is. Low entropy (near 0) signals mode collapse; high entropy
(near log(output_dim)) signals healthy diversity.
"""

import math
import os
import logging
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
import torch.distributed as dist
from typing import Any, Dict, Optional


class TeacherEntropyCallback(Callback):
    """
    Callback that logs the Shannon entropy of the teacher's output distribution to wandb.

    Uses the proper DINO-centered distribution:
        p = softmax((teacher_cls_token - center) / teacher_temp)
    so the entropy reflects the actual distribution the loss operates on.

    Logs:
        collapse_monitor/teacher_entropy         — entropy in nats
        collapse_monitor/teacher_entropy_normalized — entropy / log(output_dim), in [0, 1]

    Interpretation:
        - Healthy: normalized entropy ≈ 0.7–1.0
        - Collapse warning: normalized entropy < 0.1
    """

    def __init__(
        self,
        log_every_n_steps: int = 100,
        max_samples: int = 128,
    ):
        """
        Args:
            log_every_n_steps: Log entropy every N training steps.
            max_samples: Maximum number of samples to use per entropy estimate.
        """
        super().__init__()
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be > 0")
        self.log_every_n_steps = log_every_n_steps
        self.max_samples = max_samples
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

        teacher_logits = None
        error_occurred = False

        try:
            if not isinstance(batch, (list, tuple)) or len(batch) < 1:
                if trainer.global_rank == 0:
                    logging.warning(
                        "TeacherEntropyCallback: invalid batch structure at step %d",
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
                                    "TeacherEntropyCallback: no views in batch at step %d",
                                    trainer.global_step,
                                )
                            error_occurred = True
                        else:
                            views = views_container[: min(2, len(views_container))]
                            model_outputs = pl_module.model(views)
                            pred_dict = model_outputs
                            teacher_logits = pred_dict.get("teacher_cls_token", None)
                            if teacher_logits is None or teacher_logits.dim() != 2:
                                if trainer.global_rank == 0:
                                    logging.warning(
                                        "TeacherEntropyCallback: 'teacher_cls_token' not found "
                                        "or wrong shape at step %d",
                                        trainer.global_step,
                                    )
                                teacher_logits = None
                                error_occurred = True
                finally:
                    if model is not None and model_training_states:
                        self._restore_training_states(model_training_states)
        except Exception as e:
            error_occurred = True
            teacher_logits = None
            if trainer.global_rank == 0:
                logging.warning(
                    "TeacherEntropyCallback: feature extraction failed at step %d: %s",
                    trainer.global_step,
                    e,
                )

        # DDP all_reduce gate — any rank failure propagates to all
        has_data = 0 if (error_occurred or teacher_logits is None) else 1
        device = teacher_logits.device if teacher_logits is not None else pl_module.device
        gate = torch.tensor(has_data, device=device, dtype=torch.int)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(gate, op=dist.ReduceOp.MIN)
        if gate.item() == 0:
            return

        try:
            # F11: random subsample (avoids deterministic stride bias from linspace)
            if teacher_logits.shape[0] > self.max_samples:
                indices = torch.randperm(
                    teacher_logits.shape[0], device=teacher_logits.device
                )[: self.max_samples]
                teacher_logits = teacher_logits[indices]

            self._compute_and_log_entropy(trainer, pl_module, teacher_logits)
        finally:
            if teacher_logits is not None:
                del teacher_logits

    @rank_zero_only
    def _compute_and_log_entropy(
        self, trainer, pl_module, teacher_logits: torch.Tensor
    ) -> None:
        try:
            dino_loss_fn = getattr(
                getattr(pl_module, "criterion", None), "dino_loss_fn", None
            )
            if dino_loss_fn is None:
                logging.warning(
                    "TeacherEntropyCallback: cannot access criterion.dino_loss_fn at step %d; "
                    "skipping entropy computation.",
                    trainer.global_step,
                )
                return

            # F4: explicit reshape instead of squeeze — avoids silent broadcast on unexpected shapes
            center = dino_loss_fn.center.detach().reshape(-1).to(teacher_logits.device)
            if center.shape[0] != teacher_logits.shape[-1]:
                logging.warning(
                    "TeacherEntropyCallback: center dim %d does not match teacher_logits dim %d "
                    "at step %d; skipping.",
                    center.shape[0],
                    teacher_logits.shape[-1],
                    trainer.global_step,
                )
                return

            teacher_temp = float(dino_loss_fn.teacher_temp)
            if teacher_temp <= 0:
                teacher_temp = 0.04  # fallback to DINO default min temperature

            # Proper DINO-centered distribution
            probs = F.softmax(
                (teacher_logits - center) / teacher_temp, dim=-1
            )  # [N, D]

            # Shannon entropy in nats, averaged over samples
            eps = 1e-9
            entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean().item()

            output_dim = teacher_logits.shape[-1]
            max_entropy = math.log(output_dim)
            entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0.0

            self._log_metrics(
                trainer,
                {
                    "collapse_monitor/teacher_entropy": entropy,
                    "collapse_monitor/teacher_entropy_normalized": entropy_normalized,
                },
            )
        except Exception as e:
            logging.warning(
                "TeacherEntropyCallback: entropy computation failed at step %d: %s",
                trainer.global_step,
                e,
            )

    @rank_zero_only
    def _log_metrics(self, trainer, metrics: dict) -> None:
        if self._is_wandb_disabled():
            if not self._warned_wandb_disabled:
                logging.warning(
                    "WANDB is disabled via environment variables; "
                    "teacher entropy will not be logged."
                )
                self._warned_wandb_disabled = True
            return
        try:
            from pytorch_lightning.loggers import WandbLogger

            wandb_logger = self._get_wandb_logger(trainer, WandbLogger)
            if wandb_logger is None:
                if not self._warned_no_wandb:
                    logging.warning(
                        "WandbLogger not found on trainer; teacher entropy will not be logged."
                    )
                    self._warned_no_wandb = True
                return
            wandb_logger.experiment.log(metrics, step=trainer.global_step)
        except (ImportError, AttributeError) as e:
            logging.warning(
                "TeacherEntropyCallback: could not log metrics: %s", e
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
