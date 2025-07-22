"""
PyTorch LightningModule for 3D DINOv2 self-supervised training.
Handles optimizer, scheduler, training/validation steps, and teacher-student updates.
"""

from __future__ import annotations

import copy
import math
import re
from functools import partial
from typing import Any
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

from lightly.utils.benchmarking import OnlineLinearClassifier
from lightly.utils.optim import update_param_groups
from lightly.utils.scheduler import (
    CosineWarmupScheduler,
    cosine_schedule,
)

from models.meta_arch import DINOv2_3D_Meta_Architecture
from losses.dino import DINOv2Loss


class DINOv2_3D_LightningModule(LightningModule):
    """
    PyTorch LightningModule for 3D DINOv2 self-supervised learning.
    Implements training, prediction, optimizer config, and teacher-student logic.
    """

    def __init__(
        self,
        batch_size_per_device: int,
        hidden_size: int = 768,
        ibot_separate_head: bool = True,
        base_lr: float = 0.0005,  # Reduced from 0.004 as per issue #6
        min_lr: float = 1e-6,
        weight_decay: float = 0.04,
        layer_decay: float = 0.9,
        gradient_clip_val: float = 3.0,
        teacher_temp_warmup_epochs: int = 30,
        teacher_temp_min: float = 0.04,
        teacher_temp_max: float = 0.07,
        freeze_last_layer_epochs: int = 1,
        projection_dim: int = 65536,
        backbone: nn.Module = None,
    ) -> None:
        """
        Initialize the DINOv2Trainer3D LightningModule.
        Args: see model config for details.
        """
        super().__init__()
        self.save_hyperparameters()
        self.batch_size_per_device = batch_size_per_device
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.gradient_clip_val = gradient_clip_val
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.teacher_temp_min = teacher_temp_min
        self.teacher_temp_max = teacher_temp_max
        self.freeze_last_layer_epochs = freeze_last_layer_epochs
        self.metrics = {"train": None, "val": None}

        self.save_hyperparameters()

        # Model
        self.model = DINOv2_3D_Meta_Architecture(
            hidden_size=hidden_size,
            norm_last_layer=False,
            ibot_separate_head=ibot_separate_head,
            projection_dim=projection_dim,
            backbone=backbone,
        )

        # Loss
        self.criterion = DINOv2Loss(
            teacher_temp_min=teacher_temp_min,
            teacher_temp_max=teacher_temp_max,
            teacher_temp_warmup_epochs=teacher_temp_warmup_epochs,
            output_dim=projection_dim,
            ibot_loss_weight=1.0,
            koleo_loss_weight=0.1,
        )

        # self.online_classifier = OnlineLinearClassifier(
        #     feature_dim=768,
        #     num_classes=num_classes,
        # )

    def predict_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        """
        Inference step for prediction mode.
        Args:
            batch: Tuple of (inputs, targets, meta)
            batch_idx: Batch index
        Returns:
            Model outputs
        """
        inputs = batch[0]
        with torch.no_grad():
            outputs = self.model.encode(inputs)
            return outputs

    def training_step(
        self, batch: tuple[list[Tensor], Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        views, targets = batch[0], batch[1]

        # Forward pass
        outputs = self.model(views)

        # Calculate losses with proper scheduling
        loss_dict = self.criterion(
            outputs["pred"], global_step=self.trainer.global_step
        )

        # Log losses
        self.log_dict(
            {
                "train_loss": loss_dict["total_loss"],
                "train_dino_loss": loss_dict["dino_loss"],
                "train_ibot_loss": loss_dict["ibot_loss"]
                if loss_dict["ibot_loss"] is not None
                else 0.0,
                "train_koleo_loss": loss_dict["koleo_loss"]
                if loss_dict["koleo_loss"] is not None
                else 0.0,
                "teacher_temp": loss_dict["teacher_temp"],
                "global_step": float(self.trainer.global_step),  # Add for debugging
            },
            prog_bar=False,
            sync_dist=True,
            batch_size=len(targets),
        )
        return loss_dict["total_loss"]

    def validation_step(
        self, batch: tuple[Tensor, Tensor, list[str]], batch_idx: int
    ) -> Tensor:
        images, targets = batch[0], batch[1]
        cls_token = self.model.teacher_backbone(images)
        return cls_token

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        # Calculate learning rate based on batch size
        lr_scale = math.sqrt(
            self.batch_size_per_device * self.trainer.world_size / 1024
        )
        lr = self.base_lr * lr_scale
        num_layers = len(self.model.student_backbone.vit.blocks)

        def lr_layer(layer_idx: int) -> float:
            return self.layer_decay ** (num_layers + 1 - layer_idx)

        # Create parameter groups with layer-wise learning rates
        param_groups = []

        # Fix: Only include student parameters that require gradients
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Skip teacher parameters (they should not require gradients anyway)
            if "teacher" in name:
                continue

            # Skip if not student parameters
            if "student" not in name:
                continue

            group = {
                "name": name,
                "params": [param],
                "lr": lr,
                "weight_decay": self.weight_decay,
            }

            # Update lr based on layer
            if any(
                s in name
                for s in [
                    "pos_embed",
                    "mask_token",
                    "cls_token",
                    "register_tokens",
                ]
            ):
                group["lr"] = lr * lr_layer(0)
            elif "patch_embed" in name:
                group["lr"] = lr * lr_layer(0) * 0.2
            elif "residual" in name:
                group["lr"] = lr
            elif "blocks" in name:
                # Fix: More robust regex matching
                match = re.search(r"blocks\.(\d+)\.", name)
                if match:
                    layer_idx = int(match.group(1))
                    group["lr"] = lr * lr_layer(layer_idx + 1)
            elif "norm" in name:
                # Use default lr for norm layers
                pass
            elif "head" in name or "_dino_head" in name or "_ibot_head" in name:
                # Use default lr for heads
                pass
            else:
                # For any other student parameters, use default lr
                pass

            # Update weight_decay
            if name.endswith(".bias") or ".norm" in name or "gamma" in name:
                group["weight_decay"] = 0.0

            # Include parameter group
            param_groups.append(group)

        # Ensure we have parameters to optimize
        if not param_groups:
            raise ValueError("No student parameters found for optimization!")

        print(f"Found {len(param_groups)} parameter groups for optimization")
        print(
            f"Total parameters: {sum(len(group['params']) for group in param_groups)}"
        )

        optimizer = AdamW(param_groups, lr=lr)

        # Fix: Ensure proper scheduler configuration
        max_steps = max(self.trainer.estimated_stepping_batches, 1)
        warmup_steps = int(max_steps / self.trainer.max_epochs * 10)

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=warmup_steps,
                max_epochs=max_steps,
                end_value=self.min_lr / lr,
            ),
            "interval": "step",
        }

        self.criterion.max_steps = max_steps
        self.criterion.max_epochs = self.trainer.max_epochs
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: int | float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        # Gradient clipping as per issue #10
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
        )

    def on_before_optimizer_step(self, optimizer: AdamW, *args) -> None:
        # Cancel last layer gradients during warmup (issue #5)
        self.model.cancel_last_layer_gradients(self.current_epoch)

        # Apply weight decay schedule
        weight_decay = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=max(self.trainer.estimated_stepping_batches, 1),
            start_value=0.04,
            end_value=0.4,
        )
        updates = []
        for group in optimizer.param_groups:
            if group["weight_decay"] != 0.0:
                updates.append({"name": group["name"], "weight_decay": weight_decay})

        update_param_groups(optimizer, updates=updates)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update of teacher
        max_steps = max(self.trainer.estimated_stepping_batches, 1000)

        # Update teacher - this should work fine in DDP
        self.model.update_teacher(
            global_step=self.trainer.global_step, max_steps=max_steps
        )

        # Remove manual synchronization - it's causing the hangups
        # DDP will handle this automatically after the backward pass
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def _sync_teacher_parameters(self):
        """Synchronize teacher parameters across all DDP processes."""
        # Remove the problematic conditional synchronization
        # The issue is that all processes need to participate in broadcast

        if self.trainer.world_size <= 1:
            return  # No sync needed for single GPU

        # Use all_reduce instead of broadcast to avoid deadlocks
        with torch.no_grad():
            for param in self.model.teacher_backbone.parameters():
                torch.distributed.all_reduce(
                    param.data, op=torch.distributed.ReduceOp.AVG
                )
            for param in self.model.teacher_dino_head.parameters():
                torch.distributed.all_reduce(
                    param.data, op=torch.distributed.ReduceOp.AVG
                )
            if self.model.ibot_separate_head:
                for param in self.model.teacher_ibot_head.parameters():
                    torch.distributed.all_reduce(
                        param.data, op=torch.distributed.ReduceOp.AVG
                    )
