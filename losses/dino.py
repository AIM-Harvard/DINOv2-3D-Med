import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules.center import Center
from lightly.utils.scheduler import linear_warmup_schedule


# References:
#     https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss
#     https://github.com/lightly-ai/lightly/blob/master/benchmarks/imagenet/vitb16/dinov2.py


class DINOv2Loss(nn.Module):
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
        max_steps: int = 1000,
        max_epochs: int = 500,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp_min = teacher_temp_min
        self.teacher_temp_max = teacher_temp_max
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs
        self.center_momentum = center_momentum
        self.output_dim = output_dim
        self.w_ibot = ibot_loss_weight
        self.w_koleo = koleo_loss_weight
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        # Initialize loss functions with dynamic temperature
        self.dino_loss_fn = DINOLoss(
            output_dim=self.output_dim,
            teacher_temp=self.teacher_temp_min,  # Will be updated dynamically
            student_temp=self.student_temp,
            center_momentum=self.center_momentum,
        )

        self.ibot_loss_fn = (
            IBOTPatchLoss(
                output_dim=self.output_dim,
                teacher_temp=self.teacher_temp_min,  # Will be updated dynamically
                student_temp=self.student_temp,
                center_momentum=self.center_momentum,
            )
            if self.w_ibot > 0
            else None
        )

        self.koleo_loss_fn = KoLeoLoss() if self.w_koleo > 0 else None

    def get_teacher_temperature(self, global_step: int) -> float:
        """Calculate teacher temperature with linear warmup using PyTorch Lightning trainer info."""

        warmup_steps = int(
            (self.teacher_temp_warmup_epochs / self.max_epochs) * self.max_steps
        )

        return linear_warmup_schedule(
            step=global_step,
            warmup_steps=warmup_steps,
            start_value=self.teacher_temp_min,
            end_value=self.teacher_temp_max,
        )

    def forward(self, input_dict, global_step: int = 0):
        """
        input_dict: {
            "teacher_cls_out": Tensor,
            "student_cls_out": Tensor,
            "teacher_mask_patches": Tensor (optional),
            "student_mask_glob_patches": Tensor (optional),
            "student_glob_cls_token": Tensor (optional),
            "mask": Tensor (optional, for patch loss),
            "n_local_views": int (optional, number of local views)
        }
        Computes the DINOV2 loss for 3D data with dynamic teacher temperature
        """

        teacher_outputs = input_dict.get("teacher_cls_token", None)
        student_outputs = input_dict.get("student_cls_token", None)
        teacher_patches = input_dict.get("teacher_patch_tokens", None)
        student_patches = input_dict.get("student_patch_tokens", None)
        student_glob_cls_token = input_dict.get("student_glob_cls_token", None)
        mask = input_dict.get("mask", None)
        n_local_views = input_dict.get("n_local_views", 0)

        if student_outputs is None or teacher_outputs is None:
            raise ValueError(
                "Both student_outputs and teacher_outputs must be provided."
            )

        # Calculate dynamic teacher temperature
        teacher_temp = self.get_teacher_temperature(global_step)

        # Update temperature in loss functions
        self.dino_loss_fn.teacher_temp = teacher_temp
        if self.ibot_loss_fn is not None:
            self.ibot_loss_fn.teacher_temp = teacher_temp

        # Convert n_local_views to int if it's a tensor
        if isinstance(n_local_views, torch.Tensor):
            n_local_views = n_local_views.item()

        # DINO loss - proper chunking based on views
        n_views = n_local_views + 2  # 2 global + n local views
        dino_loss = self.dino_loss_fn(
            teacher_out=teacher_outputs.chunk(2),  # 2 global teacher views
            student_out=student_outputs.chunk(n_views),  # All student views
            teacher_temp=teacher_temp,
        )

        # Patch-level iBOT loss
        ibot_loss = (
            self.ibot_loss_fn(
                teacher_out=teacher_patches,
                student_out=student_patches,
                mask=mask,
                teacher_temp=teacher_temp,
            )
            if self.ibot_loss_fn is not None
            else 0
        )

        # Koleo loss - only on global views
        koleo_loss = (
            sum(self.koleo_loss_fn(s) for s in student_glob_cls_token.chunk(2))
            if self.koleo_loss_fn is not None
            else 0
        )

        losses = {
            "dino_loss": dino_loss,
            "ibot_loss": ibot_loss,
            "koleo_loss": koleo_loss,
            "total_loss": dino_loss
            + self.w_ibot * ibot_loss
            + self.w_koleo * koleo_loss,
            "teacher_temp": teacher_temp,
        }

        return losses
