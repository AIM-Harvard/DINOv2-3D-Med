"""
3D-aware iBOT patch loss.

Follows the logic of Lightly's IBOTPatchLoss (patch-level cross-entropy with
centered teacher logits and per-image weighting) while allowing volumetric
patch masks (D x H x W). Teacher/student patch tokens are selected using the
mask and then flattened to match the reference implementation semantics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightly.models.modules.center import Center


class IBOTPatchLoss3D(nn.Module):
    """
    Patch-level iBOT loss with 3D mask support.

    Args:
        output_dim: Dimension of the projection head output.
        teacher_temp: Temperature for teacher logits.
        student_temp: Temperature for student logits.
        center_mode: Mode for center calculation (kept for parity with upstream).
        center_momentum: Momentum term for center updates.
    """

    def __init__(
        self,
        output_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_mode: str = "mean",
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center = Center(
            size=(1, output_dim),
            mode=center_mode,
            momentum=center_momentum,
        )

    def forward(
        self,
        teacher_out: Tensor,
        student_out: Tensor,
        mask: Tensor,
        teacher_temp: float | None = None,
    ) -> Tensor:
        """
        Args:
            teacher_out: Teacher patch tokens. Shape [B, P, C] (full grid) or [B*N, C] (masked tokens).
            student_out: Student patch tokens, same shape semantics as teacher_out.
            mask: Boolean patch mask, shape [B, P] or [B, D, H, W]. True indicates masked tokens.
            teacher_temp: Optional override for teacher temperature.
        Returns:
            Scalar loss.
        """
        teacher_temperature = torch.tensor(
            teacher_temp if teacher_temp is not None else self.teacher_temp,
            device=teacher_out.device,
            dtype=teacher_out.dtype,
        )

        if mask.dim() > 2:
            mask_flat = mask.flatten(start_dim=1)
        else:
            mask_flat = mask
        mask_flat = mask_flat.to(torch.bool)
        B = mask_flat.shape[0]

        # Handle two input shapes:
        # 1) [B, P, C]: select masked tokens using mask.
        # 2) [B*N, C]: assume inputs already filtered to masked tokens.
        if teacher_out.dim() == 3:
            if teacher_out.shape != student_out.shape:
                raise ValueError(
                    f"Teacher/student patch shapes differ: {teacher_out.shape} vs {student_out.shape}"
                )
            if teacher_out.shape[:2] != mask_flat.shape:
                raise ValueError(
                    f"Mask shape {mask_flat.shape} does not match patch tokens {teacher_out.shape[:2]}"
                )
            teacher_sel = teacher_out[mask_flat]
            student_sel = student_out[mask_flat]
        elif teacher_out.dim() == 2:
            if teacher_out.shape != student_out.shape:
                raise ValueError(
                    f"Teacher/student patch shapes differ: {teacher_out.shape} vs {student_out.shape}"
                )
            if teacher_out.shape[0] != mask_flat.sum():
                raise ValueError(
                    f"Number of masked tokens {mask_flat.sum()} does not match provided logits {teacher_out.shape[0]}"
                )
            teacher_sel = teacher_out
            student_sel = student_out
        else:
            raise ValueError("teacher_out must be 2D or 3D (got dim={})".format(teacher_out.dim()))

        # Cross-entropy between centered teacher and student distributions
        teacher_softmax = F.softmax(
            (teacher_sel - self.center.value.to(teacher_sel.device)) / teacher_temperature, dim=-1
        )
        student_log_softmax = F.log_softmax(student_sel / self.student_temp, dim=-1)

        loss = -torch.sum(teacher_softmax * student_log_softmax, dim=-1)

        # Per-image weighting as in the reference implementation
        num_masked_per_image = mask_flat.sum(dim=1, keepdim=True).clamp(min=1.0)
        weight = (1.0 / num_masked_per_image).expand_as(mask_flat)[mask_flat]
        loss = (loss * weight).sum() / B

        self.center.update(teacher_sel)
        return loss
