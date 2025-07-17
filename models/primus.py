from dynamic_network_architectures.architectures.primus import Primus as _Primus
from dynamic_network_architectures.building_blocks.eva import Eva
from timm.layers import RotaryEmbeddingCat
import torch
from torch import nn
from timm.layers import trunc_normal_
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import numpy as np
from dynamic_network_architectures.building_blocks.patch_encode_decode import (
    LayerNormNd,
    PatchDecode,
    PatchEmbed,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from typing import Tuple


class Primus(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        num_classes: int,
        eva_depth: int = 24,
        eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
        attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
        classification: bool = False,
    ):
        super().__init__()
        ref_feat_shape = tuple(
            [i // ds for i, ds in zip(input_shape, patch_embed_size)]
        )
        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)
        self.up_projection = PatchDecode(
            patch_embed_size,
            embed_dim,
            num_classes,
            norm=decoder_norm,
            activation=decoder_act,
        )

        self.mask_token: torch.Tensor
        self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
                if num_register_tokens
                else None
            )
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None

        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, embed_dim)) if classification else None
        )
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

        self.down_projection.apply(InitWeights_He(1e-2))
        self.up_projection.apply(InitWeights_He(1e-2))

        self.vit = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=ref_feat_shape,
            num_reg_tokens=num_register_tokens + (1 if classification else 0),
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        self.sequence_length = np.prod(ref_feat_shape) + (1 if classification else 0)
        self.grid_size = ref_feat_shape

    def _pos_embed(self, x):
        pos_embed = self.vit.pos_embed
        rot_pos_embed = self.vit.rope.get_embed() if self.vit.rope is not None else None

        if pos_embed is not None:
            x = x + pos_embed

        x = self.vit.pos_drop(x)
        return x, rot_pos_embed

    def forward(self, x, mask=None):
        FW, FH, FD = x.shape[2:]  # Full W , ...
        x = self.down_projection(x)
        # last output of the encoder is the input to EVA
        B, C, W, H, D = x.shape
        num_patches = W * H * D
        x = rearrange(x, "b c w h d -> b (w h d) c")

        # Apply masking if provided
        if mask is not None:
            actual_sequence_length = x.shape[1]

            # Adjust mask size if needed
            if mask.shape[1] != actual_sequence_length:
                if mask.shape[1] > actual_sequence_length:
                    mask = mask[:, :actual_sequence_length]
                else:
                    extended_mask = torch.zeros(
                        (B, actual_sequence_length),
                        dtype=torch.bool,
                        device=mask.device,
                    )
                    extended_mask[:, : mask.shape[1]] = mask
                    mask = extended_mask

            # Apply mask tokens
            mask_tokens = self.mask_token.expand(B, actual_sequence_length - 1, -1)
            w = mask[:, 1:].unsqueeze(-1).type_as(mask_tokens)
            x = x.clone()
            x[:, 1:] = x[:, 1:] * (1 - w) + mask_tokens * w

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(B, -1, -1),
                    x,
                ),
                dim=1,
            )

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)

        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.vit.blocks:
            if self.vit.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x = blk(x, rope=rot_pos_embed)

        x = self.vit.norm(x)

        # Remove register tokens (but not class tokens) after forward pass
        if self.register_tokens is not None:
            num_reg_tokens = self.register_tokens.shape[1]
            # If cls_token is present, it is at position 0, so reg tokens are next
            start_idx = 1 if self.cls_token is not None else 0
            end_idx = start_idx + num_reg_tokens
            # Remove register tokens from x
            x = torch.cat((x[:, :start_idx, :], x[:, end_idx:, :]), dim=1)

        return x
