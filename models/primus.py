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


class Primus(_Primus):
    def __init__(self, *args, **kwargs):
        self.embed_dim = kwargs.get("embed_dim", 768)
        classification = kwargs.pop("classification", False)
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)) if classification else None
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)


        ref_feat_shape = tuple([i // ds for i, ds in zip(kwargs.get("input_shape", (128, 128, 128)), kwargs.get("patch_embed_size", (16, 16, 16)))])
        self.vit = Eva(
                embed_dim=kwargs.get("embed_dim", 768),
                depth=kwargs.get("eva_depth", 12),
                num_heads=kwargs.get("eva_numheads", 12),
                ref_feat_shape=ref_feat_shape,
                num_reg_tokens=kwargs.get("num_register_tokens", 0) + (1 if classification else 0),
                use_rot_pos_emb=kwargs.get("use_rot_pos_emb", True),
                use_abs_pos_emb=kwargs.get("use_abs_pos_embed", True),
                mlp_ratio=kwargs.get("mlp_ratio", 4 * 2 / 3),
                drop_path_rate=kwargs.get("drop_path_rate", 0),
                patch_drop_rate=kwargs.get("patch_drop_rate", 0),
                proj_drop_rate=kwargs.get("proj_drop_rate", 0),
                attn_drop_rate=kwargs.get("attn_drop_rate", 0),
                rope_impl=kwargs.get("rope_impl", RotaryEmbeddingCat),
                rope_kwargs=kwargs.get("rope_kwargs", None),
                init_values=kwargs.get("init_values", None),
                scale_attn_inner=kwargs.get("scale_attn_inner", False),
            )  

        self.sequence_length = np.prod(ref_feat_shape) + (1 if classification else 0)
                 
    def _pos_embed(self, x):
        pos_embed = self.eva.pos_embed
        rot_pos_embed = self.eva.rope.get_embed() if self.eva.rope is not None else None

        if pos_embed is not None:
            x = x + pos_embed

        x = self.eva.pos_drop(x)
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