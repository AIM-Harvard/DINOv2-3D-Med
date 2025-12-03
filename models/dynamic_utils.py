import math
from typing import Optional, Tuple, Union, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Try to import to_3tuple from timm, otherwise define it
try:
    from timm.layers import to_3tuple
except ImportError:
    def to_3tuple(x):
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x, x, x)

def resample_abs_pos_embed(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: List[int],
        num_prefix_tokens: int = 1,
        interpolation: str = 'trilinear',
):
    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] * new_size[2] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1] and new_size[1] == new_size[2]:
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], old_size[2], -1).permute(0, 4, 1, 2, 3)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation)
    posemb = posemb.permute(0, 2, 3, 4, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb

def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'trilinear',
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple[int, int, int]): target shape (depth, height, width).
        interpolation (str): interpolation for resize
    Returns:
        Resized patch embedding kernel.
    """
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    assert len(patch_embed.shape) == 5, "Five dimensions expected"
    assert len(new_size) == 3, "New shape should only be (height, width, depth)"
    old_size = patch_embed.shape[-3:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed

class DynamicPatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 8),
            in_chans: int = 1,
            embed_dim: int = 768,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = None, # Optional, default None
            norm_layer: Optional[Callable] = None,
            flatten: bool = False, # Default False to match simple PatchEmbed behavior for Primus
            bias: bool = True,
            strict_img_size: bool = False, # Default False for flexibility
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def _init_img_size(self, img_size: Union[int, Tuple[int, int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_3tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        return img_size, grid_size, num_patches
    
    def set_input_size(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = None,
            patch_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_3tuple(patch_size)
        if new_patch_size is not None and new_patch_size != self.patch_size:
            with torch.no_grad():
                new_proj = nn.Conv3d(
                    self.proj.in_channels,
                    self.proj.out_channels,
                    kernel_size=new_patch_size,
                    stride=new_patch_size,
                    bias=self.proj.bias is not None,
                )
                new_proj.weight.copy_(resample_patch_embed(self.proj.weight, new_patch_size))
                if self.proj.bias is not None:
                    new_proj.bias.copy_(self.proj.bias)
                self.proj = new_proj
            self.patch_size = new_patch_size
        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
    
    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size
        
    def dynamic_feat_size(self, img_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """ Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return (
                math.ceil(img_size[0] / self.patch_size[0]), 
                math.ceil(img_size[1] / self.patch_size[1]),
                math.ceil(img_size[2] / self.patch_size[2]),
            )
        else:
            return (
                img_size[0] // self.patch_size[0], 
                img_size[1] // self.patch_size[1],
                img_size[2] // self.patch_size[2],
            )

    def forward(self, x):
        _, _, H, W, D = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
                assert D == self.img_size[2], f"Input depth ({D}) doesn't match model ({self.img_size[2]})."
            elif not self.dynamic_img_pad:
                assert H % self.patch_size[0] == 0, \
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert W % self.patch_size[1] == 0, \
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                assert D % self.patch_size[2] == 0, \
                    f"Input depth ({D}) should be divisible by patch size ({self.patch_size[2]})."
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            pad_d = (self.patch_size[2] - D % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHWD -> NLC
        x = self.norm(x)
        return x
