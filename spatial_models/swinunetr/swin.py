import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from copy import deepcopy
from einops import rearrange
from torch.nn import LayerNorm
from typing import Optional, Sequence, Type

from spatial_models.swinunetr.utils import window_partition, window_reverse, get_window_size, compute_mask

from monai.networks.blocks import PatchEmbed
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.layers import DropPath
from spatial_models.swinunetr.WindowAttentionV1 import WindowAttentionV1
from spatial_models.swinunetr.WindowAttentionV2 import WindowAttentionV2
from spatial_models.swinunetr.AutoregressiveAttention import AutoregressiveAttention
from spatial_models.swinunetr.PatchMerging import PatchMerging

class SwinTransformerBlock(nn.Module):
    '''
    Swin Transformer block based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        dim:            int,
        num_heads:      int,
        window_size:    Sequence[int],
        shift_size:     Sequence[int],
        mlp_ratio:      float = 4.0,
        qkv_bias:       bool = True,
        drop:           float = 0.0,
        attn_drop:      float = 0.0,
        attn_version:   str = 'v2',
        drop_path:      float = 0.0,
        act_layer:      str = 'GELU',
        norm_layer:     Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        '''
        Args:
            dim: number     of feature channels.
            num_heads:      number of attention heads.
            window_size:    local window size.
            shift_size:     window shift size.
            mlp_ratio:      ratio of mlp hidden dim to embedding dim.
            qkv_bias:       add a learnable bias to query, key, value.
            drop:           dropout rate.
            attn_drop:      attention dropout rate.
            drop_path:      stochastic depth rate.
            act_layer:      activation layer.
            norm_layer:     normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        '''

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)

        self.attn_version = attn_version
        if attn_version == 'v1':
            self.attn = WindowAttentionV1(
                dim,
                window_size = self.window_size,
                num_heads   = num_heads,
                qkv_bias    = qkv_bias,
                attn_drop   = attn_drop,
                proj_drop   = drop,
            )
        elif attn_version == 'v2':
            self.attn = WindowAttentionV2(
                dim,
                window_size = self.window_size,
                num_heads   = num_heads,
                qkv_bias    = qkv_bias,
                attn_drop   = attn_drop,
                proj_drop   = drop,
            )
        elif attn_version == 'ar':
            self.attn = AutoregressiveAttention(
                dim,
                window_size = self.window_size,
                num_heads   = num_heads,
                qkv_bias    = qkv_bias,
                attn_drop   = attn_drop,
                proj_drop   = drop,
            )
        else:
            raise ValueError('unknown attn_version')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            hidden_size  = dim,
            mlp_dim      = mlp_hidden_dim,
            act          = act_layer,
            dropout_rate = drop,
            dropout_mode = 'swin'
        )

    def forward_part1(self, x, mask_matrix, temp_attn_mask):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        # For first SwinTransformer Block there is no shift and mask, but the second one has shift then has image mask
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # Shift_x is the x after cyclic shifting
        x_windows = window_partition(shifted_x, window_size)
        if self.attn_version == 'ar':
            attn_windows = self.attn(x_windows, mask=attn_mask, temp_attn_mask=temp_attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix, temp_attn_mask):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix, temp_attn_mask)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x

class BasicLayer(nn.Module):
    '''
    Basic Swin Transformer layer in one stage based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        input_resolution: Sequence[int],
        dim:              int,
        depth:            int,
        num_heads:        int,
        window_size:      Sequence[int],
        drop_path:        list,
        mlp_ratio:        float = 4.0,
        qkv_bias:         bool = False,
        drop:             float = 0.0,
        attn_drop:        float = 0.0,
        attn_version:     str = 'v2',
        norm_layer:       Type[LayerNorm] = nn.LayerNorm,
        downsample:       Optional[nn.Module] = None,
        use_checkpoint:   bool = False,
    ) -> None:
        '''
        Args:
            input_resolution: resolution of input feature maps.
            dim:              number of feature channels.
            depth:            number of layers in each stage.
            num_heads:        number of attention heads.
            window_size:      local window size.
            drop_path:        stochastic depth rate.
            mlp_ratio:        ratio of mlp hidden dim to embedding dim.
            qkv_bias:         add a learnable bias to query, key, value.
            drop:             dropout rate.
            attn_drop:        attention dropout rate.
            norm_layer:       normalization layer.
            downsample:       an optional downsampling layer at the end of the layer.
            use_checkpoint:   use gradient checkpointing for reduced memory usage.
        '''

        super().__init__()
        self.window_size = window_size
        # use shift_size if i==1 otherwise no_shift
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim            = dim,
                    num_heads      = num_heads,
                    window_size    = self.window_size,
                    shift_size     = self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio      = mlp_ratio,
                    qkv_bias       = qkv_bias,
                    drop           = drop,
                    attn_drop      = attn_drop,
                    attn_version   = attn_version,
                    drop_path      = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer     = norm_layer,
                    use_checkpoint = use_checkpoint,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(
                input_resolution = input_resolution,
                dim              = dim,
                norm_layer       = norm_layer,
                spatial_dims     = len(self.window_size)
            )

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            # If window_size == x_shape then the shift_size = 0 in that dimension.
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = rearrange(x, 'b c d h w -> b d h w c')
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask, temp_attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask, temp_attn_mask)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, 'b d h w c -> b c d h w')

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, 'b c h w -> b h w c')
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask, temp_attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask, temp_attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, 'b h w c -> b c h w')
        return x


class SwinTransformer(nn.Module):
    '''
    Swin Transformer based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        image_size:     Sequence[int],
        in_chans:       int,
        embed_dim:      int,
        window_size:    Sequence[int],
        patch_size:     Sequence[int],
        depths:         Sequence[int],
        num_heads:      Sequence[int],
        mlp_ratio:      float = 4.0,
        qkv_bias:       bool = True,
        drop_rate:      float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        attn_version:   str = 'v2',
        norm_layer:     Type[LayerNorm] = nn.LayerNorm,
        patch_norm:     bool = False,
        use_checkpoint: bool = False,
        spatial_dims:   int = 3,
    ) -> None:
        '''
        Args:
            image_size:     dimension of input image.
            in_chans:       dimension of input channels.
            embed_dim:      number of linear projection output channels.
            window_size:    local window size.
            patch_size:     patch size.
            depths:         number of layers in each stage.
            num_heads:      number of attention heads.
            mlp_ratio:      ratio of mlp hidden dim to embedding dim.
            qkv_bias:       add a learnable bias to query, key, value.
            drop_rate:      dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer:     normalization layer.
            patch_norm:     add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims:   spatial dimension.
        '''

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size   = self.patch_size,
            in_chans     = in_chans,
            embed_dim    = embed_dim,
            norm_layer   = norm_layer if self.patch_norm else None,
            spatial_dims = spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        self.resamples = []

        input_size = deepcopy(image_size)
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                input_resolution = input_size,
                dim              = int(embed_dim * 2**i_layer),
                depth            = depths[i_layer],
                num_heads        = num_heads[i_layer],
                window_size      = self.window_size,
                drop_path        = dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio        = mlp_ratio,
                qkv_bias         = qkv_bias,
                drop             = drop_rate,
                attn_drop        = attn_drop_rate,
                attn_version     = attn_version,
                norm_layer       = norm_layer,
                downsample       = PatchMerging,
                use_checkpoint   = use_checkpoint,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

            input_size = layer.downsample.output_resolution
            self.resamples.insert(0, layer.downsample.resample_scale)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, 'n c d h w -> n d h w c')
                x = F.layer_norm(x, [ch])
                x = rearrange(x, 'n d h w c -> n c d h w')
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, 'n c h w -> n h w c')
                x = F.layer_norm(x, [ch])
                x = rearrange(x, 'n h w c -> n c h w')
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]