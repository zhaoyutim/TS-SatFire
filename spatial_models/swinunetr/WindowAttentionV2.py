from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttentionV2(nn.Module):
    '''
    Window based multi-head self attention module with relative position bias based on: 'Liu et al.,
    Swin Transformer V2: Scaling Up Capacity and Resolution
    <https://arxiv.org/abs/2111.09883>'
    https://github.com/microsoft/Swin-Transformer
    '''

    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        window_size: Sequence[int],
        qkv_bias:    bool = False,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ) -> None:
        '''
        Args:
            dim: number  of feature channels.
            num_heads:   number of attention heads.
            window_size: local window size.
            qkv_bias:    add a learnable bias to query, key, value.
            attn_drop:   attention dropout rate.
            proj_drop:   dropout rate of output.
        '''

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        mesh_args = torch.meshgrid.__kwdefaults__

        scale_params = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(scale_params, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        if len(self.window_size) == 3:

            # get relative_coords_table
            relative_coords_d = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_h = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
            if mesh_args is not None:
                relative_coords_table = torch.stack(torch.meshgrid(relative_coords_d, relative_coords_h, relative_coords_w, indexing='ij'))
            else:
                relative_coords_table = torch.stack(torch.meshgrid(relative_coords_d, relative_coords_h, relative_coords_w))
            relative_coords_table = relative_coords_table.permute(1, 2, 3, 0).contiguous().unsqueeze(0)
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
            relative_coords_table[:, :, :, 2] /= (self.window_size[2] - 1)

            # get pair-wise relative position index for each token inside the window
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        elif len(self.window_size) == 2:

            # get relative_coords_table
            relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            if mesh_args is not None:
                relative_coords_table = torch.stack(torch.meshgrid(relative_coords_h, relative_coords_w, indexing='ij'))
            else:
                relative_coords_table = torch.stack(torch.meshgrid(relative_coords_h, relative_coords_w))
            relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        device = x.device

        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        max_scale = torch.log(torch.tensor(1.0 / 0.01, device=device))
        logit_scale = torch.clamp(self.logit_scale, max=max_scale).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
