# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Type

import torch
import torch.nn as nn
from torch.nn import LayerNorm


class PatchMerging(nn.Module):
    '''The `PatchMerging` module previously defined in v0.9.0.'''

    def __init__(
        self,
        input_resolution,
        dim:          int,
        norm_layer:   Type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3
    ) -> None:
        '''
        Args:
            input_resolution: resolution of input feature maps.
            dim:              number of feature channels.
            norm_layer:       normalization layer.
            spatial_dims:     number of spatial dims.
        '''

        super().__init__()
        assert spatial_dims == 3, 'PatchMerging supports spatial_dims 3 only'

        self.dim = dim
        self.input_resolution = input_resolution
        D, H, W = self.input_resolution
        self.merge_D = D % 2 == 0
        self.merge_H = H % 2 == 0
        self.merge_W = W % 2 == 0

        dims_tmp = dim
        self.resample_scale = [1, 1, 1]
        if self.merge_D:
            dims_tmp *= 2
            self.resample_scale[0] = 2
        if self.merge_H:
            dims_tmp *= 2
            self.resample_scale[1] = 2
        if self.merge_W:
            dims_tmp *= 2
            self.resample_scale[2] = 2

        self.output_resolution = [
            i // s for i, s in
            zip(input_resolution, self.resample_scale)
        ]

        self.norm = norm_layer(dims_tmp)
        self.output_dims = 2 * dim
        self.reduction = nn.Linear(dims_tmp, self.output_dims, bias=False)

    def forward(self, x):
        # x: (B, D, H, W, C)

        if self.merge_D and self.merge_H and self.merge_W:
            x = torch.cat([
                x[:, 0::2, 0::2, 0::2, :],
                x[:, 1::2, 0::2, 0::2, :],
                x[:, 0::2, 1::2, 0::2, :],
                x[:, 0::2, 0::2, 1::2, :],
                x[:, 1::2, 1::2, 0::2, :],
                x[:, 1::2, 0::2, 1::2, :],
                x[:, 0::2, 1::2, 1::2, :],
                x[:, 1::2, 1::2, 1::2, :],
            ], dim=-1)  # x: (B, D//2, H//2, W//2, 8C)
        elif (not self.merge_D) and self.merge_H and self.merge_W:
            x = torch.cat([
                x[:, :, 0::2, 0::2, :],
                x[:, :, 1::2, 0::2, :],
                x[:, :, 0::2, 1::2, :],
                x[:, :, 1::2, 1::2, :],
            ], dim=-1)  # x: (B, D, H//2, W//2, 4C)
        elif self.merge_D and (not self.merge_H) and self.merge_W:
            x = torch.cat([
                x[:, 0::2, :, 0::2, :],
                x[:, 1::2, :, 0::2, :],
                x[:, 0::2, :, 1::2, :],
                x[:, 1::2, :, 1::2, :],
            ], dim=-1)  # x: (B, D//2, H, W//2, 4C)
        elif self.merge_D and self.merge_H and (not self.merge_W):
            x = torch.cat([
                x[:, 0::2, 0::2, :, :],
                x[:, 1::2, 0::2, :, :],
                x[:, 0::2, 1::2, :, :],
                x[:, 1::2, 1::2, :, :],
            ], dim=-1)  # x: (B, D//2, H//2, W, 4C)
        elif (not self.merge_D) and (not self.merge_H) and self.merge_W:
            x = torch.cat([
                x[:, :, :, 0::2, :],
                x[:, :, :, 1::2, :],
            ], dim=-1)  # x: (B, D, H, W//2, 2C)
        elif (not self.merge_D) and self.merge_H and (not self.merge_W):
            x = torch.cat([
                x[:, :, 0::2, :, :],
                x[:, :, 1::2, :, :],
            ], dim=-1)  # x: (B, D, H//2, W, 2C)
        elif self.merge_D and (not self.merge_H) and (not self.merge_W):
            x = torch.cat([
                x[:, 0::2, :, :, :],
                x[:, 1::2, :, :, :],
            ], dim=-1)  # x: (B, D//2, H, W, 2C)
        else:
            pass

        x = self.norm(x)
        x = self.reduction(x)
        # out: (B, D*, H*, W*, 2C)

        return x
