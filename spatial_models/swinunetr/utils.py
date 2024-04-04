import torch


def window_partition(x, window_size):
    '''window partition operation based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        x:           input tensor.
        window_size: local window size.
    '''
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    '''window reverse operation based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        windows:     windows tensor.
        window_size: local window size.
        dims:        dimension values.
    '''
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

def get_temporal_mask(dims, window_size, device):
    d, h, w = dims
    temporal_diag = torch.tril(torch.ones(d, d, device=device))
    temp_mask_windows = []
    for i in range(d):
        # 128*128*8*8 image masks ti to tj
        temp_mask_imgs_ti = temporal_diag[i, :].repeat((1, h, w, 1, 1)).permute(0, 4, 1, 2, 3)
        temp_mask_windows_ti = window_partition(temp_mask_imgs_ti, window_size)
        temp_mask_windows.append(temp_mask_windows_ti.expand(-1, -1, window_size[1] * window_size[2]))
    temp_mask_windows = torch.cat(temp_mask_windows, axis=2)
    return temp_mask_windows

def get_window_size(x_size, window_size, shift_size=None):
    '''Computing window size based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        x_size:      input size.
        window_size: local window size.
        shift_size:  window shifting size.
    '''

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

def get_temporal_mask(dims, window_size, device):
    d, h, w = dims
    temporal_diag = torch.triu(torch.ones(d, d, device=device))
    temp_mask_windows = []
    for i in range(d):
        # 128*128*8*8 image masks ti to tj
        temp_mask_imgs_ti = temporal_diag[i, :].repeat((1, h, w, 1, 1)).permute(0, 4, 1, 2, 3)
        temp_mask_windows_ti = window_partition(temp_mask_imgs_ti, window_size)
        temp_mask_windows.append(temp_mask_windows_ti.expand(-1, -1, window_size[1] * window_size[2]))
    temp_mask_windows = torch.cat(temp_mask_windows, axis=2)
    return temp_mask_windows

def compute_mask(dims, window_size, shift_size, device):
    '''Computing region masks based on: 'Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>'
    https://github.com/microsoft/Swin-Transformer

    Args:
        dims:        dimension values.
        window_size: local window size.
        shift_size:  shift size.
        device:      device.
    '''

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        temp_attn_mask = get_temporal_mask(dims=dims, window_size=window_size, device=device)
        temp_attn_mask = temp_attn_mask.masked_fill(temp_attn_mask == 0, float(-100.0)).masked_fill(temp_attn_mask > 0, float(0.0))
    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        temp_attn_mask = None
    # TODO: Add temporal attention mask here

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask, temp_attn_mask