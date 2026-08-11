import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)

        # position = torch.arange(max_len).unsqueeze(1)
        div_term = (
            torch.exp(torch.arange(d_model) * (-math.log(10000.0) / d_model))
            .view(1, -1, 1)
            .float()
        )
        # pe = torch.zeros(max_len, 1, d_model)
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        # self.register_buffer("pe", pe)
        self.register_buffer("div_term", div_term, persistent=False)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        bs, _, hw = x.shape

        x_div = x * self.div_term
        pe_sin = torch.sin(x_div)
        pe_cos = torch.cos(x_div)

        pe = torch.cat([pe_sin, pe_cos], dim=1)
        # (bs, 2 * d_model, hw)
        bs, _, hw = x.shape

        return pe


def agg_dict(d_split):
    d_agg = {}
    keys = list(d_split[0].keys())
    for k in keys:
        d_agg[k] = torch.cat([sub_d.pop(k) for sub_d in d_split])
    return d_agg


def split_dict(d, batch_size):
    d_split = []
    n_splits = None
    # breakpoint()
    for k, v in d.items():
        # print(k)
        # if v is not None:
        # print(f"{v.shape=}")

        if v is None:
            v_split = [None] * n_splits
        elif k in ['amplitude', 'sigma_x', 'sigma_y', 'theta', 'offset']:
            v_split = [v] * n_splits
        elif v.numel() == 1:
            v_split = [v] * n_splits
        else:
            v_split = torch.split(v, batch_size)
            n_splits = len(v_split)

        if len(d_split) == 0:
            d_split = [{k: v_split[i]} for i in range(n_splits)]
        else:
            for i in range(n_splits):
                d_split[i][k] = v_split[i]
    return d_split


def mask_to_coords(snr, mask, n_sources_max):
    bs, H, W = snr.shape
    # bs, H, W = alpha.shape
    yy, xx = np.mgrid[:H, :W]

    yy = torch.tensor(yy, device=device).view(H * W)
    xx = torch.tensor(xx, device=device).view(H * W)

    snr_masked = (snr * mask).view(bs, -1)

    idx_sorted = torch.argsort(snr_masked, dim=1, descending=True)
    # (bs, H * W)

    idx_sorted = idx_sorted[:, :n_sources_max]
    # (bs, H * W)

    coords_y = yy[idx_sorted]
    coords_x = xx[idx_sorted]
    # (bs, n_sources_max)

    return coords_y, coords_x

    # snr_sorted = torch.gather(snr_masked, dim=1, index=idx_sorted)
    # # (bs, n_sources_max)

    # alpha_sorted = torch.gather(alpha.view(bs, H * W), dim=1, index=idx_sorted)
    # # (bs, n_sources_max)

    # z_init = torch.stack([alpha_sorted, coords_y, coords_x], dim=-1)
    # # (bs, n_sources, 3)

    # return snr_sorted, z_init


def interpolate_2d(x, coords):
    # print(f"############## CHECK THAT THIS WORKS AS INTENDED ########")
    C, H, W = x.shape
    n_coords, _ = coords.shape
    # NOTE: dimension order: (y, x)

    grid = 2 * coords.view(1, 1, n_coords, 2) / (H - 1) - 1
    # (1, 1, n_coords, 2)

    # IMPORTANT: flip coords to make them (x, y)
    grid = grid.flip(dims=[-1])

    out = F.grid_sample(
        x[None, ...],
        grid=grid,
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    )
    # (1, C, 1, n_coords)

    out = out.view(C, n_coords)

    out = out.permute(1, 0)
    # (n_coords, C)

    return out
