import torch
import torch.nn as nn
import numpy as np
from utils.viz import cube_3d_viewer

from models.exomild.psf_sampler import (
    BatchPSFSampler,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Injector(nn.Module):
    def __init__(self, psf_size, H):
        super().__init__()
        self.psf_size = psf_size
        self.H = H

        self.psf_sampler_shift = BatchPSFSampler(
            psf_size=self.psf_size, mode="shift"
        )

        assert self.psf_size % 2 == 0
        # self.margin = self.psf_size + 2
        # self.margin = self.psf_size + 4
        self.margin = self.psf_size + 16

        self.size_ext = self.H + 2 * self.margin

        yy, xx = np.mgrid[: self.psf_size, : self.psf_size]
        yy = torch.tensor(yy, dtype=torch.int64).view(
            1, 1, self.psf_size, self.psf_size
        )
        xx = torch.tensor(xx, dtype=torch.int64).view(
            1, 1, self.psf_size, self.psf_size
        )

        self.register_buffer("yy", yy, persistent=False)
        self.register_buffer("xx", xx, persistent=False)

    def forward(self, coords_t, alphas, psf_params):
        bs, n_sources, T, _ = coords_t.shape
        bs, n_sources = alphas.shape
        assert bs == 1
        # print("a")

        obj_ext = torch.zeros(
            (1, T, self.size_ext, self.size_ext),
            dtype=torch.float32,
            device=device,
        )

        # (1, T, H+, W+)

        coords_t_int = coords_t.round().long()
        # (bs, n_sources, T, 2)
        # print("b")

        idx_t = (
            torch.arange(T)
            .long()
            .view(1, T, 1, 1)
            .expand(-1, -1, self.psf_size, self.psf_size)
        )
        # (1, T, ps, ps)

        shift = coords_t - coords_t_int + self.psf_size // 2
        # (bs, n_sources, T, 2)

        # (bs, n_sources, T, ps, ps)
        y0 = shift[:, :, :, 0].view(bs, -1)
        x0 = shift[:, :, :, 1].view(bs, -1)
        # (bs, n_sources, T)

        # print("c")
        psf, _ = self.psf_sampler_shift(x0=x0, y0=y0, **psf_params)
        # (bs, n_sources * T, ps, ps)
        # print("d")

        psf = psf.view(bs, n_sources, T, self.psf_size, self.psf_size)
        # (bs, n_sources, T, ps, ps)
        torch.cuda.synchronize()
        # if (coords_t_int > 255).any():
        # breakpoint()
        # if (coords_t_int < 0).any():
        # breakpoint()

        for i in range(n_sources):
            # print("e")
            # coords_t_int_i = coords_t_int[:, i].long() + self.psf_size // 2
            coords_t_int_i = coords_t_int[:, i].long() - self.psf_size // 2 + self.margin
            # coords_t_int_i = coords_t_int[:, i].long() + self.margin
            # coords_t_int_i = coords_t_int[:, i].long() + self.psf_size // 2 - self.margin
            # (bs, T, 2)
            # print("f")

            idx_y = coords_t_int_i[0, :, 0].view(1, T, 1, 1) + self.yy
            idx_x = coords_t_int_i[0, :, 1].view(1, T, 1, 1) + self.xx
            # (bs, T, ps, ps)
            if (idx_y >= obj_ext.shape[-1]).any():
                breakpoint()
            if (idx_x >= obj_ext.shape[-1]).any():
                breakpoint()
            if (idx_y < 0).any():
                breakpoint()
            if (idx_x < 0).any():
                breakpoint()
            # print(f"{idx_x=}")
            # print(f"{idx_y=}")
            torch.cuda.synchronize()
            # print("g")

            psf_i = alphas[0, i] * psf[:, i, :, :, :]
            # (bs, T, ps, ps)
            # print("h")

            try:
                obj_ext[
                    :,
                    idx_t.flatten().long(),
                    idx_y.flatten().long(),
                    idx_x.flatten().long(),
                ] += psf_i.flatten()
            except:
                breakpoint()
            # print("i")
        obj = obj_ext[
            :,
            :,
            self.margin : -self.margin,
            self.margin : -self.margin,
        ]
        if False:
            breakpoint()
            cube_3d_viewer(obj.cpu().numpy()[0], range_hist="auto", quantile=0)
        return obj
