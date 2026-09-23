import torch
import numpy as np
import einops
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.exomild.utils import agg_dict
from datasets.transforms.injection import fit_gaussian_2d
from utils.viz import cube_3d_viewer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_patches_3d(x, kernel_size, padding=0, stride=1, dilation=1):
    # x: (B, C, D, H, W)
    bs, C, T, H, W = x.shape
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if kernel_size[0] is None:
        kernel_size = (T, kernel_size[1], kernel_size[2])
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(
        dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1
    ):
        dim_out = (
            dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1
        ) // dim_stride + 1
        return dim_out

    channels = x.shape[1]

    d_dim_in = x.shape[2]
    h_dim_in = x.shape[3]
    w_dim_in = x.shape[4]
    d_dim_out = get_dim_blocks(
        d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0]
    )
    h_dim_out = get_dim_blocks(
        h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1]
    )
    w_dim_out = get_dim_blocks(
        w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2]
    )
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    # (B, C, D, H, W)
    x = x.view(-1, channels, d_dim_in, h_dim_in * w_dim_in)
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(
        x,
        kernel_size=(kernel_size[0], 1),
        padding=(padding[0], 0),
        stride=(stride[0], 1),
        dilation=(dilation[0], 1),
    )
    # (B, C * kernel_size[0], d_dim_out * H * W)

    x = x.view(-1, channels * kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(
        x,
        kernel_size=(kernel_size[1], kernel_size[2]),
        padding=(padding[1], padding[2]),
        stride=(stride[1], stride[2]),
        dilation=(dilation[1], dilation[2]),
    )
    # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

    x = x.view(
        -1,
        channels,
        kernel_size[0],
        d_dim_out,
        kernel_size[1],
        kernel_size[2],
        h_dim_out,
        w_dim_out,
    )
    # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.contiguous().view(
        -1, channels, kernel_size[0], kernel_size[1], kernel_size[2]
    )
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    return x


def combine_patches_3d(
    x, kernel_size, output_shape, padding=0, stride=1, dilation=1
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(
        dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1
    ):
        dim_out = (
            dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1
        ) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    d_dim_in = get_dim_blocks(
        d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0]
    )
    h_dim_in = get_dim_blocks(
        h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1]
    )
    w_dim_in = get_dim_blocks(
        w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2]
    )
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    x = x.view(
        -1,
        channels,
        d_dim_in,
        h_dim_in,
        w_dim_in,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
    )
    # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

    x = x.contiguous().view(
        -1,
        channels * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2],
        h_dim_in * w_dim_in,
    )
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(
        x,
        output_size=(h_dim_out, w_dim_out),
        kernel_size=(kernel_size[1], kernel_size[2]),
        padding=(padding[1], padding[2]),
        stride=(stride[1], stride[2]),
        dilation=(dilation[1], dilation[2]),
    )
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(
        x,
        output_size=(d_dim_out, h_dim_out * w_dim_out),
        kernel_size=(kernel_size[0], 1),
        padding=(padding[0], 0),
        stride=(stride[0], 1),
        dilation=(dilation[0], 1),
    )
    # (B, C, D, H * W)

    x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, D, H, W)

    return x


class WavelengthAligner(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = None
        pass

    def init(self, C, H, device):
        yyxx = np.mgrid[:H, :H]
        yyxx = 2 * yyxx / (H - 1) - 1
        yyxx = torch.tensor(yyxx, device=device, dtype=torch.float32)[
            None, ...
        ]
        # (1, 2, H, H)

        # NOTE: pb inversion xy
        # self.grid = torch.flip(grid, dims=(3,))

        self.grid = yyxx.permute(0, 2, 3, 1).to(device)
        # grid = yyxx.permute(0, 2, 3, 1)
        # (1, H, H, 2)

        # self.offset = torch.tensor(
        # 1 / (H - 1), device=device, dtype=torch.float32
        # )
        offset = torch.tensor(1 / (H - 1), dtype=torch.float32)

        self.register_buffer("offset", offset, persistent=False)
        # self.register_buffer("grid", grid, persistent=False)


    def forward(self, x, lbda, mode="align"):
        if mode == "align":
            return self.align(x, lbda=lbda)
        elif mode == "unalign":
            return self.unalign(x, lbda=lbda)
        else:
            raise ValueError

    def interpolate(self, x, coeff):
        bs, C, T, H, W = x.shape
        assert bs == 1
        # NOTE: bug in this, frame will be rotated and mirrored
        # x_in = x.clone()
        x = rearrange(x, "b c t h w -> c (b t) h w")
        # (C, bs * T, H, W)

        # bs, C = lbda.shape
        # lbda_max = torch.amax(lbda, dim=1, keepdim=True)
        # coeff = lbda / lbda_max
        coeff = coeff.view(C, 1, 1, 1).float()

        if self.grid is None:
            self.init(C=C, H=H, device=x.device)

        grid = self.offset + (self.grid - self.offset) * coeff

        # breakpoint()
        x = F.grid_sample(
            x,
            grid=grid,
            align_corners=True,
            mode="bicubic",
            padding_mode="border",
        )

        x = rearrange(x, "c (b t) h w ->  b c t h w", b=bs, t=T)
        # (bs, C, T, H, W)
        # if True:
        # breakpoint()

        return x.permute(0,1,2,4,3)

    def align(self, x, lbda):
        bs, C = lbda.shape
        lbda_max = torch.amax(lbda, dim=1, keepdim=True)
        coeff = lbda / lbda_max

        return self.interpolate(x, coeff)

    def unalign(self, x, lbda, force_positive):
        bs, C = lbda.shape
        lbda_max = torch.amax(lbda, dim=1, keepdim=True)
        coeff = lbda / lbda_max
        if force_positive:
            return self.interpolate(torch.sqrt(x), 1 / coeff) ** 2
        return self.interpolate(x, 1 / coeff)

    # def align(self, x, lbda):
    # bs, C, T, H, W = x.shape
    # assert bs == 1

    # # cube_3d_viewer(x[0, :, 0].cpu().numpy(), range_hist="auto")

    # x = rearrange(x, "b c t h w -> c (b t) h w")
    # # (C, bs * T, H, W)

    # bs, C = lbda.shape
    # lbda_max = torch.amax(lbda, dim=1, keepdim=True)
    # coeff = lbda / lbda_max

    # coeff = coeff.view(C, 1, 1, 1)

    # if self.grid is None:
    # self.init(C=C, H=H)

    # grid = self.offset + (self.grid - self.offset) * coeff

    # x = F.grid_sample(x, grid=grid, align_corners=True, mode="bicubic")

    # x = rearrange(x, "c (b t) h w ->  b c t h w", b=bs, t=T)
    # # (bs, C, T, H, W)

    # # cube_3d_viewer(x[0, :, 0].cpu().numpy(), range_hist="auto")
    # # cube_3d_viewer(x.view(-1, H, W).cpu().numpy(), range_hist="auto")

    # return x

    # def unalign(self, x, lbda):
    # bs, C, T, H, W = x.shape
    # assert bs == 1

    # x = rearrange(x, "b c t h w -> c (b t) h w")
    # # (C, bs * T, H, W)

    # pass


class PatchExtractor(nn.Module):
    def __init__(
        self,
        dim_t,
        dilation_t,
        stride_t,
        dim_x,
        dilation_x,
        stride_x,
        symmetry,
    ):
        super().__init__()
        self.dim_t = dim_t
        self.dilation_t = dilation_t
        self.stride_t = stride_t

        self.dim_x = dim_x
        self.dilation_x = dilation_x
        self.stride_x = stride_x

        self.symmetry = symmetry
        assert self.symmetry in [1, 2, 4]
        # self.kernel_size = (self.dim_t, self.dim_x, self.dim_x)
        self.kernel_size_1 = (
            2 * self.dim_t * self.dilation_t,
            self.dim_x * self.dilation_x,
            self.dim_x * self.dilation_x,
        )
        self.kernel_size_2 = (
            self.dim_t,
            self.dim_x,
            self.dim_x,
        )
        self.dilation_1 = (1, 1, 1)
        self.dilation_2 = (self.dilation_t, self.dilation_x, self.dilation_x)

        self.stride_1 = (self.stride_t, self.stride_x, self.stride_x)
        self.stride_2 = (1, 1, 1)

        self.bilevel = self.dilation_x > 1 or self.dilation_t > 1

        # self.unfolder = nn.Unfold(
        # kernel_size=,
        # dilation=(self.dilation_t, self.dilation_x, self.dilation_x),
        # stride=(self.stride_t, self.stride_x, self.stride_x),
        # )
        self.div_1_x = self.div_1_t = None
        self.div_2_x = self.div_2_t = None

        self.wavelength_aligner = WavelengthAligner()

    def get_divisors(self, level):
        if level == 1:
            # T_in = self.T_2
            # H_in = self.H_2
            T = self.T_1
            H = self.H_1
            ks_t = self.kernel_size_1[0]
            ks_x = self.kernel_size_1[1]
            s_t = self.stride_1[0]
            s_x = self.stride_1[1]
            d_t = self.dilation_1[0]
            d_x = self.dilation_1[1]
        elif level == 2:
            # T_in = self.T_3
            T = self.T_2
            H = self.H_2
            ks_t = self.kernel_size_2[0]
            ks_x = self.kernel_size_2[2]
            s_t = self.stride_2[0]
            s_x = self.stride_2[2]
            d_t = self.dilation_2[0]
            d_x = self.dilation_2[2]
        else:
            raise ValueError()

        # ones_x = torch.ones((1, H_in), device=device)
        ones_x = torch.ones((1, 1, 1, H), device=device)
        ones_x_p = F.unfold(
            ones_x,
            # output_size=(1, H_out),
            kernel_size=(1, ks_x),
            dilation=(1, d_x),
            stride=(1, s_x),
        )
        # (1, ks_x, n_patches)
        div_x = F.fold(
            ones_x_p,
            output_size=(1, H),
            kernel_size=(1, ks_x),
            dilation=(1, d_x),
            stride=(1, s_x),
        )
        div_x = 1 / div_x.flatten()

        ones_t = torch.ones((1, 1, 1, T), device=device)
        ones_t_p = F.unfold(
            ones_t,
            # output_size=(1, H_out),
            kernel_size=(1, ks_t),
            dilation=(1, d_t),
            stride=(1, s_t),
        )
        # (1, ks_x, n_patches)
        div_t = F.fold(
            ones_t_p,
            output_size=(1, T),
            kernel_size=(1, ks_t),
            dilation=(1, d_t),
            stride=(1, s_t),
        )
        div_t = 1 / div_t.flatten()

        return div_t, div_x

    def align_wavelengths(self, x, lbda):
        bs, L, T, H, W = x.shape

        return x

    def align_symmetry(self, x):
        bs, L, T, H, W = x.shape
        if self.symmetry > 1:
            pad = (0, 1, 0, 1)
            x_pad = F.pad(x.view(-1, H, W), pad=pad, mode="replicate")
            # (bs, L, T, H + 1, W + 1)
            x_pad = x_pad.view(bs, L, T, H + 1, W + 1)

            x_pad_2 = torch.rot90(x_pad, k=2, dims=(-2, -1))
            # (bs, L, T, H + 1, W + 1)

            all_x_pad = [x_pad, x_pad_2]
            if self.symmetry == 4:
                x_pad_1 = torch.rot90(x_pad, k=1, dims=(-2, -1))
                # (bs, L, T, H + 1, W + 1)

                x_pad_3 = torch.rot90(x_pad, k=3, dims=(-2, -1))
                # (bs, L, T, H + 1, W + 1)
                all_x_pad += [x_pad_1, x_pad_3]

            x = torch.stack(all_x_pad, dim=2)
            # (bs, L, S, T, H + 1, W + 1)

            x = x[:, :, :, :, :-1, :-1]
            # (bs, L, S, T, H, W)
        else:
            x = x.unsqueeze(2)
            # (bs, L, S, T, H, W)

        return x

    def forward(self, x, mode, lbda=None):
        if mode == "unfold":
            return self.unfold(x, lbda=lbda)
        elif mode == "fold":
            return self.fold(x, lbda=lbda)
        else:
            raise ValueError()

    def unfold(self, x, lbda):
        bs, L, T, H, W = x.shape

        # x = self.align_wavelengths(x, lbda)
        # # (bs, L, T, H, W)
        if L > 1:
            x = self.wavelength_aligner(x, lbda=lbda, mode="align")
            # # (bs, L, T, H, W)

        if False:
            # cube_3d_viewer(xp[bsp // 2, 0, 0].detach().numpy(), range_hist="auto")
            cube_3d_viewer(x[0, :, 0].detach().numpy(), range_hist="auto")
            breakpoint()

        x = self.align_symmetry(x)
        # (bs, L, S, T, H, W)

        # x = x.view(bs, -1, T, H, W)
        x = x.view(bs, 1, -1, H, W)
        self.bs_1, self.C_1, self.T_1, self.H_1, self.W_1 = x.shape
        # breakpoint()

        # xp = nn.
        # xp = self.unfolder(x.view(-1, T, H, W))
        # (bs * C, t h w, n_patches)
        # TODO: there is a bug in the channel extraction,
        # put everything into the depth dimension
        xp = extract_patches_3d(
            x.contiguous(),
            kernel_size=self.kernel_size_1,
            stride=self.stride_1,
            dilation=self.dilation_1,
        )
        # (bsp, c, t, h, w)
        bsp = xp.shape[0]
        print(f"{xp.shape=}")

        # (bsp, l, s, t, h, w)
        bsp, c, t, h, w = xp.shape
        if False:
            # cube_3d_viewer(xp[bsp // 2, 0, 0].detach().numpy(), range_hist="auto")
            cube_3d_viewer(xp[0, 0, :].detach().numpy(), range_hist="auto")
            breakpoint()

        self.bs_2, self.C_2, self.T_2, self.H_2, self.W_2 = xp.shape

        if self.div_1_x is None:
            self.div_1_t, self.div_1_x = self.get_divisors(level=1)

        if self.bilevel:
            # x  = xp.view(bsp, l * s, t, h, w)
            xp = extract_patches_3d(
                xp.contiguous(),
                kernel_size=self.kernel_size_2,
                stride=self.stride_2,
                dilation=self.dilation_2,
            )
            self.bs_3, self.C_3, self.T_3, self.H_3, self.W_3 = xp.shape
            bsp = xp.shape[0]

            if self.div_2_x is None:
                self.div_2_t, self.div_2_x = self.get_divisors(level=2)

        dim_t = self.dim_t if self.dim_t else T
        xp = xp.view(bsp, L, self.symmetry, dim_t, self.dim_x, self.dim_x)

        return xp

    def fold(self, x, lbda):
        bsp, l, s, t, h, w = x.shape

        x = x.view(bsp, l * s, t, h, w)
        # (bsp, c, t, h, w)

        if self.bilevel:
            x = combine_patches_3d(
                x.contiguous(),
                kernel_size=self.kernel_size_2,
                stride=self.stride_2,
                dilation=self.dilation_2,
                output_shape=[
                    self.bs_2,
                    self.C_2,
                    self.T_2,
                    self.H_2,
                    self.W_2,
                ],
            )

            x = x * self.div_2_x.view(1, 1, 1, 1, -1)
            x = x * self.div_2_x.view(1, 1, 1, -1, 1)
            x = x * self.div_2_t.view(1, 1, -1, 1, 1)

        x = combine_patches_3d(
            x.contiguous(),
            kernel_size=self.kernel_size_1,
            stride=self.stride_1,
            dilation=self.dilation_1,
            output_shape=[self.bs_1, self.C_1, self.T_1, self.H_1, self.W_1],
        )
        # (bs, C, T, H, W)

        x = x * self.div_1_x.view(1, 1, 1, 1, -1)
        x = x * self.div_1_x.view(1, 1, 1, -1, 1)
        x = x * self.div_1_t.view(1, 1, -1, 1, 1)

        # assert self.C_1 == 1

        # x = x.view(self.bs_1, self.T_1, self.H_1, self.W_1)

        if l > 1:
            x = self.wavelength_aligner(x, lbda=lbda, mode="unalign")
            # # (bs, L, T, H, W)

        return x


class PatchFormatter(nn.Module):
    def __init__(self, dim_samples, dim_prefeatures):
        super().__init__()
        self.dim_samples = dim_samples
        self.dim_prefeatures = dim_prefeatures
        self.str_unfold = (
            f"b l t s h w -> b ({self.dim_samples}) s ({self.dim_prefeatures})"
        )
        self.str_fold = (
            f"b ({self.dim_samples}) ({self.dim_prefeatures}) -> b l t h w"
        )

    def forward(self, x, mode, lbda=None, **psf_params):
        if mode == "unfold":
            return self.unfold(x)
        elif mode == "fold":
            return self.fold(x)
        elif mode == "psf_params":
            return self.format_psf_params(**psf_params, lbda=lbda)
        else:
            raise ValueError()

    def unfold(self, x):
        # (bsp, s, l, t, h, w)
        bsp, self.l, self.t, self.s, self.h, self.w = x.shape

        x = einops.rearrange(x, self.str_unfold)
        # (bsp, dim_samples, s, dim_prefeatures)

        return x

    def fold(self, x):
        # (bsp, dim_samples, dim_prefeatures)
        bsp, d_s, d_pf = x.shape

        if d_s == 1:
            d_l = 1
            d_t = 1
        else:
            d_l = self.l
            d_t = self.t
        x = einops.rearrange(
            x, self.str_fold, l=d_l, t=d_t, h=self.h, w=self.w
        )
        # bsp, l, s, t, h, w = x.shape
        # bsp, s, l, t, h, w = x.shape
        bsp, l, t, h, w = x.shape

        return x


class PSFFormatterTemporal(nn.Module):
    def __init__(self, H=256, W=256):
        super().__init__()
        self.H = H
        self.W = W

    def fit_psf(self, psf):
        bs, h, w = psf.shape
        psf_params = []
        for _psf in psf:
            _psf_params = fit_gaussian_2d(_psf.detach().cpu().numpy())
            _psf_params["offset"] = 0
            _psf_params.pop("x0")
            _psf_params.pop("y0")
            _psf_params = {
                # k: torch.tensor(v, device=device).view(1, 1)
                k: torch.tensor(v, device=device, dtype=torch.float32).view(1)
                for (k, v) in _psf_params.items()
            }
            psf_params.append(_psf_params)
        psf_params = agg_dict(psf_params)
        self.psf_params = psf_params
        return psf_params

    def forward(
        self, tt, xx, yy, rot, theta, amplitude, sigma_x, sigma_y, **kwargs
    ):
        # (bsp, dim_samples, dim_prefeatures)
        bsp, d_samples, d_prefeatures = tt.shape
        assert tt.shape == xx.shape
        assert tt.shape == yy.shape
        # bsp, _, T = rot.shape
        assert rot.ndim == 3
        # rot = rot[:, 0, :]
        # assert bs == 1

        a = (torch.cos(theta) ** 2) / (2 * sigma_x**2) + (
            torch.sin(theta) ** 2
        ) / (2 * sigma_y**2)
        b = -(torch.sin(2 * theta)) / (4 * sigma_x**2) + (
            torch.sin(2 * theta)
        ) / (4 * sigma_y**2)
        c = (torch.sin(theta) ** 2) / (2 * sigma_x**2) + (
            torch.cos(theta) ** 2
        ) / (2 * sigma_y**2)
        # (bs)

        # trajectories
        # (bsp, dim_samples(x), dim_prefeatures(t), dim_prefeatures(t))
        yy_c = yy - self.H // 2
        xx_c = xx - self.W // 2

        yy_c = yy_c.unsqueeze(-1)
        xx_c = xx_c.unsqueeze(-1)
        # (bsp, d_samples, d_prefeatures, 1)
        # rr_c = torch.sqrt(yy_c ** 2 + xx_c ** 2)
        # rr_c_mean = torch.mean(rr_c, dim=(1, 2))

        # breakpoint()
        rot = torch.deg2rad(rot)
        # (bsp, d_sample, d_prefeatures)

        # rot_t = rot.flatten()[tt.flatten().long()].view(
        # bsp, d_samples, d_prefeatures, 1
        # )

        # rot = rot.view(bsp, 1, 1, T)

        rot_c = rot[:, 0, None, None] - rot[:, :, :, None]
        # (bsp, d_samples, d_prefeatures, d_prefeatures)

        # breakpoint()
        delta_y = torch.cos(rot_c) * yy_c - torch.sin(rot_c) * xx_c - yy_c
        delta_x = torch.sin(rot_c) * yy_c + torch.cos(rot_c) * xx_c - xx_c
        # (bsp, d_sample, d_prefeatures (pos), d_prefeatures)

        exp = torch.exp(
            -(a * delta_x**2 + 2 * b * delta_x * delta_y + c * delta_y**2)
        )
        # (bsp, d_sample, d_prefeatures (pos), d_prefeatures)

        psf = amplitude * exp

        return psf


class TemporalTerm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


def main():
    from scipy.stats import norm
    import os
    from skimage.feature import peak_local_max
    from models.layers.paco_new import PACONew

    path = "data/other/example_2"

    # path_rec = os.path.join(path, "rec_00.pt")
    # path_rec = os.path.join(path, "rec_00.pt")
    # path_input = os.path.join(path, "input.pt")
    # path_psf = os.path.join(path, "psf_00.pt")
    # path_rot = os.path.join(path, "rot_00.pt")
    # path_rec = os.path.join(path, "rec_00.pt")
    path_input = os.path.join(path, "input.pt")
    path_psf = os.path.join(path, "psf.pt")
    path_rot = os.path.join(path, "rot.pt")
    path_s_0 = os.path.join(path, "s_0.pt")

    psf = torch.load(path_psf).to(device)
    s_0 = torch.load(path_s_0).to(device)
    x_in = torch.load(path_input).to(device)
    rot = torch.load(path_rot).to(device) * 2
    # rot = torch.load(path_rot).to(device)
    rot = rot - rot.flatten()[0]
    rot_range = rot.max() - rot.min()
    print(f"{rot_range=}")

    x_in = torch.load(path_input).to(device)
    bs, T, H, W = x_in.shape

    patches_extractor = PatchExtractor(
        dim_t=16,
        dilation_t=1,
        stride_t=16,
        dim_x=8,
        dilation_x=1,
        stride_x=8,
        symmetry=1,
    )
    # formatter = Formatter(dim_samples="t", dim_prefeatures="h w l s")
    # formatter = Formatter(dim_prefeatures="t", dim_samples="h w l s")
    # formatter = Formatter(dim_prefeatures="t", dim_samples="h w l s")
    # formatter = Formatter(dim_samples="h w s", dim_prefeatures="t l")
    formatter = Formatter(dim_samples="h w s", dim_prefeatures="t l")
    psf_formatter = PSFFormatter()
    tt, yy, xx = torch.tensor(np.mgrid[:T, :H, :W], device=device).float()

    rot_cube = torch.ones(bs, 1, T, H, W)
    rot_cube = rot_cube * rot.view(bs, 1, T, 1, 1)

    xp = patches_extractor(x_in.unsqueeze(1))
    xx_p = patches_extractor(xx[None, None, :])
    yy_p = patches_extractor(yy[None, None, :])
    tt_p = patches_extractor(tt[None, None, :])
    rot_p = patches_extractor(rot_cube)

    xp = formatter(xp)

    xx_p = formatter(xx_p)
    yy_p = formatter(yy_p)
    tt_p = formatter(tt_p)
    rot_p = formatter(rot_p)

    psf_params = psf_formatter.fit_psf(psf)

    h_p = psf_formatter(tt_p, xx_p, yy_p, rot=rot_p, **psf_params)

    breakpoint()

    m = torch.mean(xp, dim=2, keepdim=True)
    s = torch.std(xp, dim=2, keepdim=True)
    xp = (xp - m) / s
    print(f"{xp.shape=}")
    idx = 520
    mean = torch.mean(xp, dim=1)[idx]
    plt.plot(xp[idx].T)
    plt.plot(mean, linewidth=5, c="r")
    plt.show()
    breakpoint()
    mean = torch.mean(xp, dim=(1, 2), keepdim=True)
    std = torch.mean(xp, dim=(1, 2), keepdim=True)
    xp = (xp - mean) / std
    cube_3d_viewer(xp.cpu().numpy(), range_hist="auto")

    breakpoint()


if __name__ == "__main__":
    main()
