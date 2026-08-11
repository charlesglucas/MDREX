import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from models.exomild.psf_sampler import BatchPSFSampler
from models.unet import get_unet
from utils.viz import cube_3d_viewer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalMasker(nn.Module):
    def __init__(self, kernel_size, size, circle):
        super().__init__()
        self.kernel_size = kernel_size
        self.size = size
        self.hks = self.kernel_size // 2
        self.circle = circle

        if self.circle:
            yy, xx = np.mgrid[: self.size, : self.size] - self.size // 2
            yy = torch.tensor(yy, device=device)
            xx = torch.tensor(xx, device=device)
            rr = torch.sqrt(yy**2 + xx**2).view(1, self.size, self.size)
            mask_circle = rr < (self.size // 2 - self.hks - 3)
            mask_circle = mask_circle.view(1, 1, self.size, self.size)
            self.register_buffer("mask_circle", mask_circle, persistent=False)

        # noise = torch.rand((1, self.size, self.size)) * 0.1
        # self.register_buffer("noise", noise, persistent=False)

    def forward(self, x):
        bs, C, H, W = x.shape
        assert H == self.size
        assert W == self.size

        x = x.view(bs, C, H, W)
        x_max = F.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        mask_1 = x == x_max
        # (bs, C, H, W)

        # mask_1[:, :selfhks, :] = 0
        # mask_1[:, -hks:, :] = 0
        # mask_1[:, :, :hks] = 0
        # mask_1[:, :, -hks:] = 0

        # avoid problems in flat regions, a single peak
        # noise = torch.rand((1, H, W), device=device) * 0.1

        noise = torch.rand((1, C, self.size, self.size), device=device) * 0.1
        mask_noise = mask_1.float() + noise

        x_max = F.max_pool2d(
            mask_noise,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )
        mask_2 = mask_noise == x_max
        mask = mask_1 & mask_2
        # (bs, C, H, W)

        if self.circle:
            mask = mask & self.mask_circle
            # (bs, C, H, W)

        if False:
            plt.imshow(mask[0].cpu().detach())
            plt.show()

        return mask.float()


def get_mask_local_max(x, kernel_size, circle=True):
    bs, H, W = x.shape

    # (bs, H, W)
    if circle:
        yy, xx = np.mgrid[:H, :W] - H // 2
        yy = torch.tensor(yy, device=device)
        xx = torch.tensor(xx, device=device)
        rr = torch.sqrt(yy**2 + xx**2)
        mask_3 = rr < (H // 2 - hks)
        mask = mask & mask_3

    return mask.float()


class Filter(nn.Module):
    def __init__(
        self,
        skip_norm_unet,
        n_channels,
        local_max,
        kernel_size,
        gate_type,
        is_residual,
    ):
        super().__init__()
        self.local_max = local_max
        self.n_channels = n_channels
        self.thresh = nn.Parameter(torch.tensor(4.0))
        self.kernel_size = kernel_size
        self.is_residual = is_residual
        print(f"[Filter] {self.is_residual=}")

        self.psf_sampler_center = BatchPSFSampler(
            psf_size=self.kernel_size, mode="centered"
        )
        # self.alpha_min = torch.tensor(1e-8, device=device)
        self.gate_type = gate_type
        print(f"[Filter] {self.gate_type=}")
        assert self.gate_type in ["sigmoid", "relu", "unet", "unet_normal"]
        if self.gate_type == "relu":
            self.snr_min = torch.tensor(0.01, device=device)
        elif self.gate_type == "sigmoid":
            self.a = nn.Parameter(torch.tensor(1.0, device=device))
            self.b = nn.Parameter(torch.tensor(-5.0, device=device))
        elif self.gate_type in ["unet", "unet_normal"]:
            self.unet = get_unet(
                timesteps=self.n_channels,
                image_channels_out=self.n_channels,
                skip_norm=skip_norm_unet,
                n_channels=16,
                n_groups=16,
                n_blocks=1,
                ch_mults=[1, 2, 2],
            )
            self.snr_min = torch.tensor(0.1, device=device)
            # self.snr_min = torch.tensor(1.0, device=device)
        else:
            raise ValueError()

        self.divisor = None

    def init_divisor(self, H):
        fold = nn.Fold(
            output_size=(H, H), stride=1, kernel_size=self.kernel_size
        )
        unfold = nn.Unfold(stride=1, kernel_size=self.kernel_size)
        ones = torch.ones((1, 1, H, H))
        divisor = fold(unfold(ones))
        self.divisor = 1 / divisor.view(1, H, H)
        # (bs, H, W)

    def forward(self, snr, alpha, sigma, t):
        bs, C, H, W = snr.shape
        bs, C, H, W = alpha.shape
        if self.divisor is None:
            self.init_divisor(H)

        # if snr.isnan().any():
        # breakpoint()
        # if alpha.isnan().any():
        # breakpoint()

        # sigma = alpha / snr

        if self.local_max:
            mask_local = get_mask_local_max(snr, kernel_size=self.kernel_size)
            # snr = snr * mask_local
        else:
            mask_local = torch.ones_like(snr)

        # if not self.training:
        # breakpoint()

        if self.gate_type == "relu":
            gate = F.relu(1 - self.thresh / torch.maximum(snr, self.snr_min))
        elif self.gate_type == "sigmoid":
            gate = torch.sigmoid(self.a * snr + self.b)
        elif self.gate_type == "unet":
            if False:
                cube_3d_viewer(
                    snr[0].cpu().detach().numpy(),
                    range_hist="auto",
                    quantile=0,
                )

            gate = self.unet(snr.unsqueeze(2), t=t).squeeze(2) / torch.maximum(
                snr, self.snr_min
            )
            # breakpoint()
            gate = F.relu(gate)
            if False:
                plt.imshow(gate[0].cpu())
                plt.colorbar()
                plt.show()
        elif self.gate_type == "unet_normal":
            # snr_new = self.unet(snr[:, None, None, ...])[:, 0, 0, ...]
            # if True:
            # breakpoint()
            if False:
                print("TO CHANGE !!!!")
                with torch.no_grad():
                    light = torch.zeros_like(snr)
                    # light[:, :, 5:10, 5:10] = 5
                    light[:, :, 5:10, 5:10] = 1
                snr_light = snr + light
                snr_new_light = self.unet(snr_light.unsqueeze(2), t=t).squeeze(2)

            snr_new = self.unet(snr.unsqueeze(2), t=t).squeeze(2)
            # snr_new = snr_new_light

            if False:
                diff = snr_new - snr_new_light
                # plt.imshow(diff[0, 0].cpu().numpy())
                # plt.imshow(snr_new_light[0, 0].cpu().numpy(), vmin=-1, vmax=1)
                # plt.show()
                # breakpoint()
            # breakpoint()
            if self.is_residual:
                snr_new = snr - snr_new
            alpha_filtered = snr_new * sigma
            if alpha_filtered.isnan().any():
                breakpoint()
            # alpha_filtered = F.relu(snr_new * sigma)
            return alpha_filtered
        else:
            raise ValueError(f"{self.gate_type=}")

        gate = gate * mask_local

        alpha_filtered = alpha * gate
        # (bs, H, W)

        # if alpha_filtered.isnan().any():
        # breakpoint()

        return alpha_filtered

        # h_kernel, _ = self.psf_sampler_center(**psf_params)
        # # (bs, 1, ks, ks)

        # hps = self.kernel_size // 2
        # rec = F.conv_transpose2d(
        # alpha_filtered.unsqueeze(1), weight=h_kernel, stride=1
        # )
        # rec = rec.squeeze(1)[:, hps:-hps, hps:-hps]
        # # (bs, H, W)

        # sigma_rec = F.conv_transpose2d(
        # sigma.unsqueeze(1), weight=h_kernel, stride=1
        # )
        # sigma_rec = sigma_rec.squeeze(1)[:, hps:-hps, hps:-hps]
        # # (bs, H, W)

        # if not self.local_max:
        # rec = rec * self.divisor
        # # (bs, H, W)

        # if False:
        # plt.imshow(mask_local[0].detach())
        # plt.figure()
        # plt.imshow(snr[0].detach())
        # plt.show()
        # breakpoint()
