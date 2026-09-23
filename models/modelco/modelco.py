import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.modelco.norms import WhitenNorm, TemporalNorm
# from inverse_pb.operators.rotation import BatchRotationOperator
from utils.rotation import BatchRotationOperator
from models.unet import get_unet
from datasets.transforms.injection import gaussian_2d, fit_gaussian_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mean_masked(x, mask_temporal, T_real):
    bs, T, H, W = x.shape
    bs, T, _, _ = mask_temporal.shape
    assert T_real.ndim == 4

    mean = torch.sum(x * mask_temporal, dim=1, keepdim=True) / T_real
    # (bs, 1, H, W)

    return mean


def get_std_masked(x, mask_temporal):
    bs, T, H, W = x.shape
    # bs, T, _, _ = mask_temporal.shape

    mask_temporal = mask_temporal.view(bs, T, 1, 1).float()
    # (bs, T, 1, 1)

    T_real = torch.sum(mask_temporal, dim=1, keepdim=True)
    # (bs, 1, 1, 1)
    assert T_real.ndim == 4

    mean_masked = get_mean_masked(
        x=x, mask_temporal=mask_temporal, T_real=T_real
    )
    # (bs, 1, H, W)

    x_c = (x - mean_masked) * mask_temporal
    # (bs, T, H, H)

    var = torch.sum(x_c**2, dim=1, keepdim=True) / (T_real - 1)
    # (bs, 1, H, W)

    std = torch.sqrt(var + 1e-3)
    # (bs, 1, H, W)

    return std


class Swish(nn.Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class PatchHandler(nn.Module):
    def __init__(self, patch_size, stride, normalize, C_ff):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.normalize = normalize
        self.C_ff = C_ff

        self.K = self.patch_size**2
        self.Kp = self.C_ff * self.patch_size**2

        self.kernel_in = (
            torch.eye(self.K)
            .view(self.patch_size**2, 1, 1, self.patch_size, self.patch_size)
            .to(device)
        )
        self.kernel_out = (
            torch.eye(self.Kp)
            .reshape(self.Kp, self.C_ff, 1, self.patch_size, self.patch_size)
            .to(device)
        )
        self.divisor = None

    def init_divisor(self, H, W):
        ones_in = torch.ones(1, 1, 1, H, W).to(device)

        x = F.conv3d(
            ones_in,
            weight=self.kernel_in,
            stride=(1, self.stride, self.stride),
        )

        x = F.conv_transpose3d(
            x, weight=self.kernel_in, stride=(1, self.stride, self.stride)
        )
        self.divisor = 1 / x
        # (b, 1, 1, H, W)

    def forward(self, x, mode, **kwargs):
        if mode == "extract":
            return self.extract(x, **kwargs)
        elif mode == "aggregate":
            return self.aggregate(x)
        else:
            raise ValueError(f"{mode=}")

    def extract(self, x, mask_temporal):
        # multiple channels are in the batch dimension
        b, C, T, H, W = x.shape
        assert C == 1
        # assert b == 1
        if mask_temporal is None:
            mask_temporal = torch.ones((b, T), device=x.device)
        assert mask_temporal.ndim == 2

        if self.divisor is None:
            self.init_divisor(H, W)

        xp = F.conv3d(
            x, weight=self.kernel_in, stride=(1, self.stride, self.stride)
        )
        # (bs, K, T, dH, dW)
        # K = xp.shape[1]

        self.mu = None
        self.std = None

        if self.normalize:
            assert mask_temporal.ndim == 2
            dim = (1, 2)
            T_real = torch.sum(mask_temporal).int().item()
            # (bs, 1)
            xp_crop = xp[:, :, :T_real, :, :]
            mu = torch.mean(xp_crop, dim=dim, keepdim=True)
            std = torch.std(xp_crop, dim=dim, keepdim=True)

            self.mu = mu
            self.std = std + 0.001

            xp = (xp - self.mu) / self.std
            # (bs, K, T, dH, dW)

        if xp.isnan().any():
            breakpoint()

        return xp

    def aggregate(self, x):
        b, Kp, T, dH, dW = x.shape
        # assert Kp == self.C_ff * self.patch_size**2
        assert Kp == self.Kp

        x_agg = F.conv_transpose3d(
            x, weight=self.kernel_out, stride=(1, self.stride, self.stride)
        )
        # (bs, 1, T, dH, dW)

        return x_agg * self.divisor


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_channels,
        spatial_size,
        affine_norm,
        n_groups,
        reg_C,
        use_whitening,
        joint_mu,
        joint_C,
        is_residual,
    ):
        super().__init__()
        self.is_residual = is_residual
        print(f"[ResidualBlock] {self.is_residual=}")
        if use_whitening:
            self.norm_1 = WhitenNorm(
                n_groups=n_groups,
                n_channels=n_channels,
                affine=affine_norm,
                reg_C=reg_C,
                joint_mu=joint_mu,
                joint_C=joint_C,
            )
        else:
            self.norm_1 = TemporalNorm(
                n_channels=n_channels,
                n_groups=n_channels,
                affine=affine_norm,
            )
        self.act_1 = Swish()
        self.conv_1 = nn.Conv3d(
            in_channels=n_channels,
            out_channels=n_channels,
            stride=(1, 1, 1),
            kernel_size=(1, spatial_size, spatial_size),
            padding="same",
        )

        self.norm_2 = TemporalNorm(
            n_channels=n_channels,
            n_groups=n_channels,
            affine=affine_norm,
        )
        self.act_2 = Swish()
        self.conv_2 = nn.Conv3d(
            in_channels=n_channels,
            out_channels=n_channels,
            stride=(1, 1, 1),
            kernel_size=(1, 1, 1),
            padding="same",
        )

    def forward(self, x, mask_temporal):
        # (bs, L, T, dH, dW)

        h = self.norm_1(x=x, mask_temporal=mask_temporal)
        h = self.conv_1(h)
        h = self.act_1(h)

        h = self.norm_2(x=h, mask_temporal=mask_temporal)
        # h = self.act_2(h)
        h = self.conv_2(h)
        # (bs, L, T, dH, dW)

        if self.is_residual:
            return x + h
        return h


class SpecklesAligned(nn.Module):
    def __init__(
        self,
        patch_size,
        stride,
        n_channels,
        spatial_size,
        normalize_patches,
        n_blocks,
        affine_norm,
        n_groups,
        reg_C,
        C_ff,
        use_whitening,
        joint_mu,
        joint_C,
        residual_speckles,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.patch_handler = PatchHandler(
            patch_size=patch_size,
            stride=stride,
            normalize=normalize_patches,
            C_ff=C_ff,
        )
        self.proj_1 = nn.Conv3d(
            in_channels=patch_size**2,
            out_channels=n_channels,
            kernel_size=(1, 1, 1),
            bias=False,
        )

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    n_channels=n_channels,
                    spatial_size=spatial_size,
                    affine_norm=affine_norm,
                    n_groups=n_groups,
                    reg_C=reg_C,
                    use_whitening=use_whitening,
                    joint_mu=joint_mu,
                    joint_C=joint_C,
                    is_residual=residual_speckles,
                )
            ]
        )

        self.proj_2 = nn.Conv3d(
            in_channels=n_channels,
            out_channels=C_ff * patch_size**2,
            kernel_size=(1, 1, 1),
            bias=False,
        )

    def forward(self, x, mask_temporal):
        b, T, H, W = x.shape

        x = x.unsqueeze(1)
        # (b, 1, T, H, W)

        x = self.patch_handler(x, mask_temporal=mask_temporal, mode="extract")
        # (bs, K, T, dH, dW)

        x = self.proj_1(x)
        # (bs, L, T, dH, dW)

        for block in self.blocks:
            x = block(x=x, mask_temporal=mask_temporal)

        x = self.proj_2(x)
        # (bs, C_ff * K, T, dH, dW)

        x = self.patch_handler(
            x, mask_temporal=mask_temporal, mode="aggregate"
        )
        # (bs, C_ff, T, H, W)

        return x


class MTA(nn.Module):
    def __init__(self, interpolation, size):
        super().__init__()
        self.size = size

        self.batch_rotation = BatchRotationOperator(
            device=device,
            out_size=self.size,
            in_size=self.size,
            mode=interpolation,
            zero_init=True,
        )

    def fit_psf(self, psf):
        assert psf.shape[0] == 1
        psf_params = fit_gaussian_2d(psf[0].detach().cpu().numpy())
        psf_params["offset"] = 0
        psf_params["x0"] = 0
        psf_params["y0"] = 0
        self.psf_params = psf_params

    def sample_psf(self, patch_size=9):
        yy, xx = np.mgrid[:patch_size, :patch_size] - patch_size // 2
        xy = (xx, yy)
        psf = gaussian_2d(xy, **self.psf_params)
        return torch.tensor(psf, dtype=torch.float32, device=device).view(
            patch_size, patch_size
        )

    def forward(self, x, rot, mask, mask_temporal):
        bs, C_ff, T, H, W = x.shape

        mask_temporal = mask_temporal.view(bs, 1, T, 1, 1).float()
        # (bs, 1, T, 1, 1)

        rot_exp = rot.view(bs, 1, T)
        rot_exp = rot_exp.expand(-1, C_ff, -1)
        rot_exp = rot_exp.reshape(bs * C_ff, T)
        # (bs * C_ff, T)

        x = x.view(bs * C_ff, T, H, W)
        # (bs * C_ff, T, H, W)

        x = self.batch_rotation.forward(x=x, rot=-rot_exp, mode="warp")
        # (bs * C_ff, T, H, W)

        x = x.view(bs, C_ff, T, H, W)
        # (bs, C_ff, T, H, W)

        if mask is None:
            if self.training:
                raise ValueError("Mask none")
            else:
                mask = torch.ones(
                    (bs, 1, T, H, W), device=device, dtype=torch.float32
                )
                # (bs, 1, T, H, W)
        else:
            mask = mask.float().view(bs, 1, T, H, W)
            # (bs, 1, T, H, W)

        mask = mask * mask_temporal
        # (bs, 1, T, H, W)

        num = torch.sum(x * mask_temporal, dim=2, keepdim=True)
        # (bs, C_ff, 1, H, W)

        mask = mask.squeeze(1)
        # (bs, T, H, W)

        den = self.batch_rotation.forward(x=mask, rot=-rot, mode="warp")
        # (bs, T, H, W)

        den = torch.sum(den, dim=1, keepdim=True)
        # (bs, 1, H, W)

        den = den.unsqueeze(1)
        # (bs, 1, 1, H, W)

        x = num / torch.sqrt(den + 1e-3)
        # (bs, C_ff, 1, H, W)

        x = x.squeeze(2)
        # (bs, C_ff, H, W)

        assert x.ndim == 4

        return x, den


class ObjectAligned(nn.Module):
    def __init__(self, C_ff):
        super().__init__()

        self.unet = get_unet(
            timesteps=C_ff,
            image_channels_out=1,
            n_channels=16,
            n_groups=16,
            n_blocks=1,
            ch_mults=[1, 2, 2],
            skip_norm=False,
        )

    def forward(self, x):
        # (bs, C_ff, H, W)
        assert x.ndim == 4
        x = self.unet(x.unsqueeze(2))
        # (bs, 1, 1, H, W)

        x = torch.nan_to_num(x)

        return x


class ObjectAlignedNonLearnable(nn.Module):
    def __init__(self, C_ff):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=C_ff, out_channels=1, kernel_size=1)
        self.thresh = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        # (bs, C_ff, H, W)
        assert x.ndim == 4
        x = self.conv(x)
        # (bs, 1, H, W)

        x = F.relu(x - self.thresh)
        # (bs, 1, H, W)

        x = x.unsqueeze(2)
        # (bs, 1, 1, H, W)

        x = torch.nan_to_num(x)
        # (bs, 1, 1, H, W)

        return x


class ModelCo(nn.Module):
    def __init__(
        self,
        patch_size,
        stride,
        normalize_patches,
        interpolation,
        interpolation_alpha,
        n_channels,
        spatial_size,
        n_blocks,
        n_groups,
        affine_norm,
        reg_C,
        C_ff,
        mask_min,
        speckles_learnable,
        object_learnable,
        use_whitening,
        joint_mu,
        joint_C,
        input_size,
        residual_speckles,
        **kwargs,
    ):
        super().__init__()
        self.mask_min = mask_min
        self.interpolation_alpha = interpolation_alpha
        self.input_size = input_size

        self.speckles_aligned = SpecklesAligned(
            patch_size=patch_size,
            stride=stride,
            normalize_patches=normalize_patches,
            n_channels=n_channels,
            spatial_size=spatial_size,
            n_blocks=n_blocks,
            affine_norm=affine_norm,
            n_groups=n_groups,
            reg_C=reg_C,
            C_ff=C_ff,
            use_whitening=use_whitening,
            joint_mu=joint_mu,
            joint_C=joint_C,
            residual_speckles=residual_speckles,
        )

        self.mta = MTA(interpolation=interpolation, size=self.input_size)
        if object_learnable:
            self.object_aligned = ObjectAligned(C_ff=C_ff)
        else:
            self.object_aligned = ObjectAlignedNonLearnable(C_ff=C_ff)
        self.std = None

    def forward(self, x, rot, mask_temporal, mask=None, **kwargs):
        bs, C, T, H, W = x.shape
        assert H == self.input_size, f"{H}!={self.input_size}"
        assert W == self.input_size, f"{W}!={self.input_size}"
        assert C == 1, f"Only monospectral supported with MODEL&CO, {C=}"

        x = x.squeeze(1)
        # (bs, T, H, W)

        x = self.speckles_aligned(x=x, mask_temporal=mask_temporal)
        x, _ = self.mta(x=x, rot=rot, mask_temporal=mask_temporal, mask=mask)
        x = self.object_aligned(x=x)
        # (bs, 1, 1, H, W)

        return {"snr": x.view(bs, C, H, W)}

    def forward_multi(self, mode, **kwargs):
        if mode == "sb":
            return self.forward(**kwargs)
        else:
            raise NotImplementedError

    def get_target(self, y, s_0, rot, mask, mask_temporal):
        # breakpoint()
        # bs, T, H, W = y.shape
        bs, C, T, H, W = y.shape
        assert C == 1, f"Only monospectral supported with MODEL&CO, {C=}"
        y = y.squeeze(1)
        s_0 = s_0.squeeze(1)
        with torch.no_grad():
            obj = y - s_0
            # (bs, T, H, W)

            std = get_std_masked(x=y, mask_temporal=mask_temporal)
            # (bs, 1, H, W)

            obj = obj / std
            # (bs, T, H, W)

            obj = obj.view(bs, 1, T, H, W)
            # (bs, 1, T, H, W)

            target, den = self.mta(
                x=obj, rot=rot, mask_temporal=mask_temporal, mask=mask
            )
            assert den.ndim == 5
            assert target.ndim == 4
            mask = den.view(bs, 1, H, W) >= self.mask_min
            target = target.view(bs, 1, H, W)

        # return target, mask
        return {"snr_target": target, "mask_target": mask}
