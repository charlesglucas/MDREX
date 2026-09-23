import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils.viz import cube_3d_viewer
from models.unet import get_unet
import matplotlib.pyplot as plt

from models.exomild.features import FeaturesPipeline
from models.exomild.engine import EngineSpatial
from models.exomild.patch_extraction import PatchExtractor, PatchFormatter
from models.exomild.patch_handler import SpatialPatchesHandler
from models.exomild.patch_extraction import WavelengthAligner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 256


class SpatialTerm(nn.Module):
    def __init__(
        self,
        patch_size,
        name,
        stride,
        dilation,
        temporal_length,
        temporal_stride,
        temporal_dilation,
        groups,
        params_pipeline,
        mode_ab,
        use_dataparallel,
        grad_params,
        batch_size,
        symmetry,
        low_memory_mode,
        dim_samples,
        dim_prefeatures,
        stat_model="gaussian",
        denoise_mean=False,
        verbose=False,
        # verbose=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.K = self.patch_size**2
        self.stride = stride
        self.groups = groups
        self.verbose = verbose
        self.use_dataparallel = use_dataparallel
        self.name = name
        print(f"[SpatialTerm] {self.name=}")
        print(f"[SpatialTerm] {self.use_dataparallel=}")
        assert self.groups == 1
        self.dilation = dilation
        print(f"[SpatialTerm] {self.dilation=}")
        self.denoise_mean = denoise_mean
        print(f"[SpatialTerm] {self.denoise_mean=}")
        self.stat_model = stat_model
        print(f"[SpatialTerm] {self.stat_model=}")

        self.n_patches = int(1 + (IMG_SIZE - self.patch_size) / self.stride)
        self.bsp = self.n_patches**2
        print(f"[SpatialTerm] {self.bsp=}")

        self.symmetry = symmetry
        print(f"[SpatialTerm] {self.symmetry=}")
        assert self.symmetry in [1, 2, 4]
        self.temporal_length = temporal_length

        features_pipeline = FeaturesPipeline(
            params_pipeline=params_pipeline, bsp=self.bsp, C=self.symmetry
        )

        self.engine = EngineSpatial(
            features_pipeline=features_pipeline,
            mode_ab=mode_ab,
            G=self.groups,
            patch_size=self.patch_size,
            S=self.symmetry,
            grad_params=grad_params,
            verbose=self.verbose,
            dilation=dilation,
            batch_size=batch_size,
            low_memory_mode=low_memory_mode,
            stat_model=self.stat_model,
        )
        self.device_ids = [i for i in range(torch.cuda.device_count())]
        print(f"[SpatialTerm] {self.device_ids=}")
        if len(self.device_ids) < 2:
            print("[SpatialTerm] Not using DataParallel")
            self.use_dataparallel = False

        if self.use_dataparallel:
            self.engine = nn.DataParallel(
                self.engine, device_ids=self.device_ids
            )

        # self.psf_sampler_shift = BatchPSFSampler(
        # psf_size=self.patch_size, mode="shift"
        # )

        # self.patch_extractor = PatchExtractor(
        # dim_t=temporal_length,
        # dilation_t=temporal_dilation,
        # stride_t=temporal_stride,
        # dim_x=patch_size,
        # dilation_x=dilation,
        # stride_x=stride,
        # symmetry=symmetry,
        # )

        self.spatial_handler = SpatialPatchesHandler(
            patch_size=patch_size,
            stride=stride,
            groups=1,
        )

        self.patch_formatter = PatchFormatter(
            dim_samples=dim_samples, dim_prefeatures=dim_prefeatures
        )
        # self.patch_formatter = PatchFormatter(
        # dim_samples="t l s", dim_prefeatures="h w"
        # )

        self.one = torch.ones(1, device=device)

        tt, yy, xx = torch.tensor(
            np.mgrid[:64, :256, :256], device=device
        ).float()

        self.wavelength_aligner = WavelengthAligner()

        # tt, yy, xx = torch.tensor(np.mgrid[:T, :H, :W], device=device).float()

        # rot_cube = torch.ones(bs, 1, T, H, W)
        # rot_cube = rot_cube * rot.view(bs, 1, T, 1, 1)

        # xp = self.patch_extractor(x_in.unsqueeze(1))
        # tt_p = self.patch_extractor(tt[None, None, :], mode="unfold")
        # yy_p = self.patch_extractor(yy[None, None, :], mode="unfold")
        # xx_p = self.patch_extractor(xx[None, None, :], mode="unfold")

        # self.tt_p = self.patch_formatter(tt_p, mode="unfold")
        # self.xx_p = self.patch_formatter(xx_p, mode="unfold")
        # self.yy_p = self.patch_formatter(yy_p, mode="unfold")
        if self.denoise_mean:
            self.mean_denoiser = get_unet(
                timesteps=64,
                image_channels_out=64,
                skip_norm=False,
                n_channels=16,
                n_groups=16,
                n_blocks=1,
                ch_mults=[1, 2, 2],
                init_zero_last=True,
            )

        self.buffered_params = None

    def to_patches(self, x, lbda, align_wavelenghts=True):
        bs, C, T, H, W = x.shape

        if False:
            # cube_3d_viewer(xp[bsp // 2, 0, 0].detach().numpy(), range_hist="auto")
            cube_3d_viewer(
                x[0].view(C * T, H, W).detach().cpu().numpy(),
                range_hist="auto",
            )
            breakpoint()

        # (bs, C, T, H, W)
        if False:
            x_in = x.clone()
            x = self.wavelength_aligner.align(x, lbda)
            x = self.wavelength_aligner.unalign(x, lbda, force_positive=False)
            breakpoint()
        if align_wavelenghts:
            # NOTE: there is a bug, the frame is rotated after this
            # this is not a big deal since the operation is symmetrical
            # and is aligned back with wavelenght unalignement
            # however this is an issue for astrometry
            x = self.wavelength_aligner.align(x, lbda)

        # all_sym = [x]

        if self.symmetry > 1:
            pad = (0, 1, 0, 1)

            x = x.view(bs * C * T, 1, H, W)

            x_pad = F.pad(x, pad=pad, mode="replicate")
            # (bs * C * T, 1, H + 1, W + 1)

            x_pad_2 = torch.rot90(x_pad, k=2, dims=(-2, -1))
            # (bs * C * T, 1, H + 1, W + 1)

            all_x_pad = [x_pad, x_pad_2]

            if self.symmetry == 4:
                x_pad_1 = torch.rot90(x_pad, k=1, dims=(-2, -1))
                # (bs * C * T, 1, H + 1, W + 1)

                x_pad_3 = torch.rot90(x_pad, k=3, dims=(-2, -1))
                # (bs * C * T, 1, H + 1, W + 1)
                all_x_pad += [x_pad_1, x_pad_3]

            x = torch.stack(all_x_pad, dim=0)
            # (S, bs * C *T, 1, H + 1, W + 1)

            x = x[:, :, :, :-1, :-1]
            # (bs * C *T, S, H, W)

            x = x.view(self.symmetry, bs, C, T, H, W)
            # (bs, C, T, S, H, W)
        else:
            x = x.unsqueeze(0)
            # (bs, C, T, 1, H, W)
            pass
        # S = x.shape[0]

        # xp = self.patch_extractor(x, lbda=lbda, mode="unfold")
        # (bsp, c, s, t, h, w)
        # xp = self.spatial_handler(x, lbda=lbda, mode="unfold")
        # (bsp, c, s, t, h, w)

        xp = []
        for _x in x:
            _xp = self.spatial_handler.unfold(_x)
            xp.append(_xp)
            # (bs, bsp, ct, K)
            # breakpoint()

        xp = torch.stack(xp, dim=3)
        # (bs, bsp, C * T, S, K)
        bs, bsp, ct, S, K = xp.shape

        # breakpoint()
        xp = xp.view(bsp, C, T, S, self.patch_size, self.patch_size)

        if False:
            # cube_3d_viewer(xp[bsp // 2, 0, 0].detach().numpy(), range_hist="auto")
            cube_3d_viewer(
                xp[bsp // 2, :, 0, 0].detach().numpy(), range_hist="auto"
            )
            breakpoint()

        xp = self.patch_formatter(xp, mode="unfold")
        # (bsp, dim_samples, s, dim_prefeatures)

        if False:
            # cube_3d_viewer(xp[bsp // 2, 0, 0].detach().numpy(), range_hist="auto")
            cube_3d_viewer(xp.detach().numpy(), range_hist="auto")
            breakpoint()

        return xp

    def from_patches(self, x, lbda, force_positive):
        bsp, d_s, G, K = x.shape
        assert G == 1

        x = x.view(bsp, d_s, K)

        # (bsp, dim_samples, dim_prefeatures)
        x = self.patch_formatter(x, mode="fold")
        # (bsp, l, s, t, h, w)
        L = x.shape[1]

        # x = self.patch_extractor(x, mode="fold", lbda=lbda)
        # (bs, T, H, W)
        x = x.reshape(1, bsp, d_s, G, K)

        x = self.spatial_handler.fold(x.contiguous())
        # (b, T, G, H, W)
        bs, CT, G, H, W = x.shape

        x = x.view(bs, L, -1, H, W)
        # (b, C, T, H, W)

        x = self.wavelength_aligner.unalign(
            x, lbda, force_positive=force_positive
        )
        # (bs, C, T, H, W)

        return x

    def get_ab(self, x, psf_params, rot, lbda, x_s=None):
        bs, C, T, H, W = x.shape
        assert H == IMG_SIZE
        assert bs == 1
        bs, T = rot.shape

        use_preprenorm = False

        xp = self.to_patches(x, lbda=lbda)
        # (bsp, CT, S, dim_pf)
        bsp, CT, S, K = xp.shape

        bsp = xp.shape[0]
        # breakpoint()
        # breakpoint()
        if False:
            cube_3d_viewer(xp.detach().numpy(), range_hist="auto")
            breakpoint()

        # assert xp.shape[-1] == self.C * self.K
        # bsp = xp.shape[1]

        # xp = xp.view(bsp, self.temporal_length, self.C * self.K)
        # (bs * bsp, T, C * K)

        if x_s is not None:
            # xp_s = self.spatial_handler.unfold(x_s.unsqueeze(2))
            # # (bs, bsp, T, K)
            # xp_s = xp_s.view(bsp, T, self.K)

            # # (bs * bsp, T, K)

            if use_preprenorm:
                x_s = (x_s - prepre_mean) / prepre_std
                # (bs, T, H, W)
            xp_s = self.to_patches(x_s, lbda=lbda)
            # (bs, bsp, T, C * K)
            # (bsp, dim_s, dim_pf)
        else:
            xp_s = None
            # (bs, bsp, T, K)
            # (bsp, dim_s, dim_pf)

        amplitudes = psf_params["amplitude"]
        coeff_amplitudes = amplitudes / amplitudes[-1]
        coeff_amplitudes = coeff_amplitudes.view(1, C, 1, 1, 1, 1)

        idx = None
        # rot = torch.cat([rot] * bsp, dim=0)
        if self.use_dataparallel:
            n_devices = len(self.device_ids)
            idx = torch.arange(self.bsp, device=device)
            # psf_params = {k: v.expand(n_devices)}
            psf_params = {
                k: torch.cat([v] * n_devices) for (k, v) in psf_params.items()
            }

        fx_mean = None
        if self.denoise_mean:
            fx = self.engine(
                mode="get_features",
                xp=xp,
                # tt_p=self.tt_p,
                # yy_p=self.yy_p,
                # xx_p=self.xx_p,
                # rot=rot,
                xp_s=xp_s,
                idx=idx,
                **psf_params
            )["fx"]
            # (bsp, T, fs)
            fs = fx.shape[-1]
            fx_mean = torch.mean(fx, dim=1)
            fx_mean = fx_mean.view(1, 63, 63, fs)
            fx_mean = fx_mean.permute(0, 3, 1, 2)
            # (1, C, H', W')
            fx_mean = F.pad(fx_mean, (0, 1, 0, 1), "constant", 0)
            # (1, C, H' + 1, W' + 1)
            rot_range = torch.abs(rot[:, -1] - rot[:, 0])

            fx_mean = fx_mean + self.mean_denoiser(
                fx_mean.unsqueeze(2),
                t=rot_range,
            ).squeeze(2)
            # (1, C, H' + 1, W' + 1)

            fx_mean = fx_mean[:, :, :-1, :-1]
            # (1, C, H', W')

            fx_mean = fx_mean.permute(0, 2, 3, 1)
            # (1, H', W', C)

            fx_mean = fx_mean.reshape(bsp, 1, fs)
            # (bsp, 1, fs)
        ab = self.engine(
            mode="get_ab",
            xp=xp,
            # tt_p=self.tt_p,
            # yy_p=self.yy_p,
            # xx_p=self.xx_p,
            # rot=rot,
            xp_s=xp_s,
            idx=idx,
            fx_mean=fx_mean,
            **psf_params,
        )
        a = ab["a"]
        b = ab["b"]
        # (bs * bsp, T, G, K)
        ct = b.shape[1]
        # if (a <= 0).any():
        # breakpoint()
        a = a.expand(-1, ct, -1, -1)

        # if (a <= 0).any():
        # breakpoint()

        # a, b = self.engine(
        # xp=xp, xp_s=xp_s, idx=idx, **psf_params
        # )

        # a = a.view(bs, bsp, -1, self.K)
        # b = b.view(bs, bsp, self.temporal_length, self.K)
        # b: (bs, bsp, T, G, K (pos))
        # a: (bs, bsp, T_jx, G, K (pos))

        # TODO: multi-spectral here
        a_ff = self.from_patches(a, lbda=lbda, force_positive=True)
        b_ff = self.from_patches(b, lbda=lbda, force_positive=False)
        # (bs, C, T_jx, H, W)

        # if (a_ff <= 0).any():
        # breakpoint()

        if use_preprenorm:
            b_ff = b_ff / prepre_std
            a_ff = a_ff / prepre_std**2
            # a_ff = a_ff / prepre_std

        a_ff = a_ff.unsqueeze(3)
        b_ff = b_ff.unsqueeze(3)
        # (bs, C, T_jx, G, H, W)

        a_ff = a_ff * coeff_amplitudes**2
        b_ff = b_ff * coeff_amplitudes

        return a_ff, b_ff, {}

    def get_idx_patches(self, coords_t, H):
        # coords_t: (bs, n_sources, T, 2, 1)
        bs, n_sources, T, _ = coords_t.shape
        patches_per_pix = self.patch_size // self.stride

        def coord_to_idx(coords_t):
            bs, _ = coords_t.shape
            hps = self.patch_size // 2

            # get coords closest pixel
            coords_t_int = torch.clip(
                coords_t.round(), min=0 * self.one, max=(H - 1) * self.one
            ).int()
            # (bs, 2)

            # coords_y = coords[:, 0]
            # coords_x = coords[:, 1]
            n_patches_H = 1 + (H - self.patch_size) // self.stride

            idx_patch = torch.clip(
                coords_t_int // self.stride,
                min=0 * self.one,
                max=(n_patches_H - 1) * self.one,
            ).long()
            # (bs,2)

            shift_patch = -torch.arange(patches_per_pix, device=device)
            # shift_patch = shift_patch.view(1, -1).expand(2, -1)
            idx_shift = torch.zeros(
                (patches_per_pix, patches_per_pix, 2), device=device, dtype=int
            )
            idx_shift[:, :, 0] += shift_patch.view(-1, 1)
            idx_shift[:, :, 1] += shift_patch.view(1, -1)
            # (ppp, ppp, 2)

            idx_shift = idx_shift.view(1, patches_per_pix, patches_per_pix, 2)
            # (1, ppp, ppp, 2)

            idx_patch = idx_patch.view(bs, 1, 1, 2) + idx_shift
            # (bs, ppp, ppp, 2)

            idx_patch = torch.clip(
                idx_patch,
                min=0 * self.one,
                max=(n_patches_H - 1) * self.one,
            ).long()

            coords_center = idx_patch * self.stride + hps
            # (bs, ppp, ppp, 2)

            shift_coords = coords_t.view(bs, 1, 1, 2) - coords_center
            # (bs, ppp, ppp, 2)

            idx_patch_flat = (
                n_patches_H * idx_patch[:, :, :, 0] + idx_patch[:, :, :, 1]
            )
            # (bs, ppp, ppp)
            if (idx_patch_flat >= n_patches_H**2).any():
                breakpoint()
            if (idx_patch_flat <= 0).any():
                breakpoint()

            return idx_patch_flat, shift_coords

        idx_patch, shift_coords = coord_to_idx(coords_t.view(-1, 2))
        idx_patch = idx_patch.view(
            bs, n_sources, T, patches_per_pix, patches_per_pix
        )
        shift_coords = shift_coords.view(
            bs, n_sources, T, patches_per_pix, patches_per_pix, 2
        )
        return idx_patch, shift_coords

    def astrometry(
        self, z, x, x_s, coords_t, j_coords_t, weights, psf_params, lbda
    ):
        bs, T, H, W = x_s.shape
        # breakpoint()

        # 1. compute idx patches affected -> n_update
        idx, shift = self.get_idx_patches(coords_t=coords_t, H=H)
        # idx, shift = self.get_idx_patches(coords_t=coords_t, H=H)
        bs, n_sources, T, ppp, _ = idx.shape
        bs, n_sources, T, ppp, ppp, _ = shift.shape
        # idx: (bs, n_sources, T, ppp, ppp)
        # shift: (bs, n_sources, T, ppp, ppp, 2)
        idx_u, idx_u_inverse = torch.unique(idx, return_inverse=True)
        bsp_u = len(idx_u)
        # idx_u: (bsp_u,)
        # idx_u_inverse: (bs, n_sources, T, ppp, ppp)
        # breakpoint()

        # print("patch handler")
        # torch.cuda.synchronize()
        # 2. compute features, jacc of patches affected (engine)
        # breakpoint()
        if False:
            xp_s_0 = self.spatial_handler.unfold(x_s.unsqueeze(1))
            xp_s_1 = self.to_patches(x_s.unsqueeze(1), lbda=lbda)
            viz = [xp_s_0.view(3969, 64, 64), xp_s_1.view(3969, 64, 64)]
            viz = torch.cat(viz, dim=2)
            cube_3d_viewer(viz.cpu().numpy())
            # cube_3d_viewer(viz.cpu().numpy())
            # breakpoint()
        if True:
            xp_s = self.to_patches(
                x_s.unsqueeze(1), lbda=lbda, align_wavelenghts=False
            )
            bsp, T, S, K = xp_s.shape
            xp_s_u = xp_s[idx_u, :, :, :]
            xp_s_u = xp_s_u.view(-1, T, S, self.K)
        else:
            xp_s = self.spatial_handler.unfold(x_s.unsqueeze(1))
            bsp = xp_s.shape[1]
            xp_s_u = xp_s[:, idx_u, :, :]
            xp_s_u = xp_s_u.view(-1, T, self.K)

        # breakpoint()
        # xp_s = self.to_patches(x_s.unsqueeze(1), lbda=lbda)
        # (bsp, T, S, K)
        # bsp, T, S, K = xp_s.shape

        # (bs, bsp, T, K)
        # bsp = xp_s.shape[0]
        # print(f"{bsp=}")
        # torch.cuda.synchronize()
        # breakpoint()

        # print(f"{idx_u.shape=}")
        # xp_s_u = xp_s[:, idx_u, :, :]
        # xp_s_u = xp_s[idx_u, :, :, :]
        # (n_update, T, S, K)

        # (bs, n_update, T, K)
        # if True:
        # if (xp_s_u == 0).all(dim=(2, 3)).any():
        # print("bug before xp_s_u")
        # print(torch.argwhere((xp_s_u == 0).all(dim=(2, 3))))
        # breakpoint()

        # torch.cuda.synchronize()

        # print(f"flatten")
        # xp_s_u = xp_s_u.view(-1, T, S, self.K)
        # xp_s_u = xp_s_u.view(-1, T, self.K)
        idx_u = idx_u.flatten()
        torch.cuda.synchronize()
        # print(f"{idx_u=}")
        # if idx_u[9] == 809:
        # breakpoint()

        # breakpoint()
        if self.verbose:
            print("before astrometry_1")
            # print(f"{xp_s_u.shape=}")
            print(f"{idx_u.shape=}")

        # torch.cuda.synchronize()
        if False:
            idx_all = torch.arange(bsp, device=device).long()
            xp_s_all = xp_s.view(-1, T, self.K)
            print(f"{idx_all.shape=}")
            print(f"{xp_s_all.shape=}")
            Lt_fx_c, Lt_jx = self.engine(
                mode="astrometry_1", xp_s=xp_s_all, idx=idx_all
            )
            print(f"{idx_u.max()=}")
            print(f"{idx_u.min()=}")
            Lt_fx_c = Lt_fx_c[idx_u]
            Lt_jx = Lt_jx[idx_u]
            # breakpoint()
        else:
            out = self.engine(mode="astrometry_1", xp_s=xp_s_u, idx=idx_u)
            Lt_fx_c = out["Lt_fx_c"]
            Lt_jx = out["Lt_jx"]

        if self.verbose:
            print(f"results_out")
            print(f"{Lt_fx_c.mean()=}")
            print(f"{Lt_jx.mean()=}")
            print(f"{Lt_fx_c.shape=}")
            print(f"{Lt_jx.shape=}")
        _, T_jx, G, fsp, K = Lt_jx.shape

        Lt_fx_c = Lt_fx_c.view(bs, bsp_u, T, self.groups, fsp, 1)
        # (bs, bsp_u, T, G, fsp, 1)

        Lt_jx = Lt_jx.view(bs, bsp_u, T_jx, self.groups, fsp, self.K)
        # (bs, bsp_u, T, G, fsp, K)

        # Lt_fx_c, Lt_jx = self.engine(xp_s=xp_s_u, idx=idx_u)
        # Lt_fx_c: (bs, bsp, T, G, fsp, 1)
        # Lt_jx: (bs, bsp, T_jx, G, fsp, K)
        # _, _, _, G, fsp, K = Lt_jx.shape

        idx_t = torch.arange(T).view(1, 1, T, 1, 1).long()
        idx_t = idx_t.expand(bs, n_sources, -1, ppp, ppp)
        # idx_t: (bs, n_sources, T, ppp, ppp)

        # breakpoint()
        Lt_fx_c_idx = Lt_fx_c[:, idx_u_inverse.flatten(), idx_t.flatten()]
        # (bs, n_sources * T * ppp * ppp, G, fsp, 1)

        # breakpoint()
        if T_jx == T:
            Lt_jx_idx = Lt_jx[
                :, idx_u_inverse.flatten(), idx_t.flatten(), :, :, :
            ]
            # (bs, n_sources * T * ppp * ppp, G, fsp, K)
        elif T_jx == 1:
            Lt_jx_idx = Lt_jx[
                :,
                idx_u_inverse.flatten(),
                torch.tensor(0, device=device, dtype=torch.int64),
                :,
                :,
                :,
            ]
            # (bs, n_sources * T * ppp * ppp, G, fsp, K)
        else:
            raise ValueError(f"{T_jx=}")

        # bsa = bs * n_sources * T * ppp * ppp
        bsa = bs * n_sources
        bsb = T * ppp * ppp

        Lt_fx_c_idx = Lt_fx_c_idx.view(bsa, bsb, G, fsp, 1)
        # (bsa, G, fsp, 1)

        Lt_jx_idx = Lt_jx_idx.view(bsa, bsb, G, fsp, K)
        # (bsa, G, fsp, K)

        shift = shift.view(bsa, bsb, 2)
        # (bsa, 2)

        z = z.view(bsa, 3)
        # (bsa, 3)

        j_coords_t = j_coords_t.view(bs, 1, T, 2, 2)
        # (bs, 1, T, 2, 2)
        j_coords_t = j_coords_t.expand(-1, n_sources, -1, -1, -1)
        # (bs, n_sources, T, 2, 2)

        j_coords_t = j_coords_t.view(bsa, T, 2, 2)
        # (bsa, T, 2, 2)

        weights = weights.view(bsa, T, G)
        # (bsa, T, G)

        if self.use_dataparallel:
            n_devices = len(self.device_ids)
            idx = torch.arange(self.bsp, device=device)
            # psf_params = {k: v.expand(n_devices)}
            psf_params = {
                k: torch.cat([v] * n_devices) for (k, v) in psf_params.items()
            }

        out = self.engine(
            mode="astrometry_2",
            shift=shift,
            z=z,
            Lt_fx_c_idx=Lt_fx_c_idx,
            Lt_jx_idx=Lt_jx_idx,
            j_coords_t=j_coords_t,
            weights=weights,
            **psf_params,
        )
        grad = out["grad"]
        hess = out["hess"]
        # grad *= 10
        # grad: (bsa, 3, 1)
        # hess: (bsa, 3, 3)

        grad = grad.view(bs, n_sources, 3, 1)
        # (bs, n_sources, 3, 1)

        hess = hess.view(bs, n_sources, 3, 3)
        # (bs, n_sources, 3, 3)

        return grad, hess

    def fit_params(self, x, lbda):
        bs, C, T, H, W = x.shape
        assert H == IMG_SIZE
        assert bs == 1

        xp = self.to_patches(x, lbda=lbda)
        # (bsp, dim_s, dim_pf)
        # (bs, bsp, T, C * K)
        # bsp = xp.shape[0]
        # breakpoint()

        # assert xp.shape[-1] == self.C * self.K
        # bsp = xp.shape[1]

        # xp = xp.view(bsp, self.temporal_length, self.C * self.K)
        # (bs * bsp, T, C * K)

        # xp_s = None

        # amplitudes = psf_params["amplitude"]
        # coeff_amplitudes = amplitudes / amplitudes[-1]
        # coeff_amplitudes = coeff_amplitudes.view(1, C, 1, 1, 1, 1)

        idx = None
        # rot = torch.cat([rot] * bsp, dim=0)
        if self.use_dataparallel:
            # n_devices = len(self.device_ids)
            idx = torch.arange(self.bsp, device=device)
            # # psf_params = {k: v.expand(n_devices)}
            # # psf_params = {
            # # k: torch.cat([v] * n_devices) for (k, v) in psf_params.items()
        # # }

        # fx_mean = None
        params = self.engine(
            mode="fit_params",
            xp=xp,
            # tt_p=self.tt_p,
            # yy_p=self.yy_p,
            # xx_p=self.xx_p,
            # rot=rot,
            # xp_s=xp_s,
            idx=idx,
            # fx_mean=fx_mean,
            # **psf_params
        )
        self.buffered_params = params

    # def get_log_likelihood(self, x, lbda):
    #     bs, C, T, H, W = x.shape
    #     assert H == IMG_SIZE
    #     assert bs == 1

    #     xp = self.to_patches(x, lbda=lbda)
    #     # (bsp, dim_s, dim_pf)
    #     # (bs, bsp, T, C * K)
    #     bsp = xp.shape[0]
    #     # breakpoint()

    #     # assert xp.shape[-1] == self.C * self.K
    #     # bsp = xp.shape[1]

    #     # xp = xp.view(bsp, self.temporal_length, self.C * self.K)
    #     # (bs * bsp, T, C * K)

    #     # xp_s = None

    #     # amplitudes = psf_params["amplitude"]
    #     # coeff_amplitudes = amplitudes / amplitudes[-1]
    #     # coeff_amplitudes = coeff_amplitudes.view(1, C, 1, 1, 1, 1)

    #     idx = None
    #     # rot = torch.cat([rot] * bsp, dim=0)
    #     if self.use_dataparallel:
    #         # n_devices = len(self.device_ids)
    #         idx = torch.arange(self.bsp, device=device)
    #         # # psf_params = {k: v.expand(n_devices)}
    #         # # psf_params = {
    #         # # k: torch.cat([v] * n_devices) for (k, v) in psf_params.items()
    #     # # }

    #     # fx_mean = None
    #     assert self.buffered_params is not None

    #     ll = self.engine(
    #         mode="get_log_likelihood",
    #         xp=xp,
    #         # tt_p=self.tt_p,
    #         # yy_p=self.yy_p,
    #         # xx_p=self.xx_p,
    #         # rot=rot,
    #         # xp_s=xp_s,
    #         idx=idx,
    #         # fx_mean=fx_mean,
    #         # **psf_params,
    #         **self.buffered_params,
    #     )["ll"]

    #     ll = ll.permute(1, 0)
    #     # (C *T, bsp)

    #     ll = ll.view(C, T, self.n_patches, self.n_patches)
    #     # (C, T, H', W')

    #     return ll

    def fit_params(self, x, lbda):
        bs, C, T, H, W = x.shape
        assert H == IMG_SIZE
        assert bs == 1
    
        xp = self.to_patches(x, lbda=lbda)
        idx = None
        if self.use_dataparallel:
            idx = torch.arange(self.bsp, device=device)

        params = self.engine(
            mode="fit_params",
            xp=xp,
            idx=idx
        )
        return params
    
    def fit_params_noweight(self, x, lbda):
        bs, C, T, H, W = x.shape
        assert H == IMG_SIZE
        assert bs == 1
    
        xp = self.to_patches(x, lbda=lbda)
        idx = None
        if self.use_dataparallel:
            idx = torch.arange(self.bsp, device=device)

        params = self.engine(
            mode="fit_params_noweight",
            xp=xp,
            idx=idx
        )
        return params
    
    def get_log_likelihood(self, x, lbda, params):
        bs, C, T, H, W = x.shape
        assert H == IMG_SIZE
        assert bs == 1

        xp = self.to_patches(x, lbda=lbda)
        idx = None
        if self.use_dataparallel:
            idx = torch.arange(self.bsp, device=device)

        ll = self.engine(
            mode="get_log_likelihood",
            xp=xp,
            idx=idx,
            **params
        )["ll"]

        ll = ll.permute(1, 0)
        ll = ll.view(C, T, self.n_patches, self.n_patches)
        return ll
    
    def extract_patches(self, x, lbda):
        bs, C, T, H, W = x.shape
        assert H == IMG_SIZE
        assert bs == 1
    
        xp = self.to_patches(x, lbda=lbda)
        idx = None
        if self.use_dataparallel:
            idx = torch.arange(self.bsp, device=device)

        params = self.engine(
            mode="get_features",
            xp=xp,
            idx=idx
        )
        return params