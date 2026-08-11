import torch
import torch.nn as nn
import contextlib
import torch.nn.functional as F
from torch import optim
from einops import einsum

from models.exomild.utils import agg_dict, split_dict
from models.exomild.patch_extraction import PSFFormatterTemporal
from models.exomild.psf_sampler import BatchPSFSampler
from utils.viz import cube_3d_viewer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_likelihood_compound(y_t, m_n, C_n, s_nt):
    b, T, K = y_t.shape
    b, N, K = m_n.shape
    b, N, K, K = C_n.shape
    b, N, T = s_nt.shape

    s_nt_2_inv = (1 / s_nt**2).view(b, N, T, 1, 1)
    # (b, N, T, 1, 1)

    C_n_inv = torch.inverse(C_n).view(b, N, 1, K, K)
    # C_n_inv = C_n.view(N, 1, K, K)
    # (b, N, 1, K, K)

    C_t_inv = torch.sum(s_nt_2_inv * C_n_inv, dim=1)
    # (b, T, K, K)

    C_t = torch.inverse(C_t_inv)
    # (b, T, K, K)

    m_n = m_n.view(b, N, 1, K, 1)
    # (b, N, 1, K, 1)

    m_t = C_t @ torch.sum(s_nt_2_inv * C_n_inv @ m_n, dim=1)
    # (b, T, K, 1)

    y_t = y_t.view(b, T, K, 1)
    # (b, T, K, 1)

    diff = y_t - m_t
    # (b, T, K, 1)

    ll = diff.transpose(-1, -2) @ C_t_inv @ diff
    # (b, T, 1, 1)
    # breakpoint()

    ld = torch.logdet(C_t_inv)
    # (b, T)

    ll = -0.5 * (ll.flatten() - ld.flatten())
    # (T, 1, 1)

    ll = torch.mean(ll)
    # ()

    loss = -ll
    # ()

    # params = {"m_t": m_t, "C_t": C_t, "m_n": m_n, "C_n": C_n}
    params = {"m_t": m_t, "C_t_inv": C_t_inv, "m_n": m_n, "C_n": C_n}

    return loss, params


class CovConverter2:
    def __init__(self, K):
        self.K = K
        self.Kp = (self.K * (self.K - 1)) // 2

        mask = torch.zeros(
            (self.K, self.K), device=device, dtype=torch.float32
        )
        # (K, K)

        # all_idx_y, all_idx_x = torch.triu_indices(self.K, self.K, offset=1)
        idx_y, idx_x = torch.tril_indices(self.K, self.K, offset=-1)
        mask[idx_y, idx_x] = 1
        self.mask = mask.view(1, 1, self.K, self.K).to(device)

        trans_diag = []
        for k in range(self.K):
            mat = torch.zeros(
                (self.K, self.K), device=device, dtype=torch.float32
            )
            mat[k, k] = 1.0
            trans_diag.append(mat)

        self.trans_diag = torch.stack(trans_diag).unsqueeze(0)
        # (1, K, K, K)

    def __call__(self, diag, offdiag):
        b, N, K = diag.shape
        b, N, K, _ = offdiag.shape

        diag = torch.exp(diag)
        # (b, N, K)

        diag = torch.diag_embed(diag)
        # (b, N, K, K)

        offdiag = self.mask * offdiag
        # (b, N, K, K)

        L = diag + offdiag
        # (b, N, K, K)

        C = L @ L.transpose(-1, -2)
        # (b, N, K, K)

        return C


class EngineSpatial(nn.Module):
    def __init__(
        self,
        features_pipeline,
        G,
        S,
        patch_size,
        dilation,
        mode_ab,
        grad_params,
        verbose,
        batch_size,
        low_memory_mode,
        stat_model,
        psf_size=7,
    ):
        super().__init__()
        self.features_pipeline = features_pipeline
        self.G = G
        self.S = S
        self.mode_ab = mode_ab
        self.patch_size = patch_size
        self.K = self.patch_size**2
        self.batch_size = batch_size
        print(f"[Engine] {self.batch_size=}")
        self.verbose = verbose
        self.low_memory_mode = low_memory_mode
        print(f"[Engine] {self.low_memory_mode=}")
        self.psf_size = psf_size
        self.stat_model = stat_model
        assert self.stat_model in [
            "gaussian",
            "compound_gaussian",
        ], f"{self.stat_model=}"

        self.psf_sampler_shift = BatchPSFSampler(
            psf_size=self.patch_size, mode="shift"
        )
        
        # self.psf_sampler_shift = BatchPSFSampler(
        # psf_size=self.patch_size, mode="shift"
        # )

        # self.psf_sampler_all = BatchPSFSampler(
        # psf_size=self.patch_size, mode="all_pixels"
        # )
        self.grad_params = grad_params

        self.psf_sampler_center = BatchPSFSampler(
            psf_size=self.psf_size, mode="centered", dilation=dilation
        )

        # self.psf_formatter = PSFFormatterTemporal()
        # self.psf_formatter = PSFFormatterSpatial()
        print(f"[Engine] {self.grad_params=}")
        print(f"[Engine] {self.low_memory_mode=}")

        self.rho=None

    def batchify(func):
        def wrap(self, *args, **kwargs):
            # print(f"{batch_size=}")
            # batch_size = kwargs.pop("batch_size")
            # batch_size = cls.batch_size
            if self.verbose:
                print(f"Batchifying {func.__name__}")
            batch_size = self.batch_size
            if batch_size is None:
                return func(self, *args, **kwargs)
            kwargs_split = split_dict(kwargs, batch_size=batch_size)
            out_split = []
            # for k_split in tqdm(kwargs_split, disable=not self.verbose):
            for k_split in kwargs_split:
                out_split.append(func(self, *args, **k_split))
                torch.cuda.empty_cache()
            if self.verbose:
                print(f"Aggregating")
            out = agg_dict(out_split)
            if self.verbose:
                print(f"Batchifying done")
            # breakpoint()
            return out

        return wrap

    def forward(
        self,
        mode,
        xp=None,
        xp_s=None,
        idx=None,
        shift=None,
        Lt_jx_idx=None,
        Lt_fx_c_idx=None,
        z=None,
        j_coords_t=None,
        weights=None,
        prenorm_mean=None,
        prenorm_std=None,
        mean=None,
        C_inv=None,
        **psf_params
    ):
        if self.verbose:
            print("forward engine")
        if mode == "get_ab":
            return self.get_ab(xp=xp, xp_s=xp_s, idx=idx, **psf_params)
        elif mode == "fit_params":
            return self.fit_params(xp=xp, idx=idx)
        elif mode == "fit_params_noweight":
            return self.fit_params_noweight(xp=xp, idx=idx)
        elif mode == "get_log_likelihood":
            return self.get_log_likelihood(
                xp=xp,
                idx=idx,
                prenorm_mean=prenorm_mean,
                prenorm_std=prenorm_std,
                mean=mean,
                C_inv=C_inv,
            )
        elif mode == "get_features":
            return self.get_features(xp=xp, xp_s=xp_s, idx=idx, **psf_params)
        elif mode == "astrometry_1":
            return self.astrometry_1(xp_s=xp_s, idx=idx)
        elif mode == "astrometry_2":
            return self.astrometry_2(
                shift=shift,
                Lt_jx_idx=Lt_jx_idx,
                Lt_fx_c_idx=Lt_fx_c_idx,
                z=z,
                j_coords_t=j_coords_t,
                weights=weights,
                **psf_params,
            )
        else:
            raise NotImplementedError(f"{mode=}")

    # @batchify
    # def get_features(self, xp, xp_s=None, idx=None, **psf_params):
    #     if self.verbose:
    #         print("get_ab")
    #         print(f"{xp.shape=}")
    #         if idx is not None:
    #             print(f"{idx.shape=}")
    #         print(f"{psf_params['amplitude'].shape=}")
    #     bsp, T, S, d_pf = xp.shape
    #     xp = xp.view(1, bsp, T, self.S, self.K)
    #     if xp_s is not None:
    #         xp_s = xp_s.view(1, bsp, T, self.S, self.K)
    #         xp_norm = xp_s
    #     else:
    #         xp_norm = xp
    #     # else:
    #     # xp_s = xp
    #     # torch.inverse(torch.ones((1, 1), device="cuda:0"))

    #     bsp = xp.shape[1]
    #     # xp_norm = xp_s if xp_s else xp
    #     # (bs, bsp, T, C, K)

    #     prenorm_mean = torch.mean(xp_norm, dim=(2, 4), keepdim=True)
    #     prenorm_std = torch.std(xp_norm, dim=(2, 4), keepdim=True)
    #     # (bs, bsp, 1, S, 1)

    #     xp = (xp - prenorm_mean) / prenorm_std
    #     # (bs, bsp, T, S, K)

    #     xp = xp.view(1, bsp, T, self.S * self.K)
    #     # (bs, bsp, T, C * K)

    #     if False:
    #         cube_3d_viewer(xp[0].detach().numpy(), range_hist="auto")

    #     fx, jx_acc = self.features_pipeline(xp, idx=idx, with_jacobian=False)
    #     # fx = xp
    #     # fx: (bs, bsp, T, fs)
    #     # jx: (bs, bsp, T, K, fs)
    #     _, bsp, T, fs = fx.shape
    #     fx = fx.view(bsp, T, fs)

    #     return {"fx": fx}

    @batchify
    def get_ab(self, xp, xp_s=None, idx=None, fx_mean=None, **psf_params):
        # def forward(self, xp, xp_s=None, idx=None, **psf_params):
        if self.verbose:
            print("get_ab")
            print(f"{xp.shape=}")
            if idx is not None:
                print(f"{idx.shape=}")
            print(f"{psf_params['amplitude'].shape=}")
        # TODO: prenorm should be spectrally dependent
        bsp, CT, S, d_pf = xp.shape
        xp = xp.view(1, bsp, CT, self.S, self.K)

        if xp_s is not None:
            xp_s = xp_s.view(1, bsp, CT, self.S, self.K)
            xp_norm = xp_s
        else:
            xp_norm = xp

        bsp = xp.shape[1]
        # xp_norm = xp_s if xp_s else xp
        # (bs, bsp, T, C, K)

        prenorm_mean = torch.mean(xp_norm, dim=(2, 4), keepdim=True)
        prenorm_std = torch.std(xp_norm, dim=(2, 4), keepdim=True)
        # (bs, bsp, 1, S, 1)

        # print("TO CHANGE HERE !!!!!!")
        # prenorm_mean = torch.zeros_like(prenorm_mean)
        # prenorm_std = torch.ones_like(prenorm_std)

        xp = (xp - prenorm_mean) / prenorm_std
        # (bs, bsp, T, S, K)

        xp = xp.view(1, bsp, CT, self.S * self.K)
        # (bs, bsp, T, S * K)

        if False:
            cube_3d_viewer(xp[0].detach().numpy(), range_hist="auto")

        fx, jx_acc = self.features_pipeline(xp, idx=idx, with_jacobian=True)
        # fx: (bs, bsp, CT, fs)
        # jx: (bs, bsp, CT, K, fs)
        bs, bsp, CT, fs = fx.shape

        fx = fx.view(bs, bsp, CT, self.G, -1)
        # fx: (bs, bsp, T, G, fsp)
        fsp = fx.shape[-1]

        # bs_jx, bsp_jx, T_jx, _, _ = jx.shape

        # TODO: check that no permute is required here
        # jx = jx.view(bs, bsp_jx, T_jx, self.K, self.groups, fsp)
        # jx: (bs, bsp(jx), T (jx), K, G, fsp)

        if xp_s is not None:
            xp_s = (xp_s - prenorm_mean) / prenorm_std
            xp_s = xp_s.view(1, bsp, CT, self.S * self.K)
            # (bs, bsp, T, C * K)
            fx_s, jx_s_acc = self.features_pipeline(
                xp_s, idx=idx, with_jacobian=True
            )
            # fx_s: (bs, bsp, T, fs)
            # jx_s: (bs, bsp, T, K, fs)
            fx_s = fx_s.view(bs, bsp, CT, self.G, -1)
            # jx_s = jx_s.view(bs, bsp_jx, T_jx, self.K, self.groups, fsp)
            # jx: (bs, bsp (jx), T (x), K, G, fsp)
        else:
            fx_s = fx
            # jx_s_acc = jx_acc

        # print(f"{fx_s.shape=}")
        if fx_s.isnan().any():
            breakpoint()
        if fx_s.isinf().any():
            breakpoint()

        # fx: (bs, bsp, CT, G, fsp)
        bg_params, metrics = self.compute_params(x=fx_s, mean=fx_mean)
        mean = bg_params["mean"]
        # (bs, bsp, Tm, G, fsp)
        # mean = mean.expand(-1, -1, CT, -1, -1)
        bs, bsp, Tm, G, fsp = mean.shape
        # print(f"{mean.shape=}")

        C_inv = bg_params["C_inv"]
        # C_inv = C_inv.expand(-1, -1, CT, -1, -1, -1)
        # (bs, bsp, Tm, G, fsp, fsp)
        bs, bsp, Tm, G, fsp, _ = C_inv.shape
        # print(f"{C_inv.shape=}")
        # breakpoint()

        fx = fx.view(bs, bsp, CT, 1, self.G, fsp)
        # (bs, bsp, CT, 1, G, fsp)

        fx_c = fx - mean.view(bs, bsp, Tm, 1, self.G, fsp)
        # (bs, bsp, T, 1, G, fsp)

        # breakpoint()
        if self.mode_ab == "right":
            a, b = self._get_ab_right(
                fx_c=fx_c,
                # jx_s_acc=jx_s_acc,
                idx=idx,
                jx_s_acc=jx_acc,
                C_inv=C_inv,
                **psf_params,
            )
            raise NotImplementedError

            # a: (bs, bsp, T (jx), G, K (pos))
            # b: (bs, bsp, T, G, K (pos))
        elif self.mode_ab == "left":
            a, b = self._get_ab_left(
                fx_c=fx_c,
                idx=idx,
                jx_s_acc=jx_acc,
                C_inv=C_inv,
                **psf_params,
            )
        else:
            raise ValueError(f"{self.mode_ab=}")
        # TODO: maintain groups dimension as a channel

        # correct prenorm
        prenorm_std = prenorm_std[:, :, :, 0, None, :]
        # (bs, bsp, 1, 1, 1)

        if b.isnan().any():
            print("breakpoint()")
            breakpoint()
        if a.isnan().any():
            print("breakpoint()")
            breakpoint()

        b = b / prenorm_std
        a = a / prenorm_std**2
        # b: (bs, bsp, CT, G, K (pos))
        # a: (bs, bsp, CT_jx, G, K (pos))

        b = b.view(bsp, CT, self.G, self.K)
        a = a.view(bsp, -1, self.G, self.K)

        out = {"a": a, "b": b}
        # breakpoint()

        return out

        # return out, metrics

    def _get_ab_left(self, fx_c, jx_s_acc, idx, C_inv, **psf_params):
        bs, bsp, CT, _, G, fsp = fx_c.shape
        bs, bsp, Tm, G, fsp, _ = C_inv.shape
        # C = psf_params["amplitude"].flatten().shape[0]

        # T = CT // C

        # Lt = torch.linalg.cholesky(C_inv, upper=True)
        # (bs, bsp, G, fsp, fsp)

        try:
            # C_inv = torch.linalg.inv(C_hat)
            Lt, info = torch.linalg.cholesky_ex(C_inv, upper=True)
            # (bs, bsp, Tm, G, K, K)
            info = (info > 0).float()
            # mask_info = (1 - info).float().view(bs, bsp, CT, G, 1, 1)
            mask_info = (1 - info).float().view(bs, bsp, Tm, G, 1, 1)
            # n_issues = info.sum()
            # print(f"n_issues: {n_issues} / {mask_info.numel()}")
            n_issues = info.sum()
            if n_issues > 0:
                breakpoint()
            # metrics["ratio_issues"] = info.sum() / info.numel()

            eye = torch.eye(fsp, device=C_inv.device).view(1, 1, 1, fsp, fsp)
            # breakpoint()
            Lt = Lt * mask_info + eye * (1 - mask_info)
        except Exception as e:
            print(e)
            breakpoint()
            raise

        Lt = Lt.view(bs, bsp, Tm, G, fsp, fsp)
        # (bs, bsp, Tm, G, fsp, fsp)

        Lt_jx = jx_s_acc.forward_left(Lt)
        # (bs, bsp, Tm, G, fsp, S * K)

        # _, bsp_jx, T_jx, _, _, _ = L_jx.shape
        _, bsp_jx, T_jx, _, _, _ = Lt_jx.shape

        Lt_jx = Lt_jx.view(bs, bsp_jx, T_jx, self.G, fsp, self.S, -1)

        # select first channel symmetry
        Lt_jx = Lt_jx[:, :, :, :, :, 0, :]

        # L_jx = L_jx.view(-1, 1, self.patch_size, self.patch_size)
        Lt_jx = Lt_jx.reshape(
            bs * bsp_jx * T_jx * self.G * fsp,
            1,
            self.patch_size,
            self.patch_size,
        )
        # (bs bsp_jx T_jx G fsp, 1, ps, ps)

        # if C > 1:
        # Lt_jx = Lt_jx.view(
        # bs * bsp,
        # C,
        # T_jx,
        # G,
        # fsp,
        # 1,
        # self.patch_size,
        # self.patch_size,
        # )
        # # (bs bsp, C, T G fsp, 1, ps, ps)

        # Lt_jx = Lt_jx.permute(0, 5, 2, 3, 4, 1, 6, 7)
        # (bs bsp, 1, T G fsp, C, ps, ps)
        # breakpoint()
        psf_params_last = {k: v[-1, None] for k, v in psf_params.items()}
        psf_kernel = self.psf_sampler_center.forward(**psf_params_last)[0]

        # breakpoint()
        # psf_params_c = [
        # {k: v.flatten()[c][None, ...] for (k, v) in psf_params.items()}
        # for c in range(C)
        # ]
        # # breakpoint()
        # psf_kernel = [
        # self.psf_sampler_center.forward(**params)[0]
        # for params in psf_params_c
        # ]

        # # TODO: scale params multiband after homotethy
        # psf_kernel = torch.stack(psf_kernel).view(
        # C, self.psf_size, self.psf_size
        # )
        # (C, h, w)

        # h_kernel, _ = self.psf_sampler_center(**psf_params)
        # (bs, 1, ks, ks)

        # Lt_jx_h = F.conv2d(Lt_jx, weight=h_kernel, padding="same")
        # psf_kernel = psf_kernel.view()

        Lt_jx_h = F.conv2d(Lt_jx, weight=psf_kernel, padding="same")
        # (bs bsp T G fsp, 1, ps, ps)

        Lt_jx_h = Lt_jx_h.view(bs, bsp_jx, T_jx, G, fsp, self.K)
        # (bs, bsp, T, G, fsp, K)

        a = torch.sum(Lt_jx_h**2, dim=4)
        # (bs, bsp, T, G, K)

        if a.isnan().any():
            breakpoint()

        # if (a <= 0).any():
        # print("PROBLEM")
        # breakpoint()

        # if not (a > 0).all():
        # print("PROBLEM")
        # breakpoint()

        Lt = Lt.view(bs, bsp, Tm, G, fsp, fsp)
        # (bs, bsp, 1, G, fsp, fsp)

        fx_c = fx_c.view(bs, bsp, CT, G, fsp)
        # (bs, bsp, T, G, fsp)

        # TODO: maybe collapse groups first
        # L_fx_c = (L.transpose(-1, -2) @ fx_c.unsqueeze(-1))
        Lt_fx_c = Lt @ fx_c.unsqueeze(-1)
        # (bs, bsp, T, G, fsp, 1)

        if self.low_memory_mode:
            b = []
            for k in range(self.K):
                b_k = torch.sum(
                    Lt_jx_h[:, :, :, :, :, k, None] * Lt_fx_c, dim=4
                )
                # (bs, bsp, T, G)
                b.append(b_k)
            b = torch.cat(b, dim=-1)
            # (bs, bsp, T, G, K)
        else:
            b = torch.sum(Lt_jx_h * Lt_fx_c, dim=4)
            # (bs, bsp, T, G, K)

        return a, b

    @batchify
    def get_ab2(
        self, xp, tt_p, yy_p, xx_p, rot, xp_s=None, idx=None, **psf_params
    ):
        # def forward(self, xp, xp_s=None, idx=None, **psf_params):

        bsp, d_s, d_pf = xp.shape
        # (bsp, d_s, d_pf)

        rot_p = rot[0, tt_p.long()]
        # (bsp, d_s, d_pf)

        breakpoint()
        h_p = self.psf_formatter(
            tt=tt_p, xx=xx_p, yy=yy_p, rot=rot_p, **psf_params
        )
        # (bsp, d_s, d_pf, d_pf)
        # breakpoint()

        if self.verbose:
            print(f"get_ab")
            print(f"{xp.shape=}")
            if idx is not None:
                print(f"{idx.shape=}")
            print(f"{psf_params['amplitude'].shape=}")
        # bsp, T, CK = xp.shape
        xp = xp.view(1, bsp, d_s, d_pf)
        if xp_s is not None:
            xp_s = xp_s.view(1, bsp, d_s, d_pf)
            xp_norm = xp_s
        else:
            xp_norm = xp
        # (1, bsp, d_s, d_pf)

        bsp = xp.shape[1]

        prenorm_mean = torch.mean(xp_norm, dim=(2, 3), keepdim=True)
        prenorm_std = torch.std(xp_norm, dim=(2, 3), keepdim=True)
        # (1, bsp, 1, 1)
        # prenorm_mean = torch.mean(xp_norm, dim=(3), keepdim=True)
        # prenorm_std = torch.std(xp_norm, dim=(3), keepdim=True)

        xp = (xp - prenorm_mean) / prenorm_std
        # (bs, bsp, d_s, d_pf)

        fx, jx_acc = self.features_pipeline(xp, idx=idx, with_jacobian=True)
        # fx: (bs, bsp, T, fs)
        # jx: (bs, bsp, T, K, fs)
        bs, bsp, d_s, fs = fx.shape

        fx = fx.view(bs, bsp, d_s, self.G, -1)
        # fx: (bs, bsp, T, G, fsp)
        fsp = fx.shape[-1]

        # bs_jx, bsp_jx, T_jx, _, _ = jx.shape

        # TODO: check that no permute is required here
        # jx = jx.view(bs, bsp_jx, T_jx, self.K, self.groups, fsp)
        # jx: (bs, bsp(jx), T (jx), K, G, fsp)

        if xp_s is not None:
            xp_s = (xp_s - prenorm_mean) / prenorm_std
            fx_s, jx_s_acc = self.features_pipeline(
                xp_s, idx=idx, with_jacobian=True
            )
            # fx_s: (bs, bsp, T, fs)
            # jx_s: (bs, bsp, T, K, fs)
            fx_s = fx_s.view(bs, bsp, d_s, self.G, -1)
            # jx_s = jx_s.view(bs, bsp_jx, T_jx, self.K, self.groups, fsp)
            # jx: (bs, bsp (jx), T (x), K, G, fsp)
        else:
            fx_s = fx
            # jx_s_acc = jx_acc

        # print(f"{fx_s.shape=}")
        # if fx_s.isnan().any():
        # breakpoint()
        # if fx_s.isinf().any():
        # breakpoint()

        # fx: (bs, bsp, T, G, fsp)
        bg_params, metrics = self.compute_params(x=fx_s)
        mean = bg_params["mean"]
        # (bs, bsp, 1, G, fsp)
        # print(f"{mean.shape=}")

        C_inv = bg_params["C_inv"]
        # (bs, bsp, G, fsp, fsp)
        # print(f"{C_inv.shape=}")

        fx = fx.view(bs, bsp, d_s, 1, self.G, fsp)
        # (bs, bsp, T, 1, G, fsp)

        fx_c = fx - mean.view(bs, bsp, 1, 1, self.G, fsp)
        # (bs, bsp, T, 1, G, fsp)
        if self.mode_ab == "right":
            a, b = self._get_ab_right(
                fx_c=fx_c,
                # jx_s_acc=jx_s_acc,
                idx=idx,
                jx_s_acc=jx_acc,
                C_inv=C_inv,
                **psf_params,
            )

            # a: (bs, bsp, T (jx), G, K (pos))
            # b: (bs, bsp, T, G, K (pos))
        elif self.mode_ab == "left":
            a, b = self._get_ab_left(
                fx_c=fx_c,
                idx=idx,
                jx_s_acc=jx_acc,
                C_inv=C_inv,
                h_p=h_p,
            )
        else:
            raise ValueError(f"{self.mode_ab=}")
        # TODO: maintain groups dimension as a channel
        # a: (bs, bsp, d_s (jx), G, d_pf (pos))
        # b: (bs, bsp, d_s, G, d_pf (pos))

        # correct prenorm
        prenorm_std = prenorm_std[:, :, :, None, :]
        # (bs, bsp, 1, 1, 1)

        # if b.isnan().any():
        # print("breakpoint()")
        # breakpoint()
        # if a.isnan().any():
        # print("breakpoint()")
        # breakpoint()

        b = b / prenorm_std
        a = a / prenorm_std**2
        # b: (bs, bsp, T, G, K (pos))
        # a: (bs, bsp, T_jx, G, K (pos))

        b = b.view(bsp, d_s, self.G, d_pf)
        a = a.view(bsp, -1, self.G, d_pf)

        # out = {"a": a, "b": b, **metrics}
        out = {"a": a, "b": b}

        # return out, metrics
        return out

    def _get_ab_right2(self, fx_c, jx_s_acc, idx, C_inv, **psf_params):
        bs, bsp, T, _, G, fsp = fx_c.shape

        all_h, _ = self.psf_sampler_all(**psf_params)
        # (bs, K (pos), ps, ps)

        all_h = all_h.view(bs, 1, 1, self.K, self.K)
        # (bs, 1, 1, K (pos), K)

        # jx_s = jx_s.view(bs, bsp_jx, T_jx, 1, self.K, -1)
        # jx: (bs, bsp (jx), T (x), 1, K, fs)

        jx_s_h = jx_s_acc.forward_right(all_h)
        # (bs, bsp (jx), T (jx), K (pos), fs)
        _, bsp_jx, T_jx, _, _ = jx_s_h.shape
        jx_s_h = jx_s_h.view(bs, bsp_jx, T_jx, self.K, self.G, fsp)
        # (bs, bsp (jx), T (jx), K (pos), G, fsp)

        # jx_s_h = all_h @ jx_s
        # (bs, bsp (jx), T (jx), K (pos), fs)

        C_inv = C_inv.view(bs, bsp, 1, 1, self.G, fsp, fsp)
        # (bs, bsp, 1, 1, G, fsp, fsp)

        # breakpoint()
        # jx_s_h = jx_s_h[:, :, :, 0, None, :, :]
        Cinv_jx_s_h = (jx_s_h.unsqueeze(-2) @ C_inv).squeeze(-2)
        # (bs, bsp, T (jx), K (pos), G, fsp)
        # Cinv_jx_s_h = batch_mm(
        # jx_s_h.unsqueeze(-2), C_inv, batch_size=8
        # ).squeeze(-2)
        # (bs, bsp, T (jx), K (pos), G, fsp)

        a = torch.sum(Cinv_jx_s_h * jx_s_h, dim=-1)
        # (bs, bsp, T (jx), K (pos), G)

        a = a.permute(0, 1, 2, 4, 3).contiguous()
        # (bs, bsp, T (jx), G, K (pos))

        b = torch.sum(fx_c * Cinv_jx_s_h, dim=-1)
        # (bs, bsp, T, K (pos), G)

        b = b.permute(0, 1, 2, 4, 3).contiguous()
        # (bs, bsp, T, G, K (pos))

        return a, b

    def _get_ab_left2(self, fx_c, jx_s_acc, idx, C_inv, h_p):
        bs, bsp, d_s, _, G, fsp = fx_c.shape
        bs, bsp, G, fsp, _ = C_inv.shape

        # Lt = torch.linalg.cholesky(C_inv, upper=True)
        # (bs, bsp, G, fsp, fsp)

        try:
            # C_inv = torch.linalg.inv(C_hat)
            Lt, info = torch.linalg.cholesky_ex(C_inv, upper=True)
            # (bs, bsp, G, K, K)
            mask_info = (1 - info).float().view(bs, bsp, G, 1, 1)
            # print(f"n_issues: {n_issues} / {mask_info.numel()}")
            # metrics["ratio_issues"] = info.sum() / info.numel()

            eye = torch.eye(fsp, device=x.device).view(1, 1, 1, fsp, fsp)
            # breakpoint()
            Lt = Lt * mask_info + eye * (1 - mask_info)
            # breakpoint()
        except Exception as e:
            print(e)
            breakpoint()
            raise
        Lt = Lt.view(bs, bsp, 1, G, fsp, fsp)
        # (bs, bsp, 1, G, fsp, fsp)

        Lt_jx = jx_s_acc.forward_left(Lt)
        # (bs, bsp, 1, G, fsp, d_pf)

        d_pf = Lt_jx.shape[-1]

        Lt_jx = Lt_jx.view(bs, bsp, 1, G, fsp, 1, d_pf)
        # (bs, bsp, 1, G, fsp, 1, d_pf)

        h_p = h_p.view(1, bsp, d_s, 1, 1, d_pf, d_pf)
        # (bs, bsp, d_s, 1, 1, d_pf, d_pf)

        Lt_jx_h = torch.sum(Lt_jx * h_p, dim=6)
        # (bs, bsp, d_s, G, fsp, d_pf)
        # breakpoint()

        # _, bsp_jx, T_jx, _, _, _ = L_jx.shape
        # _, bsp_jx, T_jx, _, _, _ = Lt_jx.shape

        # L_jx = L_jx.view(-1, 1, self.patch_size, self.patch_size)
        # Lt_jx = Lt_jx.reshape(-1, 1, self.patch_size, self.patch_size)
        # (bs bsp T G fsp, 1, ps, ps)
        # breakpoint()
        # Lt_jx = Lt_jx.reshape(1, bsp, 1, self.patch_size, self.patch_size)
        # h_p = h_p.view(1, bsp, 1, 1, d_s)

        # h_kernel, _ = self.psf_sampler_center(**psf_params)
        # (bs, 1, ks, ks)

        # Lt_jx_h = F.conv2d(Lt_jx, weight=h_kernel, padding="same")
        # (bs bsp T G fsp, 1, ps, ps)

        # Lt_jx_h = Lt_jx_h.view(bs, bsp_jx, T_jx, G, fsp, self.K)
        # (bs, bsp, T, G, fsp, K)

        a = torch.sum(Lt_jx_h**2, dim=4)
        # (bs, bsp, T, G, K)
        # (bs, bsp, d_s, G, d_pf)
        # breakpoint()

        Lt = Lt.view(bs, bsp, 1, G, fsp, fsp)
        # (bs, bsp, 1, G, fsp, fsp)

        fx_c = fx_c.view(bs, bsp, d_s, G, fsp)
        # (bs, bsp, T, G, fsp)

        # TODO: maybe collapse groups first
        # L_fx_c = (L.transpose(-1, -2) @ fx_c.unsqueeze(-1))
        Lt_fx_c = Lt @ fx_c.unsqueeze(-1)
        # (bs, bsp, T, G, fsp, 1)

        b = torch.sum(Lt_jx_h * Lt_fx_c, dim=4)
        # (bs, bsp, T, G, K)

        return a, b

    def compute_params(self, x, mean=None):
        if self.stat_model == "gaussian":
            return self._params_gaussian(x=x, mean=mean)
        elif self.stat_model == "compound_gaussian_nll":
            return self._params_compound_gaussian(x=x, mean=mean)
        elif self.stat_model == "compound_gaussian_sm":
            return self._params_compound_gaussian(x=x, mean=mean)

    def _params_compound_gaussian_sm(self, x, mean=None):
        bs, bsp, CT, G, K = x.shape
        assert G == 1
        x = x.view(bsp, CT, K)

        N = 1
        # n_iter = 200
        n_iter = 2000
        # lr = 1e-2
        lr = 5e-3

        diag = (
            0.1 * torch.randn((bsp, N, K), dtype=torch.float32, device=device)
        ).requires_grad_(True)
        offdiag = (
            0.1
            * torch.randn((bsp, N, K, K), device=device, dtype=torch.float32)
        ).requires_grad_(True)

    def _params_compound_gaussian_nll(self, x, mean=None):
        bs, bsp, CT, G, K = x.shape
        assert G == 1
        x = x.view(bsp, CT, K)

        N = 1
        # n_iter = 200
        n_iter = 2000
        # lr = 1e-2
        lr = 5e-3

        diag = (
            0.1 * torch.randn((bsp, N, K), dtype=torch.float32, device=device)
        ).requires_grad_(True)
        offdiag = (
            0.1
            * torch.randn((bsp, N, K, K), device=device, dtype=torch.float32)
        ).requires_grad_(True)

        mean = (
            0.1 * torch.randn((bsp, N, K), device=device, dtype=torch.float32)
        ).requires_grad_(True)
        logits = (
            0.1 * torch.randn((bsp, N, CT), device=device, dtype=torch.float32)
        ).requires_grad_(True)
        # diag = (0.1 * torch.randn(bsp, N, K)).requires_grad_(True)
        # offdiag = (0.1 * torch.randn(bsp, N, K, K)).requires_grad_(True)

        converter = CovConverter2(K=K)

        # breakpoint()
        optimizer = optim.Adam([logits, mean, diag, offdiag], lr=lr)
        for i0 in range(n_iter):

            if i0 in [500, 1000, 1500, 1750]:
                lr = lr / 3
                optimizer = optim.Adam([logits, mean, diag, offdiag], lr=lr)

            optimizer.zero_grad()

            C = converter(diag=diag, offdiag=offdiag)

            # with torch.no_grad():
            tr_C = K / torch.einsum("bnkk->bn", C).view(bsp, N, 1, 1)
            # C = C / tr_C
            C = C * tr_C

            # sigmas = torch.exp(logits + common)
            sigmas = torch.exp(logits)
            # print(f"{i0=}")
            # optimizer = optim.SGD([sigmas, mean, C], lr=lr)
            loss, params = get_likelihood_compound(
                y_t=x, m_n=mean, C_n=C, s_nt=sigmas
            )
            # loss = loss + reg_sigmas * torch.std(sigmas, dim=0).mean()
            loss.backward()
            # breakpoint()
            optimizer.step()
            print(f"{i0=}, {loss.item()=}")

        m_t = params["m_t"]
        # (b, T, K)

        C_t_inv = params["C_t_inv"]
        # (b, T, K, K)

        return {
            "mean": m_t.view(1, bsp, CT, 1, K),
            "C_inv": C_t_inv.view(1, bsp, CT, 1, K, K),
        }, None

    # def _params_gaussian(self, x, mean=None):
    #     if self.grad_params:
    #         context = contextlib.nullcontext()
    #     else:
    #         context = torch.no_grad()
    #     with context:
    #         bs, bsp, T, G, fsp = x.shape
    #         assert G == 1

    #         if mean is None:
    #             mean = torch.mean(x, dim=2, keepdim=True)
    #             # (bsp, bsp, 1, G, fsp)
    #         else:
    #             mean = mean.view(bs, bsp, 1, 1, fsp)

    #         x_c = x - mean
    #         # (bs, bsp, T, G, fsp)
    #         # if (x_c == 0).all(dim=(2, 3, 4)).any():
    #         # print(torch.argwhere((x_c == 0).all(dim=(2, 3, 4))))
    #         # breakpoint()

    #         # S_hat = torch.einsum("bstgi,bstgj->bsgij", x_c, x_c) / T
    #         # (bs, bsp, G, K, K)

    #         x_c = x_c.view(bs, bsp, T, fsp)
    #         x_c_T = x_c.permute(0, 1, 3, 2)
    #         S_hat = x_c_T @ x_c / T
    #         # (bs, bsp, K, K)

    #         S_hat = S_hat.unsqueeze(2)
    #         # (bs, bsp, G, K, K)

    #         # tr_S2 = torch.einsum("bsgii->bsg", S_hat**2)
    #         # tr2_S = torch.einsum("bsgii->bsg", S_hat) ** 2
    #         # tr_SS = torch.einsum("bsgii->bsg", S_hat @ S_hat)

    #         tr_S2 = torch.diagonal(S_hat**2, dim1=-2, dim2=-1).sum(dim=-1)
    #         tr2_S = torch.diagonal(S_hat, dim1=-2, dim2=-1).sum(dim=-1) ** 2
    #         tr_SS = torch.diagonal(S_hat @ S_hat, dim1=-2, dim2=-1).sum(dim=-1)

    #         # T_eff = T_eff[..., None]
    #         # torch.cuda.synchronize()

    #         num = tr_SS + tr2_S - 2 * tr_S2
    #         den = (T + 1) * (tr_SS - tr_S2)
    #         # den = torch.maximum(den, )
    #         if self.rho is None:
    #             with torch.no_grad():
    #                 rho = torch.clip(num / den, 0, 1)
    #                 self.rho=rho
    #         rho = self.rho
    #         rho = torch.clip(num / den, 0, 1)
    #         rho = rho.view(bsp, G, 1, 1)
    #         # (bs, bsp, G, 1, 1)

    #         # diag_flat = torch.einsum("bsgii->bsgi", S_hat)
    #         diag_flat = torch.diagonal(S_hat, dim1=-2, dim2=-1)
    #         # (bs, bsp, G, K)

    #         diag = torch.diag_embed(diag_flat)
    #         # (bs, bsp, G, K, K)

    #         eye = (
    #             torch.eye(fsp, device=x.device).float().view(1, 1, 1, fsp, fsp)
    #         )
    #         # (1, 1, 1, K, K)

    #         # C_hat = (1 - rho) * S_hat + rho * diag + eye * 1e-2
    #         C_hat = (1 - rho) * S_hat + rho * diag
    #         # (bs, bsp, G, K, K)

    #         metrics = {}
    #         rho_mean = torch.mean(rho)
    #         # print(f"{rho_mean=}")

    #         # C_inv, info = torch.linalg.inv(C_hat)
    #         if C_hat.isnan().any():
    #             breakpoint()
    #         if C_hat.isinf().any():
    #             breakpoint()
    #         # torch.cuda.synchronize()
    #         try:
    #             # C_inv = torch.linalg.inv(C_hat)
    #             C_inv, info = torch.linalg.inv_ex(C_hat)
    #             # (bs, bsp, G, K, K)
    #             # breakpoint()

    #             info = (info > 0).float()
    #             mask_info = (1 - info).float().view(bs, bsp, G, 1, 1)
    #             n_issues = info.sum()
    #             # print(f"n_issues: {n_issues} / {mask_info.numel()}")
    #             if n_issues > 0:
    #                 breakpoint()
    #             # breakpoint()
    #             metrics["ratio_issues"] = info.sum() / info.numel()

    #             eye = torch.eye(fsp, device=x.device).view(1, 1, 1, fsp, fsp)
    #             # breakpoint()
    #             C_inv = C_inv * mask_info + eye * (1 - mask_info)
    #             # breakpoint()
    #         except Exception as e:
    #             print(e)
    #             breakpoint()
    #             raise

    #         C_inv = C_inv.view(bs, bsp, 1, G, fsp, fsp)
    #         # (bs, bsp, 1, G, fsp, fsp)

    #         # mean:(bsp, bsp, 1, G, fsp)

    #         bg_params = {"mean": mean, "C_inv": C_inv}

    #     return bg_params, metrics

    @batchify
    def astrometry_1(self, xp_s, idx):
        # def forward(self, xp_s, idx):
        # breakpoint()
        if self.verbose:
            print(f"astrometry_1")
            print(f"{xp_s.shape=}")
            print(f"{idx.shape=}")
            print(f"{idx.max()=}")
        torch.cuda.synchronize()
        # bsp, T, K = xp_s.shape
        # xp_s = xp_s.view(1, bsp, T, K)
        if True:
            # bsp, T, K = xp_s.shape
            # xp_s = xp_s.view(bsp, T, 1, K)
            bsp, T, S, K = xp_s.shape
            xp_s = xp_s.view(1, bsp, T, S, K)
        else:
            bsp, T, K = xp_s.shape
            xp_s = xp_s.view(1, bsp, T, K)
            # (bs, bsp, T, S, K)

        # print("before prenorm")
        if True:
            prenorm_mean = torch.mean(xp_s, dim=(2, 4), keepdim=True)
            prenorm_std = torch.std(xp_s, dim=(2, 4), keepdim=True)

        else:
            prenorm_mean = torch.mean(xp_s, dim=(2, 3), keepdim=True)
            prenorm_std = torch.std(xp_s, dim=(2, 3), keepdim=True)
        # (bs, bsp, 1, S, 1)
        # if True:
        # if (xp_s == 0).all(dim=(2, 3)).any():
        # print("bug before norm")
        # print(torch.argwhere((xp_s == 0).all(dim=(2, 3))))
        # breakpoint()

        xp_s_norm = (xp_s - prenorm_mean) / prenorm_std
        # if True:
        # if (xp_s_norm == 0).all(dim=(2, 3)).any():
        # print("bug before features")
        # print(torch.argwhere((xp_s_norm == 0).all(dim=(2, 3))))
        # breakpoint()

        xp_s = xp_s_norm
        # (bs, bsp, T, S, K)
        # xp_s = xp_s.view(1, bsp, T, K)
        xp_s = xp_s.view(1, bsp, T, S * K)

        # print("before features")
        # torch.cuda.synchronize()
        # breakpoint()
        fx_s, jx_s_acc = self.features_pipeline(
            xp_s, idx=idx, with_jacobian=True
        )
        # fx: (bs, bsp, T, fs)
        # jx: (bs, bsp, T, K, fs)
        bs, bsp, T, fs = fx_s.shape
        # breakpoint()
        # torch.cuda.synchronize()
        # print("ok")
        # if True:
        # if (fx_s == 0).all(dim=(2, 3)).any():
        # print("bug after features")
        # print(torch.argwhere((fx_s == 0).all(dim=(2, 3))))
        # breakpoint()

        fx_s = fx_s.view(bs, bsp, T, self.G, -1)
        # fx: (bs, bsp, T, G, fsp)
        fsp = fx_s.shape[-1]

        torch.cuda.synchronize()
        # breakpoint()
        bg_params, _ = self.compute_params(x=fx_s)
        # torch.cuda.synchronize()
        mean = bg_params["mean"]
        # (bs, bsp, 1, G, fsp)

        C_inv = bg_params["C_inv"]
        # (bs, bsp, G, fsp, fsp)

        # breakpoint()
        # torch.cuda.synchronize()
        # print("before cholesky")
        Lt, info = torch.linalg.cholesky_ex(C_inv, upper=True)
        # (bs, bsp, G, fsp, fsp)
        # print(f"{info.sum()=}")
        # torch.cuda.synchronize()
        # print(f"after cholesky")

        Lt = Lt.view(bs, bsp, 1, self.G, fsp, fsp)
        # (bs, bsp, 1, G, fsp, fsp)

        Lt_jx = jx_s_acc.forward_left(Lt)
        # (bs, bsp, T_jx, G, fsp, SK)

        bs, bsp, T_jx, G, fsp, SK = Lt_jx.shape

        # (bs, bsp, T_jx, G, fsp, S, K)

        # (bs, bsp, T_jx, G, fsp, K)
        if True:
            Lt_jx = Lt_jx.view(bs, bsp, T_jx, G, fsp, S, SK // S)
            Lt_jx = Lt_jx[:, :, :, :, :, 0, :]
            prenorm_std = prenorm_std[:, :, :, 0, :]

        # (bs, bsp, 1, 1)

        Lt_jx = Lt_jx / prenorm_std.view(bs, bsp, 1, 1, 1, 1)
        # (bs, bsp, T_jx, G, fsp, K)

        fx_s = fx_s.view(bs, bsp, T, self.G, fsp)
        # (bs, bsp, T, 1, G, fsp)

        fx_s_c = fx_s - mean.view(bs, bsp, 1, self.G, fsp)
        # (bs, bsp, T, G, fsp)

        # torch.cuda.synchronize()
        # print(f"before matmul")
        # Lt_fx_c = Lt @ fx_s_c.unsqueeze(-1)
        Lt_fx_c = Lt @ fx_s_c.unsqueeze(-1)
        # (bs, bsp, T, G, fsp, 1)
        # torch.cuda.synchronize()
        # print(f"after matmul")

        Lt_fx_c = Lt_fx_c.squeeze(0)
        # (bsp, T, G, fsp, 1)

        Lt_jx = Lt_jx.squeeze(0)
        # (bsp, T_jx, G, fsp, K)

        if self.verbose:
            print("results_in")
            print(f"{Lt_fx_c.mean()=}")
            print(f"{Lt_jx.mean()=}")
            print(f"{Lt_fx_c.shape=}")
            print(f"{Lt_jx.shape=}")

        out = {"Lt_fx_c": Lt_fx_c, "Lt_jx": Lt_jx}

        return out

    @batchify
    def astrometry_2(
        self,
        z,
        j_coords_t,
        weights,
        shift,
        Lt_fx_c_idx,
        Lt_jx_idx,
        **psf_params,
    ):
        bsa, _ = z.shape
        # (bsa, 3)

        bsa, T, _, _ = j_coords_t.shape
        # (bsa, T, 2, 2)

        bsa, bsb, _ = shift.shape
        # (bsa, bsb, 2)
        # bsa = bs * n_sources, bsb= T *ppp * ppp

        bsa, bsb, G, fsp, _ = Lt_fx_c_idx.shape
        # (bsa, bsb, G, fsp, 1)

        bsa, bsb, G, fsp, K = Lt_jx_idx.shape
        # (bsa, bsb, G, fsp, K)

        bsa, T, G = weights.shape
        # (bsa, T, G)

        shift = shift + self.patch_size // 2
        y0 = shift[:, :, 0].view(1, -1)
        x0 = shift[:, :, 1].view(1, -1)
        psf, dg_dxt = self.psf_sampler_shift(
            y0=y0, x0=x0, **psf_params, return_jacobian=True
        )
        # psf: (1, bsa * bsb, ps, ps)
        # j_psf: (1, bsa * bsb, ps, ps, 2)
        # NOTE: g = alpha * h
        if False:
            cube_3d_viewer(psf.cpu().numpy()[0], range_hist="auto")
        # breakpoint()

        psf = psf.view(bsa, T, -1, self.K, 1)
        # (bsa, T, ppp**2, K, 1)
        ppp2 = psf.shape[2]

        dxt_dz = j_coords_t.view(bsa, T, 1, 2, 2)
        # (bsa, T, 1, 2, 2)

        dg_dxt = dg_dxt.view(bsa, T, ppp2, K, 2)
        # j_psf: (bsa, T, ppp** 2, K, 2)

        dg_dz = dg_dxt @ dxt_dz
        # (bsa, T, ppp** 2, K, 2)

        alphas = z[:, 0].view(bsa, 1, 1, 1, 1)
        # (bsa, 1, 1, 1, 1)

        dg_dz = torch.cat([psf, -alphas * dg_dz], dim=-1)
        # (bsa, T, ppp** 2, K, 3)

        dg_dz = dg_dz.view(bsa, T, ppp2, 1, self.K, 3)
        # (bsa, T, ppp** 2, 1,  K, 3)

        Lt_jx_idx = Lt_jx_idx.view(bsa, T, ppp2, self.G, fsp, self.K)
        # (bsa, T, ppp** 2, G, fsp, K)

        Lt_jx_dg_dz = Lt_jx_idx @ dg_dz
        # (bsa, T, ppp** 2, G, fsp, 3)

        Lt_jx_dg_dz_T = Lt_jx_dg_dz.permute(0, 1, 2, 3, 5, 4)
        # (bsa, T, ppp** 2, G, 3, fsp)

        hess = Lt_jx_dg_dz_T @ Lt_jx_dg_dz
        # (bsa, T, ppp** 2, G, 3, 3)

        Lt_fx_c_idx = Lt_fx_c_idx.view(bsa, T, ppp2, self.G, 1, fsp)
        # (bsa, T, ppp** 2, G, 1, fsp)

        # grad = - Lt_fx_c_idx @ Lt_jx_dg_dz
        grad = -Lt_fx_c_idx @ Lt_jx_dg_dz
        # (bsa, T, ppp** 2, G, 1, 3)

        grad = grad.view(bsa, T, ppp2, G, 3, 1)
        # (bsa, T, ppp** 2, G, 3, 1)

        weights = weights.view(bsa, T, 1, G, 1, 1)
        # (bsa, T, 1, G, 1, 1)

        grad = grad * weights
        # (bsa, T, ppp** 2, G, 3, 1)

        hess = hess * weights
        # (bsa, T, ppp** 2, G, 3, 3)

        grad = torch.sum(grad, dim=(1, 3))
        grad = torch.mean(grad, dim=1)
        # (bsa, 3, 1)

        hess = torch.sum(hess, dim=(1, 3))
        hess = torch.mean(hess, dim=1)
        # (bsa, 3, 3)

        out = {"grad": grad, "hess": hess}

        return out

    # def fit_params(self, xp, idx):

    #     bsp, T, S, d_pf = xp.shape
    #     xp = xp.view(1, bsp, T, self.S, self.K)

    #     xp_norm = xp

    #     bsp = xp.shape[1]
    #     # xp_norm = xp_s if xp_s else xp
    #     # (bs, bsp, T, C, K)

    #     prenorm_mean = torch.mean(xp_norm, dim=(2, 4), keepdim=True)
    #     prenorm_std = torch.std(xp_norm, dim=(2, 4), keepdim=True)
    #     # (bs, bsp, 1, S, 1)

    #     # print("TO CHANGE HERE !!!!!!")
    #     # prenorm_mean = torch.zeros_like(prenorm_mean)
    #     # prenorm_std = torch.ones_like(prenorm_std)

    #     xp = (xp - prenorm_mean) / prenorm_std
    #     # (bs, bsp, T, S, K)

    #     xp = xp.view(1, bsp, T, self.S * self.K)
    #     # (bs, bsp, T, S * K)

    #     fx, _ = self.features_pipeline(xp, idx=idx, with_jacobian=False)
    #     # fx: (bs, bsp, T, fs)
    #     # jx: (bs, bsp, T, K, fs)
    #     bs, bsp, T, fs = fx.shape

    #     fx = fx.view(bs, bsp, T, self.G, -1)
    #     # fx: (bs, bsp, T, G, fsp)

    #     # fx: (bs, bsp, T, G, fsp)
    #     bg_params, _ = self.compute_params(x=fx)

    #     bg_params["prenorm_mean"] = prenorm_mean.squeeze(0)
    #     bg_params["prenorm_std"] = prenorm_std.squeeze(0)
    #     bg_params["mean"] = bg_params["mean"].squeeze(0)
    #     bg_params["C_inv"] = bg_params["C_inv"].squeeze(0)

    #     return bg_params

    # @batchify
    # def get_log_likelihood(
    #     self, xp, idx, prenorm_mean, prenorm_std, mean, C_inv
    # ):
    #     bsp, T, S, d_pf = xp.shape
    #     xp = xp.view(bsp, T, self.S, self.K)

    #     # bsp = xp.shape[1]
    #     # xp_norm = xp_s if xp_s else xp
    #     # (bs, bsp, T, C, K)

    #     # prenorm_mean = torch.mean(xp_norm, dim=(2, 4), keepdim=True)
    #     # prenorm_std = torch.std(xp_norm, dim=(2, 4), keepdim=True)
    #     # (bs, bsp, 1, S, 1)

    #     # print("TO CHANGE HERE !!!!!!")
    #     # prenorm_mean = torch.zeros_like(prenorm_mean)
    #     # prenorm_std = torch.ones_like(prenorm_std)

    #     xp = (xp - prenorm_mean) / prenorm_std
    #     # (bs, bsp, T, S, K)

    #     xp = xp.view(1, bsp, T, self.S * self.K)
    #     # (bs, bsp, T, S * K)

    #     fx, _ = self.features_pipeline(xp, idx=idx, with_jacobian=False)
    #     # fx: (bs, bsp, T, fs)
    #     # jx: (bs, bsp, T, K, fs)
    #     bs, bsp, T, fs = fx.shape

    #     fx = fx.view(bs, bsp, T, self.G, -1)
    #     # fx: (bs, bsp, T, G, fsp)

    #     # C_inv: (bs, bsp, G, K, K)

    #     C_inv = C_inv.view(bs, bsp, 1, self.G, fs, fs)
    #     # C_inv: (bs, bsp, 1, G, K, K)

    #     mean = mean.view(bs, bsp, 1, self.G, fs)
    #     # fx: (bs, bsp, 1, G, fsp)

    #     fx_c = fx - mean
    #     # fx: (bs, bsp, T, G, fsp)

    #     fx_c = fx_c.view(bs, bsp, T, self.G, fs, 1)
    #     # C_inv: (bs, bsp, T, G, K, 1)

    #     ll = -0.5 * fx_c.transpose(-1, -2) @ C_inv @ fx_c + .5 * torch.logdet(C_inv).unsqueeze(2).unsqueeze(3)
    #     # C_inv: (bs, bsp, T, G, 1, 1)

    #     # ll = torch.mean(ll, dim=(2, 3))
    #     # C_inv: (bs, bsp, 1, 1)

    #     ll = torch.mean(ll, dim=3)
    #     # C_inv: (bs, bsp, T, 1, 1)

    #     # ll = ll.view(bsp)
    #     # C_inv: (bsp)

    #     ll = ll.view(bsp, T)
    #     # (bsp, T)

    #     return {"ll": ll}

    def _params_gaussian(self, x, mean=None):
        if self.grad_params:
            context = contextlib.nullcontext()
        else:
            context = torch.no_grad()
        with context:
            bs, bsp, T, G, fsp = x.shape
            T_t = T // 2
            x_tmp = x.view(bs, bsp, T_t, 2, G, fsp)
            assert G == 1

            if mean is None:
                mean = torch.mean(x_tmp, dim=2, keepdim=True)
                mean = mean.expand(-1, -1, T_t, -1, -1, -1)
                #mean = x.view(bs, bsp, -1, 2, 1, G).mean(dim=2)
                mean = mean.reshape(bs, bsp, 2*T_t, G, fsp)
                # (bsp, bsp, 1, G, fsp)
            else:
                mean = mean.view(bs, bsp, 2*T_t, 1, fsp)

            x_c = x - mean
            # (bs, bsp, T, G, fsp)
            # if (x_c == 0).all(dim=(2, 3, 4)).any():
            # print(torch.argwhere((x_c == 0).all(dim=(2, 3, 4))))
            # breakpoint()

            # S_hat = torch.einsum("bstgi,bstgj->bsgij", x_c, x_c) / T
            # (bs, bsp, G, K, K)

            x_c = x_c.view(bs, bsp, T, fsp)
            x_c_T = x_c.permute(0, 1, 3, 2)
            S_hat = x_c_T @ x_c / T
            # (bs, bsp, K, K)

            S_hat = S_hat.unsqueeze(2)
            # (bs, bsp, G, K, K)

            # tr_S2 = torch.einsum("bsgii->bsg", S_hat**2)
            # tr2_S = torch.einsum("bsgii->bsg", S_hat) ** 2
            # tr_SS = torch.einsum("bsgii->bsg", S_hat @ S_hat)

            tr_S2 = torch.diagonal(S_hat**2, dim1=-2, dim2=-1).sum(dim=-1)
            tr2_S = torch.diagonal(S_hat, dim1=-2, dim2=-1).sum(dim=-1) ** 2
            tr_SS = torch.diagonal(S_hat @ S_hat, dim1=-2, dim2=-1).sum(dim=-1)

            # T_eff = T_eff[..., None]
            # torch.cuda.synchronize()

            num = tr_SS + tr2_S - 2 * tr_S2
            den = (T + 1) * (tr_SS - tr_S2)
            # den = torch.maximum(den, )
            if self.rho is None:
                with torch.no_grad():
                    rho = torch.clip(num / den, 0, 1)
                    self.rho=rho
            rho = self.rho
            # rho = torch.clip(num / den, 0, 1)
            rho = rho.view(bsp, G, 1, 1)
            # (bs, bsp, G, 1, 1)

            # diag_flat = torch.einsum("bsgii->bsgi", S_hat)
            diag_flat = torch.diagonal(S_hat, dim1=-2, dim2=-1)
            # (bs, bsp, G, K)

            diag = torch.diag_embed(diag_flat)
            # (bs, bsp, G, K, K)

            eye = (
                torch.eye(fsp, device=x.device).float().view(1, 1, 1, fsp, fsp)
            )
            # (1, 1, 1, K, K)

            # C_hat = (1 - rho) * S_hat + rho * diag + eye * 1e-2
            C_hat = (1 - rho) * S_hat + rho * diag
            # (bs, bsp, G, K, K)

            metrics = {}
            rho_mean = torch.mean(rho)
            # print(f"{rho_mean=}")

            # C_inv, info = torch.linalg.inv(C_hat)
            if C_hat.isnan().any():
                breakpoint()
            if C_hat.isinf().any():
                breakpoint()
            # torch.cuda.synchronize()
            try:
                # C_inv = torch.linalg.inv(C_hat)
                C_inv, info = torch.linalg.inv_ex(C_hat)
                # (bs, bsp, G, K, K)
                # breakpoint()

                info = (info > 0).float()
                mask_info = (1 - info).float().view(bs, bsp, G, 1, 1)
                n_issues = info.sum()
                # print(f"n_issues: {n_issues} / {mask_info.numel()}")
                if n_issues > 0:
                    breakpoint()
                # breakpoint()
                metrics["ratio_issues"] = info.sum() / info.numel()

                eye = torch.eye(fsp, device=x.device).view(1, 1, 1, fsp, fsp)
                # breakpoint()
                C_inv = C_inv * mask_info + eye * (1 - mask_info)
                # breakpoint()
            except Exception as e:
                print(e)
                breakpoint()
                raise

            C_inv = C_inv.view(bs, bsp, 1, G, fsp, fsp)
            # (bs, bsp, 1, G, fsp, fsp)

            # mean:(bsp, bsp, 1, G, fsp)

            bg_params = {"mean": mean, "C_inv": C_inv}

        return bg_params, metrics

    def fit_params(self, xp, idx):

        bsp, T, S, d_pf = xp.shape
        xp = xp.view(1, bsp, T, self.S, self.K)

        xp_norm = xp

        bsp = xp.shape[1]
        # xp_norm = xp_s if xp_s else xp
        # (bs, bsp, T, C, K)

        prenorm_mean = torch.mean(xp_norm, dim=(2, 4), keepdim=True)
        prenorm_std = torch.std(xp_norm, dim=(2, 4), keepdim=True)
        # (bs, bsp, 1, S, 1)

        # print("TO CHANGE HERE !!!!!!")
        prenorm_mean = torch.zeros_like(prenorm_mean)
        prenorm_std = torch.ones_like(prenorm_std)

        xp = (xp - prenorm_mean) / prenorm_std
        # (bs, bsp, T, S, K)

        xp = xp.view(1, bsp, T, self.S * self.K)
        # (bs, bsp, T, S * K)

        fx, _ = self.features_pipeline(xp, idx=idx, with_jacobian=False)
        # fx = xp
        # fx: (bs, bsp, T, fs)
        # jx: (bs, bsp, T, K, fs)
        # fx = xp
        bs, bsp, T, fs = fx.shape
        fx = fx.view(bs, bsp, T, self.G, -1)
        # fx: (bs, bsp, T, G, fsp)

        # fx: (bs, bsp, T, G, fsp)
        bg_params, _ = self.compute_params(x=fx)

        bg_params["prenorm_mean"] = prenorm_mean.squeeze(0)
        bg_params["prenorm_std"] = prenorm_std.squeeze(0)
        bg_params["mean"] = bg_params["mean"].squeeze(0)
        bg_params["C_inv"] = bg_params["C_inv"].squeeze(0)
        bg_params["patches"] = fx
        
        return bg_params

    def fit_params_noweight(self, xp, idx):

        bsp, T, S, d_pf = xp.shape
        xp = xp.view(1, bsp, T, self.S, self.K)

        bsp = xp.shape[1]

        # prenorm_mean = torch.zeros_like(prenorm_mean)
        # prenorm_std = torch.ones_like(prenorm_std)
        # xp = (xp - prenorm_mean) / prenorm_std
        # xp = xp.view(1, bsp, T, self.S * self.K)
        fx = xp.view(1, bsp, T, self.G, -1)
        bg_params, _ = self.compute_params(x=fx)

        bg_params["mean"] = bg_params["mean"].squeeze(0)
        bg_params["C_inv"] = bg_params["C_inv"].squeeze(0)
        bg_params["patches"] = fx
        
        return bg_params
    
    @batchify
    def get_log_likelihood(
        self, xp, idx, prenorm_mean, prenorm_std, mean, C_inv
    ):
        bsp, T, S, d_pf = xp.shape
        xp = xp.view(bsp, T, self.S, self.K)

        # bsp = xp.shape[1]
        # xp_norm = xp_s if xp_s else xp
        # (bs, bsp, T, C, K)

        # prenorm_mean = torch.mean(xp_norm, dim=(2, 4), keepdim=True)
        # prenorm_std = torch.std(xp_norm, dim=(2, 4), keepdim=True)
        # (bs, bsp, 1, S, 1)

        # print("TO CHANGE HERE !!!!!!")
        # prenorm_mean = torch.zeros_like(prenorm_mean)
        # prenorm_std = torch.ones_like(prenorm_std)

        xp = (xp - prenorm_mean) / prenorm_std
        # (bs, bsp, T, S, K)

        xp = xp.view(1, bsp, T, self.S * self.K)
        # (bs, bsp, T, S * K)

        fx, _ = self.features_pipeline(xp, idx=idx, with_jacobian=False)
        # fx = xp
        # fx: (bs, bsp, T, fs)
        # jx: (bs, bsp, T, K, fs)
        #fx = xp
        bs, bsp, T, fs = fx.shape

        fx = fx.view(bs, bsp, T, self.G, -1)
        # fx: (bs, bsp, T, G, fsp)

        # C_inv: (bs, bsp, G, K, K)

        C_inv = C_inv.view(bs, bsp, 1, self.G, fs, fs)
        # C_inv: (bs, bsp, 1, G, K, K)

        mean = mean.view(bs, bsp, T, self.G, fs)
        # fx: (bs, bsp, 1, G, fsp)

        fx_c = fx - mean
        # fx: (bs, bsp, T, G, fsp)

        fx_c = fx_c.view(bs, bsp, T, self.G, fs, 1)
        # C_inv: (bs, bsp, T, G, K, 1)

        sign, logabsdet = torch.linalg.slogdet(C_inv)
        # logabsdet = -torch.sum(torch.log(torch.linalg.eigvals(C_inv)))
        ll = -0.5 * fx_c.transpose(-1, -2) @ C_inv @ fx_c + .5 * logabsdet.unsqueeze(2).unsqueeze(3)
        # q = self.quad_form_full(fx_c, C_inv)
        # ll = -0.5 * q + 0.5 * logabsdet.unsqueeze(2).unsqueeze(3) 

        # ll = torch.mean(ll, dim=(2, 3))
        # C_inv: (bs, bsp, 1, 1)

        ll = torch.mean(ll, dim=3)
        # C_inv: (bs, bsp, T, 1, 1)

        # ll = ll.view(bsp)
        # C_inv: (bsp)

        ll = ll.view(bsp, T)
        # (bsp, T)

        return {"ll": ll}
    
    # def extract_patches(self, xp, idx):
    #     bsp, T, S, d_pf = xp.shape
    #     xp = xp.view(1, bsp, T, self.S * self.K)
    #     fx, _ = self.features_pipeline(xp, idx=idx, with_jacobian=False)
    #     bs, bsp, T, fs = fx.shape
    #     fx = fx.view(bs, bsp, T, self.G, -1)
    #     bg_params = {"patches": fx}
    #     return bg_params

    @batchify
    def get_features(self, xp, xp_s=None, idx=None, **psf_params):
        bsp, T, S, d_pf = xp.shape
        # xp = xp.view(1, bsp, T, self.S, self.K)
        # if xp_s is not None:
        #     xp_s = xp_s.view(1, bsp, T, self.S, self.K)
        #     xp_norm = xp_s
        # else:
        #     xp_norm = xp
        # # else:
        # # xp_s = xp
        # # torch.inverse(torch.ones((1, 1), device="cuda:0"))

        # bsp = xp.shape[1]
        # # xp_norm = xp_s if xp_s else xp
        # # (bs, bsp, T, C, K)

        # prenorm_mean = torch.mean(xp_norm, dim=(2, 4), keepdim=True)
        # prenorm_std = torch.std(xp_norm, dim=(2, 4), keepdim=True)
        # # (bs, bsp, 1, S, 1)

        # prenorm_mean = torch.zeros_like(prenorm_mean)
        # prenorm_std = torch.ones_like(prenorm_std)

        # xp = (xp - prenorm_mean) / prenorm_std
        # # (bs, bsp, T, S, K)

        xp = xp.view(1, bsp, T, self.S * self.K)
        # (bs, bsp, T, C * K)

        fx, jx_acc = self.features_pipeline(xp, idx=idx, with_jacobian=False)
        # fx = xp
        # fx: (bs, bsp, T, fs)
        # jx: (bs, bsp, T, K, fs)
        _, bsp, T, fs = fx.shape
        fx = fx.view(bsp, T, fs)

        return {"fx": fx}