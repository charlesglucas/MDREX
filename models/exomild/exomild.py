import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import contextlib

from models.exomild.psf_sampler import BatchPSFSampler
from models.exomild.filter import LocalMasker, Filter
from models.exomild.utils import mask_to_coords, interpolate_2d, agg_dict
from models.exomild.injector import Injector
from models.exomild.psf_sampler import compute_trajectories
from models.exomild.spatial_term import SpatialTerm

from utils.viz import cube_3d_viewer
from utils.rotation import BatchRotationOperator
from datasets.transforms.injection import fit_gaussian_2d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# to fix lazy loading problem
torch.inverse(torch.ones((1, 1), device=device))

IMG_SIZE = 256


class ExoMILD(nn.Module):
    def __init__(
        self,
        n_channels,
        params_terms,
        interpolation,
        filter_alpha,
        skip_norm_unet,
        gate_type,
        is_residual,
        learnable_weights,
        pixel_wise_weights,
        oracle,
        use_local_max,
        n_iter,
        n_sources_max,
        iterative,
        linalg_library,
        grad_iterative,
        grad_astrometry,
        # weights_predictor,
        use_modelco_loss,
        rot_zero_init,
        # weights_spectrum=FLAT_SPECTRUM,
        # alpha_min=1e-10,
        alpha_min=1e-10,
        **kwargs,
    ):
        super().__init__()
        self.params_terms = params_terms
        self.interpolation = interpolation
        self.n_channels = n_channels
        self.use_modelco_loss = use_modelco_loss
        print(f"[ExoMILD] {self.use_modelco_loss=}")
        self.rot_zero_init = rot_zero_init

        all_terms = []
        G_tot = 0

        for i, params in enumerate(self.params_terms):
            params = dict(params)
            print(f"{i=} {params=}")
            term_type = params.pop("type")
            print(f"{params=}")
            n_repeat = params.pop("repeat")
            for _ in range(n_repeat):
                try:
                    G_tot += params["groups"]
                except KeyError:
                    pass
                if term_type == "spatial":
                    term = SpatialTerm(**params)
                else:
                    raise NotImplementedError
                all_terms.append(term)

        self.G_tot = G_tot
        print(f"[ExoMILD] {self.G_tot=}")

        self.all_terms = nn.ModuleList(all_terms)
        print(f"{all_terms=}")
        self.batch_rotation = None
        self.learnable_weights = learnable_weights
        print(f"[ExoMILD] {self.learnable_weights=}")
        # self.weights_terms = torch.zeros(len(self.all_terms), device=device)
        self.pixel_wise_weights = pixel_wise_weights
        print(f"[ExoMILD] {self.pixel_wise_weights=}")
        self.use_local_max = use_local_max
        print(f"[ExoMILD] {self.use_local_max=}")
        self.n_iter = n_iter

        self.n_sources_max = n_sources_max
        print(f"[ExoMILD] {self.n_sources_max=}")

        self.iterative = iterative
        print(f"[ExoMILD] {self.iterative=}")

        self.oracle = oracle
        print(f"[ExoMILD] {self.oracle=}")
        if self.pixel_wise_weights:
            self.weights_terms = torch.zeros(
                (1, 1, 1, self.G_tot, IMG_SIZE, IMG_SIZE), device=device
            )
        else:
            self.weights_terms = torch.zeros(
                (1, 1, 1, self.G_tot, 1, 1), device=device
            )
            # (bs, C, T, G, H, W)

        if self.learnable_weights:
            self.weights_terms = nn.Parameter(self.weights_terms)

        # self.filter = Filter(local_max=True, kernel_size=5)

        self.filter_alpha = filter_alpha
        if self.filter_alpha:
            self.filter = Filter(
                skip_norm_unet=skip_norm_unet,
                local_max=False,
                kernel_size=5,
                gate_type=gate_type,
                n_channels=self.n_channels,
                is_residual=is_residual,
            )
        self.psf_sampler_center = BatchPSFSampler(
            psf_size=10,
            mode="centered",
        )
        self.injector = Injector(psf_size=8, H=IMG_SIZE)

        self.register_buffer(
            "alpha_min", torch.tensor(alpha_min), persistent=False
        )

        self.local_masker = LocalMasker(
            kernel_size=5, size=IMG_SIZE, circle=True
        )

        print(f"[ExoMILD] {linalg_library=}")
        if linalg_library is not None and torch.cuda.is_available():
            torch.backends.cuda.preferred_linalg_library(linalg_library)
        self.grad_iterative = grad_iterative
        print(f"[ExoMILD] {self.grad_iterative=}")
        self.grad_astrometry = grad_astrometry
        print(f"[ExoMILD] {self.grad_astrometry=}")

        self.alpha_min = 1e-12 * torch.ones(1, dtype=torch.float32).to(device)

    def fit_psf(self, psf):
        # breakpoint()
        bs, C, h, w = psf.shape
        psf_params = []
        # c = -1
        for b in range(bs):
            for c in range(C):
                _psf = psf[b, c]
                _psf_params = fit_gaussian_2d(_psf.detach().cpu().numpy())
                _psf_params["offset"] = 0
                _psf_params.pop("x0")
                _psf_params.pop("y0")
                _psf_params = {
                    # k: torch.tensor(v, device=device).view(1, 1)
                    k: torch.tensor(
                        v, device=device, dtype=torch.float32
                    ).view(1)
                    for (k, v) in _psf_params.items()
                }
                psf_params.append(_psf_params)
        psf_params = agg_dict(psf_params)

        for k, v in psf_params.items():
            # print(f"{k=}")
            # psf_params[k] = v.view(bs, C, -1)
            # psf_params[k] = v.view(bs, -1)
            psf_params[k] = v.flatten()
        self.psf_params = psf_params
        return psf_params

    def sample_psf(self):
        # breakpoint()
        h_kernel, _ = self.psf_sampler_center(**self.psf_params)
        # (bs, 1, ks, ks)
        # return h_kernel.squeeze(0).squeeze(0)
        return h_kernel

    def forward(
        self,
        x,
        rot,
        psf,
        t=None,
        lbda=None,
        mask=None,
        s_0=None,
        skip_filter=False,
        # output="snr",
        output="alpha_sigma",
        **kwargs,
    ):
        bs, C, T, H, W = x.shape
        bs, T = rot.shape
        bs, C, h, w = psf.shape
        assert x.shape[1] == psf.shape[1]
        assert output in ["snr", "alpha_sigma", "sample"], f"{output=}"
        if lbda is not None:
            assert lbda.shape[1] == x.shape[1]

        psf_max = torch.amax(psf, dim=[2, 3], keepdim=True)
        # (bs, C)

        psf = psf / psf_max
        x = x / psf_max.unsqueeze(2)

        if s_0 is not None:
            s_0 = s_0 / psf_max.unsqueeze(2)

        # if True:
        # rot_range = torch.abs(rot[:, 0] - rot[:, -1])
        # print(f"{rot_range=}")

        psf_params = self.fit_psf(psf)
        # list of params

        if self.oracle:
            assert s_0 is not None

        out, metrics = self._forward_ff(
            x=x,
            x_s=s_0 if self.oracle else None,
            rot=rot,
            mask=mask,
            lbda=lbda,
            psf_params=psf_params,
        )
        alpha = out["alpha"]
        sigma = out["sigma_alpha"]

        alpha_raw = alpha

        if self.filter_alpha and (not skip_filter):
            snr = alpha / sigma
            alpha = self.filter(
                alpha=alpha,
                sigma=sigma,
                snr=snr,
                t=t,
            )

        if output == "snr":
            return alpha / sigma
        elif output == "sample":
            return {"alpha": alpha, "sigma": sigma, "alpha_raw": alpha_raw}

        return {"alpha": alpha, "sigma": sigma}

    def astrometry(self, x, z, rot, psf_params, lbda):
        """
        z: (bs,  K, [alpha, y, x]) ?
        """
        bs, n_sources, _ = z.shape
        bs, T, H, W = x.shape
        hist_z = [z]

        rot = rot[:, :] - rot[:, 0, None]

        z0 = z.clone()
        z_min = z0[:, :, 1:].round() - 0.7
        z_max = z0[:, :, 1:].round() + 0.7
        # z_min = z0[:, :, 1:] - 0.5
        # z_max = z0[:, :, 1:] + 0.5
        # print(f"{z_max=}")
        # print(f"{z_min=}")

        weights_terms = F.softmax(self.weights_terms, dim=-3)
        # (1, 1 | C, 1 | T, G_tot, H, W)

        if self.grad_astrometry:
            context = contextlib.nullcontext()
        else:
            context = torch.no_grad()

        for k in range(self.n_iter):
            # print(f"## iter={k}")
            # if self.verbose:
            # print(f"{k=}")
            # print(f"{z=}")
            with context:
                xt, dxt_dz = compute_trajectories(
                    coords=z[:, :, 1:],
                    rot=rot,
                    center=H // 2,
                    return_grad=True,
                )
                xt = xt.squeeze(-1)
                # (bs, n_sources, T, 2)
                # breakpoint()
                alphas = z[:, :, 0]
                # print(f"{alphas.flatten()=}")

                # print(f"{xt=}")
                obj = self.injector(
                    coords_t=xt, alphas=alphas, psf_params=psf_params
                )
                x_s = x - obj
                # (bs, T, H, W)
                # breakpoint()

                # weights_coords = interpolate_2d(
                # weights_terms[:, 0, 0], coords=xt.view(-1, 2)
                # )
                # breakpoint()
                if self.pixel_wise_weights:
                    weights_coords = interpolate_2d(
                        weights_terms[0, 0, 0, :, :, :], coords=xt.view(-1, 2)
                    )
                    # (bs * n_sources * T, G)
                else:
                    weights_coords = weights_terms.view(1, -1).expand(
                        bs * n_sources * T, -1
                    )
                    # (bs * n_sources * T, G)

                hessian = torch.zeros((bs, n_sources, 3, 3), device=device)
                grad = torch.zeros((bs, n_sources, 3, 1), device=device)

                G_cum = 0
                for i, term in enumerate(self.all_terms):
                    G = self.params_terms[i]["groups"]
                    G_new = G_cum + G
                    _weights = weights_coords[:, G_cum:G_new]

                    _grad, _hess = term.astrometry(
                        z=z,
                        # x=x,
                        x_s=x_s,
                        x=x,
                        # rot=rot,
                        # obj=obj,
                        coords_t=xt,
                        j_coords_t=dxt_dz,
                        weights=_weights,
                        psf_params=psf_params,
                        lbda=lbda,
                    )

                    grad = grad + _grad
                    hessian = hessian + _hess
                    G_cum = G_new

                hess_inv, info = torch.linalg.inv_ex(hessian)
                # info (bs, n_sources)
                # print(f"{info=}")
                mask_info = (1 - info)[..., None, None]
                # info (bs, n_sources, 1, 1)
                id_mat = (
                    torch.eye(3, device=device)
                    .view(1, 1, 3, 3)
                    .expand(bs, n_sources, -1, -1)
                )

                # breakpoint()
                hess_inv = hess_inv * mask_info + id_mat * (1 - mask_info)

                step = -hess_inv @ grad
                # print(f"{step[:, :, 1:]=}")

            step = step.squeeze(-1)
            # (bs, n_sources, 3)
            # print(f"{step=}")
            # if False:
            # step[:, :, 1:] = 0
            z_new = z + step
            coords = z[:, :, 1:]
            alpha = z[:, :, 0, None]
            dalpha = step[:, :, 0, None]
            coords_new = z_new[:, :, 1:]
            coords_new_clipped = torch.clip(
                coords_new, min=z_min + 0.01, max=z_max - 0.01
            )

            dcoords = torch.sqrt((coords_new - coords).pow(2).sum(-1))
            dcoords_clipped = torch.sqrt(
                (coords_new_clipped - coords).pow(2).sum(-1)
            )
            ratio = dcoords_clipped / (dcoords + 1e-12)
            ratio = ratio[:, :, None]

            alpha_new = alpha + ratio * dalpha

            z_new = torch.cat([alpha_new, coords_new_clipped], dim=-1)
            # z_new = torch.cat([alpha_new, coords], dim=-1)
            # dist_clipped = torch.sqrt(
            # step_clipped[:, :, 1:].pow(2).sum(dim=-1)
            # )
            # z_alpha = z_new[:, :, 0, None]
            # z_alpha_clipped = torch.clip(z_alpha, min=torch.tensor(1e-9))
            # z_alpha_clipped = z_alpha
            # z_clipped = torch.cat([z_alpha_clipped, z_coords_clipped], dim=-1)
            # z_new = z + ratio[:, :, None] * step
            # print(ratio)
            # breakpoint()

            # step_clipped = z_clipped - z
            # dist_clipped = torch.sqrt(
            # step_clipped[:, :, 1:].pow(2).sum(dim=-1)
            # )
            # dist = torch.sqrt(step[:, :, 1:].pow(2).sum(dim=-1))
            # ratio = dist_clipped / (dist + 1e-5)
            # # ratio = dist_clipped / dist
            # ratio = ratio.view(bs, n_sources, 1)
            # (bs, n_sources, 1)

            # dz = ratio * step
            # dist_dz = torch.sqrt(dz[:, :, 1:].pow(2).sum(dim=-1))
            # print(f"{dist_dz=}")
            # if (dist_dz > 1).any():
            # breakpoint()
            # z_new = z + dz
            # (bs, n_sources, 3)
            # print(f"{dz=}")
            # z = z_new

            # ensure that all flux are positives

            alpha = z_new[:, :, 0, None]
            alpha = torch.clip(alpha, min=self.alpha_min)
            z = torch.cat([alpha, z_new[:, :, 1:]], dim=-1)
            coords = z_new[:, :, 1:]
            # breakpoint()

            # if (coords < 0).any():
            # breakpoint()

            if (coords < z_min).any():
                breakpoint()
            if (coords > z_max).any():
                breakpoint()

            # z[:, :, 0] =
            # z[:, :, 0] = torch.clip(z[:, :, 0], min=self.alpha_min)
            # breakpoint()

            hist_z.append(z)

        return z, hist_z

    def _get_weights(self, x, rot):
        bs, C, T, H, W = x.shape

        weights_terms = F.softmax(self.weights_terms, dim=-3)
        # (1, 1 | C, 1 | T, G_tot, H, W)

        return weights_terms

    def _forward_ff(self, x, x_s, rot, psf_params, lbda, mask=None):
        """
        Return full frame map of alpha and sigma, the corresponding loss
        is patch-wise
        """
        assert rot.ndim == 2
        bs, C, T, H, W = x.shape
        bs, T = rot.shape

        if self.batch_rotation is None:
            self.batch_rotation = BatchRotationOperator(
                device=device,
                in_size=H,
                out_size=H,
                mode=self.interpolation,
                zero_init=self.rot_zero_init,
            )
            self.batch_rotation_mask = BatchRotationOperator(
                device=device,
                in_size=H,
                out_size=H,
                mode="bilinear",
                zero_init=self.rot_zero_init,
            )

        a = torch.zeros_like(x)
        b = torch.zeros_like(x)
        # (bs, C, T, H, W)

        weights_terms = self._get_weights(x, rot)
        # out: (1, C, T, G, H, W)
        G_cum = 0
        metrics = []
        for i, term in enumerate(self.all_terms):
            # print(f"{i=} {self.params_terms[i]=}")
            _a, _b, _metrics = term.get_ab(
                x=x,
                x_s=x_s,
                rot=rot,
                psf_params=psf_params,
                lbda=lbda,
            )
            metrics.append(_metrics)
            # a: (bs, C, T_jx, G, H, W)
            # b: (bs, C, T,    G, H, W)
            G = _b.shape[3]
            G_new = G_cum + G

            weight = weights_terms[:, :, :, G_cum:G_new, :, :]
            # (1, 1, 1, G, H, W)

            # a = a + torch.sum(_a * weight, dim=2)
            # b = b + torch.sum(_b * weight, dim=2)
            a = a + torch.sum(_a * weight, dim=3)
            b = b + torch.sum(_b * weight, dim=3)
            # (bs, C, T, H, W)

            G_cum = G_new
        assert G_new == self.G_tot

        # if mask is not None:
        # mask = mask.float()
        # print("no mask used")
        # a = a * mask
        # b = b * mask
        bs, C, T, H, W = b.shape
        bs, C, _, H, W = a.shape
        a = a.view(bs, C * T, H, W)
        b = b.view(bs, C * T, H, W)
        rot_c = rot.view(1, 1, T)
        rot_c = rot_c.expand(-1, C, -1)
        rot_c = rot_c.reshape(1, C * T)

        # if a.isnan().any():
        # breakpoint()
        # if (a <= 0).any():
        # breakpoint()

        a = (
            self.batch_rotation.forward(
                x=torch.sqrt(a), rot=-rot_c, mode="warp"
            )
            ** 2
        )
        # if a_after.isnan().any():
        # breakpoint()
        # a = a_after

        b = self.batch_rotation.forward(x=b, rot=-rot_c, mode="warp")
        # (bs, C * T, H, W)

        a = a.view(bs, C, T, H, W)
        b = b.view(bs, C, T, H, W)

        if False:
            # snr = b
            cube_3d_viewer(b.detach()[0, 0].numpy(), range_hist="auto")

        if mask is not None:
            # NOTE: masking should be done after rotation, so that a > 0
            # (no weird interpolation effect)
            mask = mask.float()
            mask = mask.view(1, T, H, W)
            mask_rot = self.batch_rotation_mask.forward(
                x=mask, rot=-rot, mode="warp"
            )
            mask_rot = mask_rot.view(1, 1, T, H, W)
            a = a * mask_rot
            b = b * mask_rot

        b = torch.sum(b, dim=2)
        a = torch.sum(a, dim=2) + 1e-8

        alpha = b / a
        sigma_alpha = 1 / torch.sqrt(a)
        # if self.training:

        return {"alpha": alpha, "sigma_alpha": sigma_alpha}, metrics

    def _forward_selection(self, x, rot, psf):
        """
        Return full frame map of alpha and sigma, the corresponding loss
        is patch-wise
        """
        assert x.ndim == 4
        assert rot.ndim == 2
        assert psf.ndim == 3
        bs, T, H, W = x.shape
        bs, T = rot.shape
        psf, h, w = psf.shape

        alpha = torch.zeros(bs, self.n_sources_max)
        sigma_alpha = torch.ones(bs, self.n_sources_max)

        return {"alpha": alpha, "sigma_alpha": sigma_alpha}

    def get_target(
        self,
        pred=None,
        mask=None,
        rot=None,
        y=None,
        s_0=None,
        **kwargs,
    ):
        bs, _, T, H, W = mask.shape

        mask = mask.view(1, T, H, W)
        mask_rot = self.batch_rotation_mask.forward(
            x=mask.float(), rot=-rot, mode="warp"
        )
        mask_sum = torch.sum(mask_rot, dim=1)
        mask_target = (mask_sum >= 8).float()
        mask_target = mask_target.view(1, 1, 1, H, W)

        # NOTE: maybe include effect of mask in the target signal as well
        # (could improve stability)
        signal_gt_0 = y[:, :, 0] - s_0[:, :, 0]
        # (bs, C, H, W)

        bs, C, H, W = signal_gt_0.shape
        # (bs, C, H, W)
        h_kernel = self.sample_psf()
        # (C, 1, h, w)

        h_kernel = h_kernel[:, :, 1:, 1:]
        norm2 = torch.sum(h_kernel**2, dim=(1, 2, 3), keepdim=True)
        h_kernel = h_kernel / norm2

        assert h_kernel.shape == (C, 1, 9, 9), f"{h_kernel.shape=}"
        alpha_target = F.conv2d(
            signal_gt_0, weight=h_kernel, padding="same", groups=C
        )
        # (bs, C, H, W)

        return {
            "alpha_target": alpha_target,
            "mask_target": mask_target,
        }

    # def fit_params(self, x, lbda):
    #     bs, C, T, H, W = x.shape

    #     for i, term in enumerate(self.all_terms):
    #         # print(f"{i=} {self.params_terms[i]=}")
    #         term.fit_params(
    #             x=x,
    #             lbda=lbda
    #         )

    # def get_log_likelihood(self, x, lbda, is_mean=False):
    #     bs, C, T, H, W = x.shape

    #     # ll = 0
    #     all_ll = []
    #     for i, term in enumerate(self.all_terms):
    #         # print(f"{i=} {self.params_terms[i]=}")
    #         _ll = term.get_log_likelihood(
    #             x=x,
    #             lbda=lbda,
    #         )
    #         all_ll.append(_ll)

    #     return all_ll

    def fit_params(self, x, lbda):
        params_all_terms = []
        for term in self.all_terms:
            params = term.fit_params(x, lbda)   # ← retourne un dict
            params_all_terms.append(params)
        return params_all_terms

    def fit_params_noweight(self, x, lbda):
        term = self.all_terms[0]
        params = term.fit_params_noweight(x, lbda)  
        return params
    
    def get_log_likelihood(self, x, lbda, params_all_terms):
        all_ll = []
        for term, params in zip(self.all_terms, params_all_terms):
            ll = term.get_log_likelihood(x, lbda, params)
            all_ll.append(ll)
        return all_ll

    def extract_patches(self, x, lbda):
        params_all_terms = []
        for term in self.all_terms:
            params = term.extract_patches(x, lbda)   # ← retourne un dict
            params_all_terms.append(params)
        return params_all_terms