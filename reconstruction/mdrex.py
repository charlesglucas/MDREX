from einops import rearrange
import torch
import hydra
import numpy as np
import torch.nn.functional as F
import utils.optm as optm
from utils.rotation import BatchRotationOperator
from models.exomild.exomild import ExoMILD
import models.exomild.engine as exomild_engine
import torch.nn as nn
import gc
import os
    
class PatchesHandler:
    def __init__(self, patch_size=8, stride=8):
        self.patch_size = patch_size
        self.stride = stride
        self.unfolder = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.folder = None
        self.divisor_inv = None

    def init_folder(self, H, W, device):
        self.folder = nn.Fold(output_size=(H, W), kernel_size=self.patch_size, stride=self.stride)
        divisor = self.folder(self.unfolder(torch.ones((1, H, W), device=device, dtype=torch.get_default_dtype())))[None, ...]
        self.divisor_inv = 1 / divisor

    def forward(self, x, lbda):
        bsp, C, T, H, W = x.shape
        x = self.wavelength_align(x,lbda)
        x = x.reshape(bsp, C*T, H, W)

        if self.folder is None:
            self.init_folder(H, W, device=x.device)
        x = self.unfolder(x).permute(0, 2, 1).reshape(-1, C*T, self.patch_size, self.patch_size)
        return x
    
    def wavelength_align(self, x, lbda):
        lbda_max = torch.amax(lbda, keepdim=True)
        coeff = lbda / lbda_max
        return self.interpolate(x, coeff)
    
    def interpolate(self, x, coeff):
        bs, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> c (b t) h w")
        coeff = coeff.view(C, 1, 1, 1).float()
        yyxx = 2 * np.mgrid[:H, :H] / (H - 1) - 1
        yyxx = torch.tensor(yyxx, device=x.device, dtype=torch.float32)[None, ...]
        grid = yyxx.permute(0, 2, 3, 1).to(x.device)
        offset = torch.tensor(1 / (H - 1), dtype=torch.float32)
        grid = (offset + (grid - offset) * coeff).to(x.dtype)   # float64 objective (to_double)
        x = F.grid_sample(x, grid=grid, align_corners=True, mode="bicubic", padding_mode="border")
        x = rearrange(x, "c (b t) h w ->  b c t h w", b=bs, t=T)
        return x.permute(0,1,2,4,3)

class ParamsGaussianHandler:
    def __init__(self):
        super().__init__()
        self.rho=None

    def forward(self, x):
        with torch.no_grad():
            bs, bsp, T, G, fsp = x.shape
            T_t = T // 2
            x_tmp = x.view(bs, bsp, T_t, 2, G, fsp)
            assert G == 1

            mean = torch.mean(x_tmp, dim=2, keepdim=True)
            mean = mean.expand(-1, -1, T_t, -1, -1, -1)
            mean = mean.reshape(bs, bsp, 2*T_t, G, fsp)
            
            x_c = x - mean
            
            x_c = x_c.view(bs, bsp, T, fsp)
            x_c_T = x_c.permute(0, 1, 3, 2)
            S_hat = x_c_T @ x_c / T
            S_hat = S_hat.unsqueeze(2)
            
            ## by channel
            # x_c = x_c.view(bs, bsp, T_t, 2, G, fsp)
            
            # # x_c : (bs, bsp, T_t, 2, G, fsp)
            # # On enlève G=1 pour simplifier comme ton code original
            # x_c = x_c[:, :, :, :, 0, :]   # (bs, bsp, T_t, 2, fsp)

            # # Séparation des deux moitiés temporelles
            # x1 = x_c[:, :, :, 0, :]   # (bs, bsp, T_t, fsp)
            # x2 = x_c[:, :, :, 1, :]   # (bs, bsp, T_t, fsp)

            # # (bs, bsp, fsp, fsp)
            
            # # --- Covariance moitié 1 ---
            # x1_T = x1.permute(0, 1, 3, 2)      # (bs, bsp, fsp, T_t)
            # S1 = (x1_T @ x1) / T_t             # (bs, bsp, fsp, fsp)

            # # --- Covariance moitié 2 ---
            # x2_T = x2.permute(0, 1, 3, 2)      # (bs, bsp, fsp, T_t)
            # S2 = (x2_T @ x2) / T_t             # (bs, bsp, fsp, fsp)

            # # Empilement des deux covariances
            # S_hat = torch.stack([S1, S2], dim=2)   # (bs, bsp, 2, fsp, fsp)

            # # Ajout de la dimension G=1 comme dans ton code original
            # S_hat = S_hat.unsqueeze(3)             # (bs, bsp, 2, 1, fsp, fsp)

            # diag_flat = torch.diagonal(S_hat, dim1=-2, dim2=-1)  # (bs,bsp,2,G,fsp)
            # diag = torch.diag_embed(diag_flat)                   # (bs,bsp,2,G,fsp,fsp)

            # termes pour rho
            tr_S2 = torch.diagonal(S_hat**2, dim1=-2, dim2=-1).sum(dim=-1)
            tr2_S = torch.diagonal(S_hat, dim1=-2, dim2=-1).sum(dim=-1)**2
            tr_SS = torch.diagonal(S_hat @ S_hat, dim1=-2, dim2=-1).sum(dim=-1)

            num = tr_SS + tr2_S - 2 * tr_S2
            den = (T + 1) * (tr_SS - tr_S2)   # T = C x T_frames samples: (T |Lambda| + 1) of the paper (was T_t + 1)

            if self.rho is None:
                with torch.no_grad():
                    rho = torch.clip(num / den, 0, 1)
                    self.rho = rho

            rho = self.rho  # (bs,bsp,2,G)
            rho = rho.view(bsp, G, 1, 1)
            # rho = rho.view(bs, bsp, 2, G, 1, 1)
            
            diag_flat = torch.diagonal(S_hat, dim1=-2, dim2=-1)  # (bs, bsp, G, K)
            diag = torch.diag_embed(diag_flat)

            # eye = (
            #     torch.eye(fsp, device=x.device).float().view(1, 1, 1, 1, fsp, fsp)
            # )
            # # (1, 1, 1, 1, K, K)

            eye = (
                torch.eye(fsp, device=x.device).float().view(1, 1, 1, fsp, fsp)
            )
            # (1, 1, 1, K, K)

            # shrinkage final
            C_hat = (1 - rho) * S_hat + rho * diag  # (bs,bsp,2,G,fsp,fsp)

            metrics = {}

            # Vérifications NaN / Inf
            if C_hat.isnan().any():
                breakpoint()
            if C_hat.isinf().any():
                breakpoint()

            try:
                # C_inv : (bs, bsp, 2, G, fsp, fsp)
                # info  : (bs, bsp, 2, G)
                C_inv, info = torch.linalg.inv_ex(C_hat)

                # info > 0 signifie inversion ratée
                info = (info > 0).float()
                mask_info = (1 - info).float().view(bs, bsp, G, 1, 1)

                # # masque pour remplacer les matrices non inversibles
                # # mask_info = 1 si OK, 0 si problème
                # mask_info = (1 - info).float().view(bs, bsp, 2, G, 1, 1)
                
                n_issues = info.sum()
                if n_issues > 0:
                    breakpoint()

                # metrics["ratio_issues"] = info.sum() / info.numel()
                
                eye = torch.eye(fsp, device=x.device).view(1, 1, 1, fsp, fsp)

                # # matrice identité pour fallback
                # eye = torch.eye(fsp, device=x.device).view(1, 1, 1, 1, fsp, fsp)

                # Remplacement des matrices non inversibles
                C_inv = C_inv * mask_info + eye * (1 - mask_info)

            except Exception as e:
                print(e)
                breakpoint()
                raise

            # # reshape final pour rester compatible avec ton pipeline
            # C_inv = C_inv.view(bs, bsp, 2, G, fsp, fsp)
            # C_hat = C_hat.view(bs, bsp, 2, G, fsp, fsp)
            
            C_inv = C_inv.view(bs, bsp, 1, G, fsp, fsp)
            C_hat = C_hat.view(bs, bsp, 1, G, fsp, fsp)

            # mean:(bsp, bsp, 1, G, fsp)

            bg_params = {"mean": mean, "C_inv": C_inv, "C_hat": C_hat}

        return bg_params, metrics

class MDREX:
    def __init__(self, y, rot, psf, mask, lbda, model_state=None, **cfg_model):
        super().__init__()
        self.exomild = ExoMILD(**cfg_model).to(y.device) 
        if model_state is not None: 
            self.exomild.load_state_dict(model_state)
        self.exomild.eval()  # ← désactive Dropout et fixe BatchNorm
        self.C, self.T, H = y.shape[1], y.shape[2], y.shape[3]
        self.batch_rotation = BatchRotationOperator(device=y.device, in_size=H, out_size=H, mode="bicubic", zero_init=True)
        self.psf = psf
        self.rot = rot
        self.mask = mask
        self.lbda = lbda
        self.dtype = torch.float32   # precision of the objective and its gradient (to_double: float64)
        # shrinkage coefficients of the covariances fixed (Algorithm 1 of the paper): computed once on the data y
        # (residual at x = 0) and kept constant in the objective and its gradient; False -> recomputed at each
        # evaluation on the residual y - A x (differentiated or not according to exomild_engine.DETACH_RHO)
        self.fixed_rho = True
        self.rho_cache = {}          # data key -> coefficients in call order (a few data sets: y, y + xi delta)
        # sparse term mu_sparse sum(x) (= sum|x| on x >= 0, the lower bound of VMLMB): its gradient is mu_sparse at
        # x = 0. True -> sum|x| as before 2026-11-07, whose autograd gradient is 0 at x = 0 (sign(0) = 0): a pixel at 0
        # then looks free to VMLMB and the gradient jumps by mu_sparse when it becomes positive, so that the
        # reconstructions stop with a projected gradient ~1/10 of its initial norm, on a non-stationary point
        self.sparse_abs = False
        # epsilon of the edge-preserving term sqrt(mean_c |grad x|^2 + eps^2) (absolute, in units of x): l2 (quadratic)
        # regime for |grad x| << eps, l1 (TV) regime for |grad x| >> eps
        self.tv_eps = 1e-6
        # "REXPACO-lite" data term instead of the ExoMILD terms (None: MD-REX): dict(patch=9, stride=4), see _lite_nll
        self.lite = None
        self.lite_rho = {}
        # robust per-frame weights in the ExoMILD terms (compound Gaussian as in REXPACO): number of fixed-point
        # iterations (0 = data term of the ExoMILD terms recomputed here without weights; None = original code path)
        self.robust = None
        self.robust_rho = {}
        self.robust_chunk = 1024

        # SURE handlers
        self.to_patches = None
        self.to_params = None

    @staticmethod
    def _data_key(y):
        y64 = y.detach().double()
        return (tuple(y.shape), y64.sum().item(), (y64 * y64).sum().item())

    def _fixed_rho(self, y):
        """Shrinkage coefficients of the data y, for every term and batch of patches (computed at the first call for
        these data: fit_params on the residual at x = 0, i.e. y)."""
        key = self._data_key(y)
        if key not in self.rho_cache:
            if len(self.rho_cache) >= 4:
                self.rho_cache.pop(next(iter(self.rho_cache)))
            exomild_engine.RHO_STATE = {"mode": "record", "store": []}
            try:
                with torch.no_grad():
                    for term in self.exomild.all_terms:
                        term.fit_params(y.to(self.dtype), self.lbda)
                self.rho_cache[key] = exomild_engine.RHO_STATE["store"]
            finally:
                exomild_engine.RHO_STATE = None
            torch.cuda.empty_cache()
        return self.rho_cache[key]

    def _lite_patches(self, x):
        """(1, C, T, H, W) -> (L, C, T, K) wavelength-aligned patches of size lite["patch"], stride lite["stride"]."""
        p, st = self.lite["patch"], self.lite["stride"]
        if self.to_patches is None:
            self.to_patches = PatchesHandler()
        z = self.to_patches.wavelength_align(x, self.lbda)           # (1, C, T, H, W)
        _, C, T, H, W = z.shape
        q = F.unfold(z.reshape(C * T, 1, H, W), kernel_size=p, stride=st)   # (C*T, K, L)
        return q.view(C, T, p * p, -1).permute(3, 0, 1, 2)          # (L, C, T, K)

    def _lite_nll(self, diff, y):
        """REXPACO-lite data term: one Gaussian per patch (no learned projection), mean per channel constant in time,
        spatial covariance shared by the two channels (samples = frames x channels) shrunk towards its diagonal with
        the coefficient rho fixed on the data y (Algorithm 1). Means and covariances are re-estimated on diff = y - A x
        and differentiated (as the ExoMILD terms). Returns sum over patches and samples of 1/2 (z^T C^-1 z + log|C|).
        lite["robust"] = n > 0: robust per-frame weights as in REXPACO (compound Gaussian, Flasseur et al. 2021, 2024):
        Cov(v_{l,s}) = sigma2_{l,s} C_l for each patch l and sample s (frame, channel); mean (weights 1 / sigma2),
        covariance and scales estimated jointly by n fixed-point iterations from sigma2 = 1 (unrolled, differentiated);
        data term 1/2 sum (z^T C^-1 z / sigma2 + K log sigma2 + log|C|)."""
        q = self._lite_patches(diff)                                  # (L, C, T, K)
        L, C, T, K = q.shape
        N = C * T
        key = self._data_key(y)
        if key not in self.lite_rho:
            with torch.no_grad():
                q0 = self._lite_patches(y.to(self.dtype))
                q0 = (q0 - q0.mean(dim=2, keepdim=True)).reshape(L, N, K)
                S0 = q0.transpose(1, 2) @ q0 / N
                d0 = torch.diagonal(S0, dim1=-2, dim2=-1)
                tr_S2, tr2_S, tr_SS = (d0 ** 2).sum(-1), d0.sum(-1) ** 2, (S0 * S0).sum((-2, -1))
                self.lite_rho = {key: torch.clip((tr_SS + tr2_S - 2 * tr_S2) / ((N + 1) * (tr_SS - tr_S2)), 0, 1).view(L, 1, 1)}
        rho = self.lite_rho[key]
        n_rob = int(self.lite.get("robust", 0))
        w = torch.ones((L, C, T, 1), dtype=q.dtype, device=q.device)  # 1 / sigma2
        for it in range(n_rob + 1):
            m = (w * q).sum(dim=2, keepdim=True) / w.sum(dim=2, keepdim=True)   # weighted mean per channel
            zc = (q - m).reshape(L, N, K)
            ww = w.reshape(L, N, 1)
            S = (zc * ww).transpose(1, 2) @ zc / N
            d = torch.diagonal(S, dim1=-2, dim2=-1)
            Ch = (1 - rho) * S + rho * torch.diag_embed(d)
            Lc = torch.linalg.cholesky(Ch)
            mz = (zc.transpose(1, 2) * torch.cholesky_solve(zc.transpose(1, 2), Lc)).sum(dim=1)   # z^T C^-1 z, (L, N)
            if it < n_rob:
                s2 = mz / K
                s2 = torch.clamp(s2, min=1e-6 * s2.detach().median(dim=1, keepdim=True).values)
                w = (1 / s2).reshape(L, C, T, 1)
        logdet = 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum()
        if n_rob == 0:
            return 0.5 * mz.sum() + 0.5 * N * logdet
        wf = w.reshape(L, N)
        return 0.5 * (mz * wf).sum() - 0.5 * K * torch.log(wf).sum() + 0.5 * N * logdet

    def _lite_nll_spec(self, diff, y):
        """REXPACO-lite with spectral covariance (REXPACO ASDI, Flasseur et al. 2024, Eqs. 5-12): for each patch l and
        frame t, the K x C residual V_{l,t} (K pixels, C channels) is N(M_l, sigma2_{l,t} C_spec_l (x) C_spat_l), M_l the
        mean per pixel and channel (weights 1 / sigma2), C_spat (K x K) shrunk towards its diagonal (rho fixed on y),
        C_spec (C x C, trace C), sigma2 = 1 unless lite["robust"] > 0. Joint estimation by lite["flipflop"] alternations
        (flip-flop, Lu & Zimmerman 2005), unrolled and differentiated. Data term
        1/2 sum_{l,t} [tr(C_spat^-1 Z C_spec^-1 Z^T) / sigma2 + K C log sigma2 + C log|C_spat| + K log|C_spec|]."""
        q = self._lite_patches(diff)                                  # (L, C, T, K)
        L, C, T, K = q.shape
        V = q.permute(0, 2, 3, 1)                                     # (L, T, K, C)
        key = self._data_key(y)
        if key not in self.lite_rho:
            with torch.no_grad():
                q0 = self._lite_patches(y.to(self.dtype))
                q0 = (q0 - q0.mean(dim=2, keepdim=True)).reshape(L, C * T, K)
                S0 = q0.transpose(1, 2) @ q0 / (C * T)
                d0 = torch.diagonal(S0, dim1=-2, dim2=-1)
                tr_S2, tr2_S, tr_SS = (d0 ** 2).sum(-1), d0.sum(-1) ** 2, (S0 * S0).sum((-2, -1))
                self.lite_rho = {key: torch.clip((tr_SS + tr2_S - 2 * tr_S2) / ((C * T + 1) * (tr_SS - tr_S2)), 0, 1).view(L, 1, 1)}
        rho = self.lite_rho[key]
        n_rob, n_ff = int(self.lite.get("robust", 0)), int(self.lite.get("flipflop", 3))
        eyeC = torch.eye(C, dtype=q.dtype, device=q.device)
        w = torch.ones((L, T, 1, 1), dtype=q.dtype, device=q.device)  # 1 / sigma2
        Cspec = eyeC.expand(L, C, C)
        for it in range(n_rob + 1):
            M = (w * V).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True)
            Z = V - M                                                 # (L, T, K, C)
            for _ in range(n_ff):
                Ps = torch.linalg.inv(Cspec)                          # (L, C, C)
                S = torch.einsum("ltkc,lcd,ltjd->lkj", Z * w, Ps, Z) / (T * C)
                d = torch.diagonal(S, dim1=-2, dim2=-1)
                Csp = (1 - rho) * S + rho * torch.diag_embed(d)
                Lsp = torch.linalg.cholesky(Csp)
                Y = torch.cholesky_solve(Z.reshape(L, T * K, C).reshape(L, T, K, C).permute(0, 2, 1, 3).reshape(L, K, T * C), Lsp)
                Y = Y.reshape(L, K, T, C).permute(0, 2, 1, 3)        # C_spat^-1 Z, (L, T, K, C)
                G = torch.einsum("ltkc,ltkd->lcd", Z * w, Y) / (T * K)
                Cspec = G * (C / torch.diagonal(G, dim1=-2, dim2=-1).sum(-1)).view(L, 1, 1)   # trace C
            Ps = torch.linalg.inv(Cspec)
            mz = torch.einsum("ltkc,lcd,ltkd->lt", Z, Ps, Y)          # tr(C_spat^-1 Z C_spec^-1 Z^T)
            if it < n_rob:
                s2 = mz / (K * C)
                s2 = torch.clamp(s2, min=1e-6 * s2.detach().median(dim=1, keepdim=True).values)
                w = (1 / s2).reshape(L, T, 1, 1)
        logdet_sp = 2 * torch.log(torch.diagonal(Lsp, dim1=-2, dim2=-1)).sum(-1)       # (L,)
        logdet_sc = torch.logdet(Cspec)                                                  # (L,)
        wf = w.reshape(L, T)
        return 0.5 * ((mz * wf).sum() - K * C * torch.log(wf).sum() + T * (C * logdet_sp + K * logdet_sc).sum())

    @staticmethod
    def _robust_nll_feats(f, rho, n_rob, C):
        """Compound-Gaussian data term on features f (P patches, N = T x C samples ordered (t, c), F features):
        Cov(f_{p,s}) = sigma2_{p,s} C_p, mean per channel (weights 1 / sigma2), C_p shrunk towards its diagonal (rho
        fixed), sigma2 by n_rob fixed-point iterations from 1 (unrolled). Returns
        1/2 sum (z^T C^-1 z / sigma2 + F log sigma2 + log|C|); n_rob = 0: the Gaussian term of the ExoMILD engine."""
        P, N, F = f.shape
        Tf = N // C
        f4 = f.view(P, Tf, C, F)
        w = torch.ones((P, Tf, C, 1), dtype=f.dtype, device=f.device)
        for it in range(n_rob + 1):
            m = (w * f4).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True)
            z = (f4 - m).reshape(P, N, F)
            S = (z * w.reshape(P, N, 1)).transpose(1, 2) @ z / N
            d = torch.diagonal(S, dim1=-2, dim2=-1)
            Lc = torch.linalg.cholesky((1 - rho) * S + rho * torch.diag_embed(d))
            mz = (z.transpose(1, 2) * torch.cholesky_solve(z.transpose(1, 2), Lc)).sum(dim=1)   # (P, N)
            if it < n_rob:
                s2 = mz / F
                s2 = torch.clamp(s2, min=1e-6 * s2.detach().median(dim=1, keepdim=True).values)
                w = (1 / s2).reshape(P, Tf, C, 1)
        wf = w.reshape(P, N)
        logdet = 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum()
        return 0.5 * ((mz * wf).sum() - F * torch.log(wf).sum() + N * logdet)

    def _robust_term(self, k, term, diff, y):
        """Data term of ExoMILD term k with robust per-frame weights, and its gradient w.r.t. diff (patches processed
        by chunks of self.robust_chunk: the statistics are per patch, so the chunking is exact)."""
        C = diff.shape[1]
        with torch.enable_grad():
            xp_full = term.to_patches(diff, lbda=self.lbda)            # (bsp, T*C, S, K), samples ordered (t, c)
        bsp, N, S_, K = xp_full.shape
        feats = lambda xp: term.engine.features_pipeline(xp.reshape(1, xp.shape[0], N, S_ * K), idx=None,
                                                          with_jacobian=False)[0][0]   # (P, N, F)
        key = (self._data_key(y), k)
        if key not in self.robust_rho:
            if len(self.robust_rho) >= 48:
                self.robust_rho.pop(next(iter(self.robust_rho)))
            with torch.no_grad():
                xp0 = term.to_patches(y.to(self.dtype), lbda=self.lbda)
                rhos = []
                for i in range(0, bsp, self.robust_chunk):
                    f0 = feats(xp0[i:i + self.robust_chunk])
                    P, _, F = f0.shape
                    z0 = (f0.view(P, N // C, C, F) - f0.view(P, N // C, C, F).mean(1, keepdim=True)).reshape(P, N, F)
                    S0 = z0.transpose(1, 2) @ z0 / N
                    d0 = torch.diagonal(S0, dim1=-2, dim2=-1)
                    tr_S2, tr2_S, tr_SS = (d0 ** 2).sum(-1), d0.sum(-1) ** 2, (S0 * S0).sum((-2, -1))
                    rhos.append(torch.clip((tr_SS + tr2_S - 2 * tr_S2) / ((N + 1) * (tr_SS - tr_S2)), 0, 1))
                self.robust_rho[key] = torch.cat(rhos).view(-1, 1, 1)
                del xp0
        rho = self.robust_rho[key]
        xp_leaf = xp_full.detach().requires_grad_(True)
        g_xp = torch.zeros_like(xp_leaf)
        val = 0.0
        for i in range(0, bsp, self.robust_chunk):
            with torch.enable_grad():
                nll = self._robust_nll_feats(feats(xp_leaf[i:i + self.robust_chunk]), rho[i:i + self.robust_chunk],
                                             int(self.robust), C)
                (g,) = torch.autograd.grad(nll, xp_leaf)
            g_xp += g
            val += float(nll.detach().double())
            del nll, g
        (g_diff,) = torch.autograd.grad(xp_full, diff, grad_outputs=g_xp)
        return val, g_diff

    def objective(self, x_disc, y, mu_sparse, mu_smooth):
        if self.lite is not None or self.robust is not None:
            return self._objective(x_disc, y, mu_sparse, mu_smooth)
        if not self.fixed_rho:
            return self._objective(x_disc, y, mu_sparse, mu_smooth)
        store = self._fixed_rho(y)
        exomild_engine.RHO_STATE = {"mode": "replay", "store": store, "i": 0}
        try:
            out = self._objective(x_disc, y, mu_sparse, mu_smooth)
            assert exomild_engine.RHO_STATE["i"] == len(store), "fixed shrinkage coefficients: calls do not match"
        finally:
            exomild_engine.RHO_STATE = None
        return out

    def _objective(self, x_disc, y, mu_sparse, mu_smooth):
        """Objective f(x) = data term + mu_sparse R_sparse(x) + mu_smooth R_smooth(x) and its gradient (numpy float64 /
        float32), exactly the function minimized by run_bfgs (also used by SUGAR for Hessian-vector products)."""
        x_tensor = torch.tensor(x_disc, dtype=self.dtype, device=y.device, requires_grad=True)
        with torch.set_grad_enabled(True):
            im = self.forward_model(x_tensor)
            diff = y - im
            # Data term: backprop term by term into a detached leaf so that
            # only one term's graph lives in memory at a time.
            diff_leaf = diff.detach().requires_grad_(True)
            # No torch.no_grad() on fit_params: the gradient also flows through the estimated means and
            # covariances (dependence of m_hat and C_hat on x, as in the original scripts). The parameters are
            # fitted term by term inside the loop, so that only one term's graph lives in memory at a time.
            n_terms = len(self.exomild.all_terms)
            phi = 0.0
            grad_diff = torch.zeros_like(diff_leaf)
            if self.lite is not None:
                nll = (self._lite_nll_spec if self.lite.get("spec") else self._lite_nll)(diff_leaf, y)
                (grad_diff,) = torch.autograd.grad(nll, diff_leaf)
                phi = float(nll.detach().double())
            if self.lite is None and self.robust is not None:
                for k, term in enumerate(self.exomild.all_terms):
                    v, g = self._robust_term(k, term, diff_leaf, y)
                    grad_diff += g / n_terms
                    phi += v / n_terms
                    del g
            for term in ([] if (self.lite is not None or self.robust is not None) else self.exomild.all_terms):
                p = term.fit_params(diff_leaf, self.lbda)
                ll = term.get_log_likelihood(diff_leaf, self.lbda, p)
                phi_term = torch.sum(-ll) / n_terms
                # Only d/d diff: .backward() would also compute (huge, unused)
                # gradients w.r.t. the ExoMILD parameters.
                (g,) = torch.autograd.grad(phi_term, diff_leaf)
                grad_diff += g
                # objective value accumulated in float64: in float32, |f| ~ 1e7-1e8 has a resolution of ~1-8,
                # coarser than the convergence test of VMLMB (ftol * |f|)
                phi += torch.sum(-ll.detach(), dtype=torch.float64).item() / n_terms
                del ll, phi_term, g, p
            torch.cuda.empty_cache()
            reg_sparse = mu_sparse * self.l2_l1_sparse_2d(x_tensor)
            reg_smooth = mu_smooth * self.l2_l1_edge_preserving_2d(x_tensor)
            # Chain rule through forward_model: d phi/dx = J^T (d phi/d diff)
            surrogate = torch.sum(diff * grad_diff) + reg_sparse + reg_smooth
            with torch.no_grad():
                x64 = x_tensor.detach().double()
                loss = (phi + mu_sparse * self.l2_l1_sparse_2d(x64).item()
                        + mu_smooth * self.l2_l1_edge_preserving_2d(x64).item())
                del x64
        (grad_x,) = torch.autograd.grad(surrogate, x_tensor, retain_graph=False, create_graph=False, allow_unused=False)
        fx = np.float64(loss)
        gx = grad_x.detach().cpu().numpy().astype(np.float64 if self.dtype == torch.float64 else np.float32)
        del loss, surrogate, grad_x, grad_diff, im, diff, diff_leaf, x_tensor
        gc.collect()
        torch.cuda.empty_cache()
        return fx, gx

    def run_bfgs(self, x_disc_0, y, mu_sparse, mu_smooth, ftol=1.0e-8, xtol=1.0e-6, gtol=1.0e-5, observer=None):
        """
        Run BFGS optimization for given regularization parameters and return the optimized solution, function value, gradient, and status.
        ftol: relative tolerance of VMLMB on the objective (|f - f_prev| <= ftol |f|). The objective is dominated by
        the data term (|f| ~ 1e7-1e8), so ftol=1e-8 stops as soon as an iteration gains less than ~0.1-1:
        use a smaller ftol (or 0, convergence then decided by gtol / xtol) if the solution depends on the run.
        xtol, gtol: relative tolerances of VMLMB on the variables and on the gradient (VMLMB defaults).
        The number of evaluations of the objective of the last call is stored in self.n_eval.
        """
        # --- objective + gradient ---
        self.n_eval = 0
        debug = bool(os.environ.get("MDREX_DEBUG"))  # diagnostic prints only, the computation is unchanged
        def fg(x_disc):
            self.n_eval += 1
            if debug:
                print(f"[fg {self.n_eval}] ||x||={np.linalg.norm(x_disc):.3e} max x={np.max(x_disc):.3e} "
                      f"finite={bool(np.all(np.isfinite(x_disc)))}", flush=True)
            # non-finite objective (trial step of the line search outside the domain where the covariances are
            # defined): +inf, so that the Armijo test fails and the line search of VMLMB backtracks (step x 0.2);
            # with NaN the comparisons are false and the next step is NaN. Unchanged whenever the objective is finite.
            try:
                fx, gx = self.objective(x_disc, y, mu_sparse, mu_smooth)
            except FloatingPointError as err:
                print(f"[fg {self.n_eval}] {err} -> f = +inf", flush=True)
                fx, gx = np.inf, np.zeros_like(x_disc, dtype=np.float32)
            if not np.isfinite(fx):
                print(f"[fg {self.n_eval}] non-finite objective ({fx}) -> f = +inf", flush=True)
                fx = np.inf
            if debug:
                print(f"[fg {self.n_eval}] f={fx:.6e} ||g||={np.linalg.norm(gx):.3e} "
                      f"finite g={bool(np.all(np.isfinite(gx)))}", flush=True)
            return fx, gx
        # observer(iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg): optional, called at each iteration (diagnostics
        # only, e.g. convergence curves), the optimization is unchanged
        xdisc_opt, fx, gx, status = optm.vmlmb(fg, x_disc_0, verb=1, lower=0, maxiter=100000, observer=observer, ftol=ftol, xtol=xtol, gtol=gtol)
        return xdisc_opt, fx, gx, status

    def to_double(self):
        """Objective and gradient computed in float64 (diagnostic of the numerical precision of the objective):
        ExoMILD model, PSF, mask, wavelengths and rotation grid in float64; y must be converted by the caller."""
        self.dtype = torch.float64
        self.exomild = self.exomild.double()
        self.psf, self.mask, self.lbda = self.psf.double(), self.mask.double(), self.lbda.double()
        self.rot = self.rot.double()
        H = self.batch_rotation.in_size
        self.batch_rotation = BatchRotationOperator(device=self.psf.device, in_size=H, out_size=H, mode="bicubic",
                                                    zero_init=True, dtype=torch.float64)
        return self

    ## Data term
    def forward_model(self, x_disc):
        im = x_disc.unsqueeze(0)   # (1, C, H, W)
        im = im.expand(self.T, -1, -1, -1)        # (T, C, H, W)
        im = im.permute(1, 0, 2, 3)          # (C, T, H, W)
        im = self.batch_rotation.forward(x=im, rot=self.rot.expand(self.C,-1))  # (C, T, H, W) .expand(C,-1)
        im = im.permute(1, 0, 2, 3)   
        im = F.conv2d(im, weight=self.psf, padding="same", groups=self.C)  # (T, C, H, W)
        im = im.permute(1, 0, 2, 3) 
        im = im * self.mask.unsqueeze(1)  # mask: (C, 1, H, W)
        return im.unsqueeze(0) # (1, C, T, H, W)

    ## Regularization terms
    def l2_l1_edge_preserving_2d(self, x, epsilon=None):
        epsilon = self.tv_eps if epsilon is None else epsilon
        # x : (C, H, W) 
        dx = x[:, :-1, 1:] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        dy = x[:, 1:, :-1] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        grad_sq = dx**2 + dy**2  # (C, H-1, W-1) 
        grad_sq = grad_sq.mean(dim=0)  # (H-1, W-1) 
        return torch.sum(torch.sqrt(grad_sq + epsilon**2))
    
    def l2_l1_sparse_2d(self, x):
        return torch.sum(torch.abs(x)) if self.sparse_abs else torch.sum(x)

    def fit_params_noweight(self, x):
        xp = self.to_patches.forward(x, self.lbda)
        
        bsp, T, S, d_pf = xp.shape
        fx = xp.view(1, bsp, T, 1, -1)
        
        bg_params, _ = self.to_params.forward(x=fx)

        bg_params["mean"] = bg_params["mean"].squeeze(0)
        bg_params["C_inv"] = bg_params["C_inv"].squeeze(0)
        bg_params["C_hat"] = bg_params["C_hat"].squeeze(0)
        bg_params["patches"] = fx
        
        return bg_params

    def compute_mc_sure(self, y, x_tensor_opt, xdisc_0, mu_sparse, mu_smooth, n_mc=1, ftol=1.0e-8, xtol=1.0e-6, seed=42):
        self.to_patches = PatchesHandler()
        self.to_params = ParamsGaussianHandler()

        C, T = y.shape[1], y.shape[2]
        Ax_tensor_opt = self.forward_model(x_tensor_opt)

        diff = y - Ax_tensor_opt
        params = self.fit_params_noweight(diff)
        diffp = params["patches"].squeeze(0) - params["mean"].squeeze(0)
        Cinv = params["C_inv"]
        bsp, _, _, fs = diffp.shape
        diff_vec = diffp.view(bsp, C, T, fs).unsqueeze(-1)
        Cx = Cinv @ diff_vec
        maha = (diff_vec.transpose(-1, -2) @ Cx).squeeze(-1).squeeze(-1)
        data_term = maha.sum().item()

        res_mean = torch.mean(diff, dim=2, keepdim=True)
        # res_mean = params["mean"].squeeze(0)
        # Chat = params["C_hat"].squeeze(0)
        # params_AX = self.fit_params_noweight(Ax_tensor_opt)
        # AXp = params_AX["patches"].squeeze(0)
        div_est = 0.0
        delta = 1e-1 * (y - y.median()).abs().median()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(n_mc):
            # drawn on the CPU: the CUDA generator gives different vectors on different GPU models for the same seed
            # (the couples of a grid computed on different clusters would not share the same Monte Carlo vector)
            v = torch.randn(y.shape, generator=gen, dtype=torch.float32).to(device=y.device, dtype=y.dtype)
            y_eps = y + delta * v
            (xdisc_opt_eps, fx, gx, status) = self.run_bfgs(xdisc_0, y_eps, mu_sparse, mu_smooth, ftol=ftol, xtol=xtol)
            x_tensor_eps = torch.tensor(xdisc_opt_eps, dtype=y.dtype, device=y.device)

            diff_eps = y_eps - self.forward_model(x_tensor_eps)
            # params_eps = self.fit_params_noweight(diff_eps)
            # res_mean_eps = params_eps["mean"].squeeze(0)
            res_mean_eps = torch.mean(diff_eps, dim=2, keepdim=True)
            

            # params_AXeps = self.fit_params_noweight(self.forward_model(x_tensor_eps))
            # AXepsp = params_AXeps["patches"].squeeze(0)
            # div_x = AXepsp - AXp + res_mean_eps - res_mean
            AXeps = self.forward_model(x_tensor_eps)
            div_x = AXeps - Ax_tensor_opt + res_mean_eps - res_mean
            # div_vec = div_x.view(bsp, C, T, fs).unsqueeze(-1)

            div_est += torch.sum(v * div_x) / delta
            # vp = self.to_patches.forward(v, self.lbda).view(bsp, C, T, fs).unsqueeze(-1)
            # div_est += torch.sum(vp * div_vec) / delta
        div_est /= n_mc

        # trace_term = Chat.diagonal(dim1=-2, dim2=-1).sum() * C * T
        trace_term = y.numel()
        patch_sure = data_term - trace_term + 2.0 * div_est

        print(
            f"patch_size={self.to_patches.patch_size}: "
            f"Data term={data_term:.4f}, "
            f"Trace term={trace_term:.4f}, "
            f"Divergence estimate={div_est.item():.4f}, "
            f"SURE={patch_sure.item():.4f}"
        )

        return patch_sure.item(), data_term, div_est.item() * 2.0