from einops import rearrange
import torch
import hydra
import numpy as np
import torch.nn.functional as F
import utils.optm as optm
from utils.rotation import BatchRotationOperator
from models.exomild.exomild import ExoMILD
import torch.nn as nn
    
class PatchesHandler:
    def __init__(self, patch_size=8, stride=8):
        self.patch_size = patch_size
        self.stride = stride
        self.unfolder = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.folder = None
        self.divisor_inv = None

    def init_folder(self, H, W, device):
        self.folder = nn.Fold(output_size=(H, W), kernel_size=self.patch_size, stride=self.stride)
        divisor = self.folder(self.unfolder(torch.ones((1, H, W), device=device, dtype=torch.float32)))[None, ...]
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
        grid = offset + (grid - offset) * coeff
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
            den = (T_t + 1) * (tr_SS - tr_S2)

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
        self.C, self.T, H = y.shape[1], y.shape[2], y.shape[3]
        self.batch_rotation = BatchRotationOperator(device=y.device, in_size=H, out_size=H, mode="bicubic", zero_init=True)
        self.psf = psf
        self.rot = rot
        self.mask = mask
        self.lbda = lbda

        # SURE handlers
        self.to_patches = None
        self.to_params = None

    def run_bfgs(self, x_disc_0, y, mu_sparse, mu_smooth):
        """
        Run BFGS optimization for given regularization parameters and return the optimized solution, function value, gradient, and status.
        """
        # --- objective + gradient ---
        def fg(x_disc):
            x_tensor = torch.tensor(x_disc, dtype=torch.float32, device=y.device, requires_grad=True)
            with torch.set_grad_enabled(True):
                im = self.forward_model(x_tensor)
                diff = y - im
                params = self.exomild.fit_params(diff, self.lbda)
                log_likelihood = self.exomild.get_log_likelihood(diff, self.lbda, params)
                phi = torch.stack([torch.sum(-ll) for ll in log_likelihood]).mean()
                reg_sparse = mu_sparse * self.l2_l1_sparse_2d(x_tensor)
                reg_smooth = mu_smooth * self.l2_l1_edge_preserving_2d(x_tensor) 
                loss = phi + reg_sparse + reg_smooth
            (grad_x,) = torch.autograd.grad(loss, x_tensor, retain_graph=False, create_graph=False, allow_unused=False)
            fx = loss.detach().cpu().numpy().astype(np.float32)
            gx = grad_x.detach().cpu().numpy().astype(np.float32)
            del loss, grad_x, im, diff, log_likelihood, params, x_tensor
            torch.cuda.empty_cache()
            return fx, gx
        xdisc_opt, fx, gx, status = optm.vmlmb(fg, x_disc_0, verb=1, lower=0, maxiter=100000, observer=None)
        return xdisc_opt, fx, gx, status

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
    def l2_l1_edge_preserving_2d(self, x, epsilon=1e-6): 
        # x : (C, H, W) 
        dx = x[:, :-1, 1:] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        dy = x[:, 1:, :-1] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        grad_sq = dx**2 + dy**2  # (C, H-1, W-1) 
        grad_sq = grad_sq.mean(dim=0)  # (H-1, W-1) 
        return torch.sum(torch.sqrt(grad_sq + epsilon**2))
    
    def l2_l1_sparse_2d(self, x):
        return torch.sum(torch.abs(x))

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

    def compute_mc_sure(self, y, x_tensor_opt, xdisc_0, mu_sparse, mu_smooth, n_mc=1):
        if self.to_patches is None or self.to_params is None:
            self.to_patches = PatchesHandler()
            self.to_params = ParamsGaussianHandler()
            
        C, T = y.shape[1], y.shape[2]

        # Compute transformed solution
        Ax_tensor_opt = self.forward_model(x_tensor_opt)

        # Data term
        diff = y - Ax_tensor_opt
        params = self.fit_params_noweight(diff)
        diffp = params["patches"].squeeze(0) - params["mean"].squeeze(0) # params["mean"].squeeze(0)
        bsp, _, _, fs = diffp.shape
        diff_vec = diffp.view(bsp, C, T, fs).unsqueeze(-1) # [bsp, C, T, fs, 1]
        data_term = diff_vec.pow(2).sum() 

        # Divergence term (MC)
        res_mean = params["mean"].squeeze(0)
        Chat = params["C_hat"].squeeze(0) # [bsp, G, fs, fs]
        Linvp = torch.linalg.cholesky(Chat, upper=False)
        params_AX = self.fit_params_noweight(Ax_tensor_opt)
        AXp = params_AX["patches"].squeeze(0)
        div_est = 0.0
        delta = 1e-1 * (y - y.median()).abs().median()
        for _ in range(n_mc): 
            v = torch.randn_like(y)
            y_eps = y + delta * v
            (xdisc_opt_eps, fx, gx, status) = self.run_bfgs(xdisc_0, y_eps, mu_sparse, mu_smooth)
            x_tensor_eps = torch.tensor(xdisc_opt_eps, dtype=torch.float32, device=y.device)

            diff_eps = y_eps - self.forward_model(x_tensor_eps)
            params_eps = self.fit_params_noweight(diff_eps)
            res_mean_eps =  params_eps["mean"].squeeze(0)
            params_AXeps = self.fit_params_noweight(self.forward_model(x_tensor_eps))
            AXepsp = params_AXeps["patches"].squeeze(0)
            div_x = AXepsp - AXp + res_mean_eps - res_mean
            div_vec = div_x.view(bsp, C, T, fs).unsqueeze(-1)

            vp = self.to_patches.forward(v,torch.tensor([1, 1], device=y.device)).view(bsp, C, T, fs).unsqueeze(-1)  
            div_est += torch.sum(vp * (Chat @ div_vec)) / delta
        div_est /= n_mc
        
        # Trace term
        trace_term = Chat.diagonal(dim1=-2, dim2=-1).sum()*C*T

        print(f"Data term: {data_term.item():.4f}, Trace term: {trace_term:.4f}, Divergence estimate: {div_est.item():.4f}")

        return (data_term - trace_term + 2.0 * div_est).item(), data_term.item(), div_est.item()*2