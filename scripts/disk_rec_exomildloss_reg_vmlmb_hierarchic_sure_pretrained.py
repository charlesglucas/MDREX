import gc
from pyexpat import model
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Configure GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from einops import rearrange
from matplotlib.colors import LogNorm
import torch
import hydra
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils.optm as optm
import torch.nn as nn

from models import get_model
from inference.inference import load_folder
from models.exomild.exomild import ExoMILD
from utils.rotation import BatchRotationOperator
from astropy.io import fits

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # 1. Real-world data loading and preprocessing
    # ------------------------------------------------------------

    # Load only selected frames
    path_folder = ROOT / "SyntheticDisks/DataAssessment/HIP_72192/2015-06-11/IRDIS/data"
    path_frame = ROOT / "SyntheticDisks/DataAssessment/HIP_72192/2015-06-11/IRDIS/frame_selection_vector/"
    hdul = fits.open(path_frame / "ird_sortframes_vector_dc-IRD_FRAME_SELECTION_VECTOR-frame_selection_vector.fits")
    frames = hdul[0].data

    # Indices of frames to keep
    frames = np.asarray(frames).reshape(-1)
    idx = np.where(frames == 1)[0]

    # Load data
    inputs = load_folder(path_folder=path_folder, use_centered=False, channel_sortframes=0, channel_idx=None,)
    y = inputs["y"].astype(np.float32)[:,idx,:,:] # (C, T, H, W) 
    C, T, H, W = y.shape
    print(f"{y.shape=}")
    lbda = inputs["lbdas"].astype(np.float32)
    print(f"{lbda=}")
    rot = inputs["rot"].astype(np.float32)[idx] 
    print(f"{rot.shape=}")
    psf = inputs["psf"].astype(np.float32)
    print(f"{psf.shape=}")
    

    # Convert to torch tensors
    y = torch.tensor(y, device=device).unsqueeze(0) # (b, C, T, H, W)
    lbda = torch.tensor(lbda, device=device).unsqueeze(0) # (b, C)
    rot = torch.tensor(rot, device=device).unsqueeze(0)
    psf = torch.tensor(psf, device=device)
    print(f"{y.shape=}")
    print(f"{lbda.shape=}")
    print(f"{rot.shape=}")
    print(f"{psf.shape=}")

    # Forward model preprocessing of PSF
    psf_size = psf.shape[-1]
    border = 15
    mask_out = torch.ones_like(psf)[0].bool()
    mask_out[border : psf_size - border, border : psf_size - border] = 0
    mean_0 = psf[0, mask_out].mean()
    mean_1 = psf[1, mask_out].mean()
    psf[0] = psf[0] - mean_0
    psf[1] = psf[1] - mean_1
    crop_size = 13
    psf_crop = psf[
        :,
        1 + (psf_size - crop_size) // 2 : 1 + (psf_size + crop_size) // 2,
        1 + (psf_size - crop_size) // 2 : 1 + (psf_size + crop_size) // 2,
    ]   
    psf_crop = psf_crop.view(2, 1, crop_size, crop_size) # (1, 1, crop, crop)
    device = y.device
    
    # Load coronograph mask
    path_coronograph = ROOT / "data/coronograph"
    k1k2_path = os.path.join(path_coronograph,"sphere_irdis_k1_k2_coronagraph_transmission_map.fits")
    with fits.open(k1k2_path, memmap=False) as hdul:
            mask = np.array(hdul[0].data, dtype=np.float32)
    deltaH = (mask.shape[1] - H) // 2; deltaW = (mask.shape[2] - W) // 2; 
    mask = mask[:, deltaH:deltaH+H, deltaW:deltaW+W] # (C, H, W)
    mask = torch.tensor(mask, device=device) # (C, H, W)

    # ------------------------------------------------------------
    # 2. Set-up forward model and regularization
    # ------------------------------------------------------------

    # Initialize rotation operator
    batch_rotation = BatchRotationOperator(device=device, in_size=H, out_size=H, mode="bicubic", zero_init=True,)

    # Forward model
    def forward(x_disc):
        # x_disc : (C, H, W)
        im = x_disc.unsqueeze(0)   # (1, C, H, W)
        im = im.expand(T, -1, -1, -1)        # (T, C, H, W)
        im = im.permute(1, 0, 2, 3)          # (C, T, H, W)
        im = batch_rotation.forward(x=im, rot=rot.expand(C,-1))  # (C, T, H, W)
        im = im.permute(1, 0, 2, 3)   
        im = F.conv2d(im, weight=psf_crop, padding="same", groups=C)  # (T, C, H, W)
        im = im.permute(1, 0, 2, 3) 
        im = im * mask.unsqueeze(1)  # mask: (C, 1, H, W)
        return im.unsqueeze(0) # (1, C, T, H, W)

    cfg_model = cfg.model
    cfg_model.repeats = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None
    cfg_model.n_channels = 2
    ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/1_asdi/2024-11-09_20-10-01/banger_ms_bs16_lr5e-4_unetnormal_aug_111_100_100_100_seed5/ckpt/ckpt_40000.pt"
    print(f"Loading checkpoint {ckpt_path}")
    new_repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    keep_indices = [i for i, v in enumerate(new_repeats) if v == 1]
    print("Conserved blocks :", keep_indices)
    state = torch.load(ckpt_path, map_location=device)["net"]
    state = {k.replace(".module.", "."): v for k, v in state.items()}
    state_cleaned = {}
    BLOCK_PREFIX = "model.blocks."   
    for key, val in state.items():
        if key.startswith(BLOCK_PREFIX):
            parts = key[len(BLOCK_PREFIX):].split(".")
            idx = int(parts[0])
            if idx in keep_indices:
                state_cleaned[key] = val
        else:
            state_cleaned[key] = val
    print("Conserved weights :", len(state_cleaned), "out of", len(state))
    cfg_model.repeats = new_repeats
    cfg_dict = dict(cfg_model)
    cfg_dict.pop('name', None)
    cfg_dict.pop('ckpt', None)
    model = get_model(name="exomild", ckpt=None, **cfg_dict)
    keys_to_remove = [k for k in state_cleaned.keys() if "weights_terms" in k or "features_pipeline.transforms.0.D" in k]
    for k in keys_to_remove:
        state_cleaned.pop(k, None)
    model.load_state_dict(state_cleaned, strict=False)
    model_state = model.state_dict()
    del model
    gc.collect()

    # Regularization terms
    def l2_l1_edge_preserving_2d(x, epsilon=10**(-6)): 
        dx = x[:, :-1, 1:] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        dy = x[:, 1:, :-1] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        grad_sq = dx**2 + dy**2  # (C, H-1, W-1) 
        grad_sq = grad_sq.mean(dim=0)  # (H-1, W-1) 
        return torch.sum(torch.sqrt(grad_sq + epsilon**2))
    
    def l2_l1_sparse_2d(x):
        return torch.sum(x)
    
    # Wavelength aligner
    def wavelength_align(x, lbda):
        lbda_max = torch.amax(lbda, keepdim=True)
        coeff = lbda / lbda_max
        return interpolate(x, coeff)
    
    def interpolate(x, coeff):
        bs, C, T, H, W = x.shape
        x = rearrange(x, "b c t h w -> c (b t) h w")
        coeff = coeff.view(C, 1, 1, 1).float()
        yyxx = 2 * np.mgrid[:H, :H] / (H - 1) - 1
        yyxx = torch.tensor(yyxx, device=device, dtype=torch.float32)[None, ...]
        grid = yyxx.permute(0, 2, 3, 1).to(device)
        offset = torch.tensor(1 / (H - 1), dtype=torch.float32)
        grid = offset + (grid - offset) * coeff
        x = F.grid_sample(x, grid=grid, align_corners=True, mode="bicubic", padding_mode="border")
        x = rearrange(x, "c (b t) h w ->  b c t h w", b=bs, t=T)
        return x.permute(0,1,2,4,3)
    
    # ------------------------------------------------------------
    # 3. Ground truth loading
    # ------------------------------------------------------------

    # Load disk
    flux = 1e-7
    path_disk = ROOT / "SyntheticDisks/DataAssessment/medium_ellipse/"
    hdul = fits.open(path_disk / "hid_fake_disk_image_medium_ellipse_0degrees.fits")
    data = hdul[0].data

    # Ensure the byte order is native
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128]
    with torch.no_grad():
        x_gt = flux*torch.tensor(datadisk, dtype=torch.float32, device=device) 
        x_gt = x_gt.unsqueeze(0).repeat(C, 1, 1)
        y = forward(x_gt) + y 

    # ------------------------------------------------------------
    # 4. MSE and SURE implementation
    # ------------------------------------------------------------

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
            x = wavelength_align(x,lbda)
            x = x.reshape(bsp, C*T, H, W)

            if self.folder is None:
                self.init_folder(H, W, device=x.device)
            x = self.unfolder(x).permute(0, 2, 1).reshape(-1, C*T, self.patch_size, self.patch_size)
            return x
 
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
                x_c = x_c.view(bs, bsp, T_t, 2, G, fsp)
                
                # x_c : (bs, bsp, T_t, 2, G, fsp)
                # On enlève G=1 pour simplifier comme ton code original
                x_c = x_c[:, :, :, :, 0, :]   # (bs, bsp, T_t, 2, fsp)

                # Séparation des deux moitiés temporelles
                x1 = x_c[:, :, :, 0, :]   # (bs, bsp, T_t, fsp)
                x2 = x_c[:, :, :, 1, :]   # (bs, bsp, T_t, fsp)

                # --- Covariance moitié 1 ---
                x1_T = x1.permute(0, 1, 3, 2)      # (bs, bsp, fsp, T_t)
                S1 = (x1_T @ x1) / T_t             # (bs, bsp, fsp, fsp)

                # --- Covariance moitié 2 ---
                x2_T = x2.permute(0, 1, 3, 2)      # (bs, bsp, fsp, T_t)
                S2 = (x2_T @ x2) / T_t             # (bs, bsp, fsp, fsp)

                # Empilement des deux covariances
                S_hat = torch.stack([S1, S2], dim=2)   # (bs, bsp, 2, fsp, fsp)

                # Ajout de la dimension G=1 comme dans ton code original
                S_hat = S_hat.unsqueeze(3)             # (bs, bsp, 2, 1, fsp, fsp)

                diag_flat = torch.diagonal(S_hat, dim1=-2, dim2=-1)  # (bs,bsp,2,G,fsp)
                diag = torch.diag_embed(diag_flat)                   # (bs,bsp,2,G,fsp,fsp)

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
                rho = rho.view(bs, bsp, 2, G, 1, 1)

                eye = (
                    torch.eye(fsp, device=x.device).float().view(1, 1, 1, 1, fsp, fsp)
                )
                # (1, 1, 1, 1, K, K)


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

                    # masque pour remplacer les matrices non inversibles
                    # mask_info = 1 si OK, 0 si problème
                    mask_info = (1 - info).float().view(bs, bsp, 2, G, 1, 1)
                    
                    n_issues = info.sum()
                    if n_issues > 0:
                        breakpoint()

                    # metrics["ratio_issues"] = info.sum() / info.numel()

                    # matrice identité pour fallback
                    eye = torch.eye(fsp, device=x.device).view(1, 1, 1, 1, fsp, fsp)

                    # Remplacement des matrices non inversibles
                    C_inv = C_inv * mask_info + eye * (1 - mask_info)

                except Exception as e:
                    print(e)
                    breakpoint()
                    raise

                # reshape final pour rester compatible avec ton pipeline
                C_inv = C_inv.view(bs, bsp, 2, G, fsp, fsp)
                C_hat = C_hat.view(bs, bsp, 2, G, fsp, fsp)

                # mean:(bsp, bsp, 1, G, fsp)

                bg_params = {"mean": mean, "C_inv": C_inv, "C_hat": C_hat}

            return bg_params, metrics
        
    to_patches = PatchesHandler()
    to_params = ParamsGaussianHandler()

    def fit_params_noweight(x, lbda):
        xp = to_patches.forward(x, lbda)
        
        bsp, T, S, d_pf = xp.shape
        fx = xp.view(1, bsp, T, 1, -1)
        
        bg_params, _ = to_params.forward(x=fx)

        bg_params["mean"] = bg_params["mean"].squeeze(0)
        bg_params["C_inv"] = bg_params["C_inv"].squeeze(0)
        bg_params["C_hat"] = bg_params["C_hat"].squeeze(0)
        bg_params["patches"] = fx
        
        return bg_params

    def compute_mc_sure(y, x_tensor_opt, xdisc_0, mu_sparse, mu_smooth, exomild, n_mc=1):
        # Compute transformed solution
        Ax_tensor_opt = forward(x_tensor_opt)

        # Data term
        diff = y - Ax_tensor_opt
        params = fit_params_noweight(diff, lbda)
        diffp = params["patches"].squeeze(0) - params["mean"].squeeze(0) # params["mean"].squeeze(0)
        bsp, _, _, fs = diffp.shape
        diff_vec = diffp.view(bsp, C, T, fs).unsqueeze(-1) # [bsp, C, T, fs, 1]
        # r = bsp * fs / (H * W)
        data_term = diff_vec.pow(2).sum() # /r
        # data_term = data_term / torch.tensor(len(params))


        # Divergence term (MC)
        res_mean = params["mean"].squeeze(0)
        Chat = params["C_hat"].squeeze(0) # [bsp, G, fs, fs]
        Linvp = torch.linalg.cholesky(Chat, upper=False)
        params_AX = fit_params_noweight(Ax_tensor_opt, lbda)
        AXp = params_AX["patches"].squeeze(0)
        div_est = 0.0
        delta = 1e-1 * (y - y.median()).abs().median()
        for _ in range(n_mc): 
            v = torch.randn_like(y)
            y_eps = y + delta * v
            (xdisc_opt_eps, fx, gx, status) = run_bfgs(xdisc_0, y_eps, mu_sparse, mu_smooth, exomild)
            x_tensor_eps = torch.tensor(xdisc_opt_eps, dtype=torch.float32, device=device)

            diff_eps = y_eps - forward(x_tensor_eps)
            params_eps = fit_params_noweight(diff_eps, lbda)
            res_mean_eps =  params_eps["mean"].squeeze(0)
            params_AXeps = fit_params_noweight(forward(x_tensor_eps), lbda)
            AXepsp = params_AXeps["patches"].squeeze(0)
            div_x = AXepsp - AXp + res_mean_eps - res_mean
            div_vec = div_x.view(bsp, C, T, fs).unsqueeze(-1)

            vp = to_patches.forward(v,torch.tensor([1, 1], device=device)).view(bsp, C, T, fs).unsqueeze(-1)  
            div_est += torch.sum(vp * (Chat @ div_vec)) / delta
        div_est /= n_mc
        
        # Trace term
        trace_term = Chat.diagonal(dim1=-2, dim2=-1).sum()*C*T


        print(f"Data term: {data_term.item():.4f}, Trace term: {trace_term:.4f}, Divergence estimate: {div_est.item():.4f}")

        return (data_term - trace_term + 2.0 * div_est).item(), data_term.item(), div_est.item()*2

    def mahalanobis_mse(y, x_gt, x_tensor_opt):
        diffp = to_patches.forward(forward(x_gt) - forward(x_tensor_opt), lbda)
        mse_total = diffp.pow(2).sum() 
        return mse_total.item()


    # ------------------------------------------------------------
    # 5. BFGS scheme for one set of regularization parameters
    # ------------------------------------------------------------

    def run_bfgs(x_disc_0, y, mu_sparse, mu_smooth, exomild):
        """
        Run BFGS optimization for given regularization parameters and return the optimized solution, function value, gradient, and status.
        """
        # --- objective + gradient ---
        def fg(x_disc):
            x_tensor = torch.tensor(x_disc, dtype=torch.float32, device=device, requires_grad=True)
            with torch.set_grad_enabled(True):
                im = forward(x_tensor)
                diff = y - im
                params = exomild.fit_params(diff, lbda)
                log_likelihood = exomild.get_log_likelihood(diff, lbda, params)
                phi = torch.stack([torch.sum(-ll) for ll in log_likelihood]).mean()
                reg_sparse = mu_sparse * l2_l1_sparse_2d(x_tensor)
                reg_smooth = mu_smooth * l2_l1_edge_preserving_2d(x_tensor) 
                loss = phi + reg_sparse + reg_smooth
            (grad_x,) = torch.autograd.grad(loss, x_tensor, retain_graph=False, create_graph=False, allow_unused=False, )
            fx = loss.detach().cpu().numpy().astype(np.float32)
            gx = grad_x.detach().cpu().numpy().astype(np.float32)
            del loss, grad_x, im, diff, log_likelihood, params, x_tensor
            torch.cuda.empty_cache()
            return fx, gx
        xdisc_opt, fx, gx, status = optm.vmlmb(fg, x_disc_0, verb=1, lower=0, maxiter=100000, observer=None)
        return xdisc_opt, fx, gx, status

    # ------------------------------------------------------------
    # 6. Grid search over regularization parameters
    # ------------------------------------------------------------

    s1 = 3; s2 = 3 # number of values for mu_smooth and mu_sparse
    n_smooth = 5; n_sparse = 5 # starting exponents for mu_smooth and mu_sparse
    MSE = np.zeros((s1,s2)); SURE = np.zeros((s1,s2)); DA = np.zeros((s1,s2)); DE = np.zeros((s1,s2))
    x_disk_store = np.empty((s1,s2), dtype=object)
    exomild = ExoMILD(**cfg_model).to(device)
    exomild.load_state_dict(model_state)
    for k in range(s1):
        for j in range(s2):
            print(f"\nIterations {k+1,j+1} / {s1,s2} with mu_smooth=1e{(k+n_smooth)} and mu_sparse=1e{(j+n_sparse)}")
            xdisc_0 = np.zeros((C,H,W))
            mu_smooth = 10.0**(k+n_smooth); mu_sparse = 10.0**(j+n_sparse)
            (xdisc_opt, fx, gx, status) = run_bfgs(xdisc_0, y, mu_sparse, mu_smooth, exomild)

            ## Compute MC-SURE and MSE
            x_tensor_opt = torch.tensor(xdisc_opt, dtype=torch.float32, device=device)
            sure_total, data_term, div_est = compute_mc_sure(y, x_tensor_opt, xdisc_0, mu_sparse, mu_smooth, exomild)
            mse_total = mahalanobis_mse(y, x_gt, x_tensor_opt)

            # Store results
            MSE[k,j] = mse_total
            SURE[k,j] = sure_total
            DA[k,j] = data_term
            DE[k,j] = div_est
            x_disk_store[k, j] = xdisc_opt

            torch.cuda.empty_cache()

    # Find best parameters
    idx_best = np.unravel_index(np.argmin(MSE), MSE.shape)
    k_best, j_best = idx_best
    best_x_disk_mse = x_disk_store[k_best, j_best]
    idx_best = np.unravel_index(np.argmin(SURE), SURE.shape)
    k_best, j_best = idx_best
    best_x_disk_sure = x_disk_store[k_best, j_best]

    # ------------------------------------------------------------
    # 7. Save results and visualize
    # ------------------------------------------------------------

    y_numpy = y.detach().cpu().numpy()
    x_gt_numpy = x_gt.detach().cpu().numpy()
    np.savez(ROOT / "results/results_hierarchic_ellipse1em5_grid_sure_pretrained.npz", x=x_disk_store, x_gt=x_gt_numpy, mse=MSE, sure=SURE, n_smooth=n_smooth, n_sparse=n_sparse, data_term=DA, div_est=DE)

    plt.figure(1)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(best_x_disk_sure.mean(axis=0, keepdims=True))); plt.title(r"$\mathbf{x}_{\rm SURE}$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(np.squeeze(best_x_disk_mse.mean(axis=0, keepdims=True))); plt.title(r"$\mathbf{x}_{\rm MSE}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(SURE, cmap="viridis");  plt.title(r"$\mathrm{SURE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.subplot(2,2,4); plt.imshow(MSE, cmap="viridis");  plt.title(r"$\mathrm{MSE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.subplot(1,2,1); plt.imshow(DA, cmap="viridis");  plt.title(r"$\mathrm{Data Term}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.subplot(1,2,2); plt.imshow(DE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{Divergence Estimate}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()