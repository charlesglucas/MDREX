import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Configure GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from matplotlib.colors import LogNorm
import torch
import hydra
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils.optm as optm

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
    # idx = idx[:40]  # Keep only the first 40 frames for memory reasons

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

    ## Data-fidelity term
    # Configuration parameters
    cfg_model = cfg.model
    cfg_model.repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    cfg_model.use_dataparallel = True
    cfg_model.batch_size = None
    cfg_model.n_channels = 2

    # Regularization terms
    def l2_l1_edge_preserving_2d(x, epsilon=10**(-6)): 
        dx = x[:, :-1, 1:] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        dy = x[:, 1:, :-1] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        grad_sq = dx**2 + dy**2  # (C, H-1, W-1) 
        grad_sq = grad_sq.mean(dim=0)  # (H-1, W-1) 
        return torch.sum(torch.sqrt(grad_sq + epsilon**2))
    
    def l2_l1_sparse_2d(x):
        return torch.sum(torch.abs(x))
    
    # ------------------------------------------------------------
    # 3. Ground truth loading
    # ------------------------------------------------------------

    # Load disk
    flux = 1e-5
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

    def compute_mc_sure(y, x_tensor_opt, xdisc_0, mu_sparse, mu_smooth, exomild, n_mc=1):
        # Compute transformed solution
        Ax_tensor_opt = forward(x_tensor_opt)

        # Divergence term (MC)
        
        div_est = 0.0
        delta = 0.1 * (y - y.median()).abs().median()
        for _ in range(n_mc):
            v = torch.randn_like(y)  
            (xdisc_opt_eps, fx, gx, status) = run_bfgs(xdisc_0, y + delta * v, mu_sparse, mu_smooth, exomild)
            x_tensor_eps = torch.tensor(xdisc_opt_eps, dtype=torch.float32, device=device)
            div_est += torch.sum(v * (forward(x_tensor_eps) - Ax_tensor_opt)) / delta
        div_est /= n_mc

        # Data term
        diff = y - Ax_tensor_opt
        params = exomild.fit_params(diff, lbda)
        data_term = 0
        for p in range(len(params)):
            diffp = params[p]["patches"].squeeze(0) - params[p]["mean"].squeeze(0)
            Cinv = params[p]["C_inv"]               # [bsp, 1, 1, fs, fs]
            bsp, _, _, fs = diffp.shape
            diff_vec = diffp.view(bsp, C, T, fs).unsqueeze(-1) # [bsp, C, T, fs, 1]
            Cx = Cinv @ diff_vec                              # [bsp, C, T, fs, 1]
            maha = (diff_vec.transpose(-1, -2) @ Cx).squeeze(-1).squeeze(-1)
            data_term += maha.sum()
        
        # Trace term
        trace_term = y.numel()
        
        del v, x_tensor_eps
        torch.cuda.empty_cache()

        return (data_term - trace_term + 2.0 * div_est).item()
        # return (2.0 * div_est).item()
    
    def mahalanobis_mse(y, x_gt, x_tensor_opt, exomild):
        patches = exomild.extract_patches(forward(x_gt) - forward(x_tensor_opt), lbda)
        params = exomild.fit_params(y - forward(x_gt), lbda)
        mse_total = 0
        for p in range(len(params)):
            diffp = patches[p]["fx"].unsqueeze(2)
            Cinv = params[p]["C_inv"]               # [bsp, 1, 1, fs, fs]     
            bsp, _, _, fs = diffp.shape
            diff_vec = diffp.view(bsp, C, T, fs).unsqueeze(-1)  # [bsp, C, T, fs, 1]             
            Cx = Cinv @ diff_vec                              # [bsp, C, T, fs, 1]
            quad = (diff_vec.transpose(-1, -2) @ Cx).squeeze(-1).squeeze(-1)
            mse_total += quad.sum()
        return mse_total

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
        xdisc_opt, fx, gx, status = optm.vmlmb(fg, x_disc_0, verb=1, lower=0, maxiter=10000, observer=None)
        return xdisc_opt, fx, gx, status

    # ------------------------------------------------------------
    # 6. Grid search over regularization parameters
    # ------------------------------------------------------------


    s1 = 5; s2 = 5 # number of values for mu_smooth and mu_sparse
    n_smooth = 3; n_sparse = 3 # starting exponents for mu_smooth and mu_sparse
    MSE = np.zeros((s1,s2)); SURE = np.zeros((s1,s2))
    x_disk_store = np.empty((s1,s2), dtype=object)
    exomild = ExoMILD(**cfg_model).to(device)

    # xdisc_0 = np.zeros((C,H,W))
    # (x_smooth, fx, gx, status) = run_bfgs(xdisc_0, y, mu_sparse=0, mu_smooth=100, exomild=exomild)
    # x_tensor_smooth = torch.tensor(x_smooth, dtype=torch.float32, device=device)
    # noise_resid = y - forward(x_tensor_smooth)
    # params_noise = exomild.fit_params(noise_resid, lbda)

    for k in range(s1):
        for j in range(s2):
            print(f"\nIterations {k+1,j+1} / {s1,s2} with mu_smooth=1e{(k+n_smooth)} and mu_sparse=1e{(j+n_sparse)}")
            xdisc_0 = np.zeros((C,H,W))
            mu_smooth = 10.0**(k+n_smooth); mu_sparse = 10.0**(j+n_sparse)
            (xdisc_opt, fx, gx, status) = run_bfgs(xdisc_0, y, mu_sparse, mu_smooth, exomild)

            ## Compute MC-SURE and MSE
            x_tensor_opt = torch.tensor(xdisc_opt, dtype=torch.float32, device=device)
            sure_total = compute_mc_sure(y, x_tensor_opt, xdisc_0, mu_sparse, mu_smooth, exomild)
            mse_total = mahalanobis_mse(y, x_gt, x_tensor_opt, exomild)

            # Store results
            MSE[k,j] = mse_total
            SURE[k,j] = sure_total
            x_disk_store[k, j] = xdisc_opt

            del x_tensor_opt, sure_total, mse_total 
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
    np.savez(ROOT / "results/results_hierarchic_ellipse1em5_grid_sure_5x5.npz", x=x_disk_store, x_gt=x_gt_numpy, mse=MSE, sure=SURE, n_smooth=n_smooth, n_sparse=n_sparse)

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

if __name__ == "__main__":
    main()