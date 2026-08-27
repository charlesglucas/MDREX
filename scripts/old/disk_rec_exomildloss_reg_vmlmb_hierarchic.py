import sys, pathlib, os

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from matplotlib.colors import LogNorm
import torch
import hydra
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils.optm as optm

from inference.inference import load_folder

from models.exomild.exomild import ExoMILD
from utils.rotation import BatchRotationOperator
from astropy.io import fits
from hydra.utils import to_absolute_path


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # 1. Real-world data loading and preprocessing
    # ------------------------------------------------------------

    # Load only selected frames
    path_folder = ("/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/DataAssessment/HIP_72192/2015-06-11/IRDIS/data")
    path_frame = "/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/DataAssessment/HIP_72192/2015-06-11/IRDIS/frame_selection_vector/"
    hdul = fits.open(path_frame+"ird_sortframes_vector_dc-IRD_FRAME_SELECTION_VECTOR-frame_selection_vector.fits")
    frames = hdul[0].data

    # Indices of frames to keep
    frames = np.asarray(frames).reshape(-1)
    idx = np.where(frames == 1)[0]
    idx = idx[:40] # keep only 40 frames for faster experiments

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
    path_coronograph = to_absolute_path("/scratch2/clear/chalucas/codes/DiscRec/data/coronograph")
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

    ## Forward model
    def forward(x_disc):
        im = x_disc.unsqueeze(0)   # (1, C, H, W)
        im = im.expand(T, -1, -1, -1)        # (T, C, H, W)
        im = im.permute(1, 0, 2, 3)          # (C, T, H, W)
        im = batch_rotation.forward(x=im, rot=rot.expand(C,-1))  # (C, T, H, W) .expand(C,-1)
        im = im.permute(1, 0, 2, 3)   
        im = F.conv2d(im, weight=psf_crop, padding="same", groups=C)  # (T, C, H, W)
        im = im.permute(1, 0, 2, 3) 
        im = im * mask.unsqueeze(1)  # mask: (C, 1, H, W)
        return im.unsqueeze(0) # (1, C, T, H, W)

    ## Data-fidelity term
    cfg_model = cfg.model
    # cfg_model.repeats = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0] 
    cfg_model.repeats = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # cfg_model.repeats = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # cfg_model.repeats = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None

    ## Regularization term
    def l2_l1_edge_preserving_2d(x, epsilon=10**(-6)): 
        # x : (C, H, W) 
        dx = x[:, :-1, 1:] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        dy = x[:, 1:, :-1] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        grad_sq = dx**2 + dy**2  # (C, H-1, W-1) 
        grad_sq = grad_sq.mean(dim=0)  # (H-1, W-1) 
        return torch.sum(torch.sqrt(grad_sq + epsilon**2))
    
    def l2_l1_sparse_2d(x):
        return torch.sum(x)
    
    # ------------------------------------------------------------
    # 3. Ground truth loading
    # ------------------------------------------------------------

    ## Load disk
    flux = 1e-5
    path_disk = "/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/DataAssessment/medium_ellipse/"
    hdul = fits.open(path_disk+"hid_fake_disk_image_medium_ellipse_0degrees.fits")
    data = hdul[0].data

    # Ensure the byte order is native
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128]
    with torch.no_grad():
        x_gt = flux*torch.tensor(datadisk, dtype=torch.float32, device=device) 
        x_gt = x_gt.unsqueeze(0).repeat(C, 1, 1)
        speckle = y
        y = forward(x_gt) + y 
    x_gt_numpy = x_gt.detach().cpu().numpy()

    # ------------------------------------------------------------
    # 4. Monitoring
    # ------------------------------------------------------------

    # Observer for vmlmb monitoring
    plt.ion()  # mode interactif
    fig, ax = plt.subplots(figsize=(7,4))
    line, = ax.plot([], [], '-b')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Hierarchic algorithm")
    ax.grid(True)

    fig2, ax2 = plt.subplots(figsize=(5,5))
    im2 = ax2.imshow(np.zeros((H, W)))
    ax2.set_title("Estimate (live)")
    plt.tight_layout()
    plt.colorbar(im2, ax=ax2)

    fig3, ax3 = plt.subplots(figsize=(5,5))
    im3 = ax3.imshow(np.zeros((H, W)))
    ax3.set_title("Estimate (live)")
    plt.tight_layout()
    plt.colorbar(im3, ax=ax3)

    loss_values = []

    history = {'f':[], 'grad_norm':[], 'rmse':[], 'alpha':[]}
    def observer(iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg):
        history['f'].append(f)
        history['grad_norm'].append(np.linalg.norm(g))
        history['alpha'].append(alpha)
        rmse = np.sqrt(np.mean((x.flatten() - x_gt_numpy.flatten())**2))
        history['rmse'].append(rmse)

        loss_values.append(f)

        # Loss plot
        line.set_xdata(range(len(loss_values)))
        line.set_ydata(loss_values)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)  # rafraîchissement sans bloquer
        plt.show(block=False)   

        # Disk plot
        if iters % 1 == 0:  
            x_img = np.array(x, dtype=float, copy=True).reshape(C, H, W)[0]
            im2.set_clim(vmin=x_img.min(), vmax=x_img.max())
            im2.set_data(x_img)
            ax2.set_title(f"Canal 1 - Iter {iters} — f={f:.3e}")
            fig2.canvas.draw_idle()
            plt.pause(0.001)   # rafraîchissement sans bloquer
            plt.show(block=False)  

            x_img = np.array(x, dtype=float, copy=True).reshape(C, H, W)[1]
            im3.set_clim(vmin=x_img.min(), vmax=x_img.max())
            im3.set_data(x_img)
            ax3.set_title(f"Canal 2 - Iter {iters} — f={f:.3e}")
            fig2.canvas.draw_idle()
            plt.pause(0.001)   # rafraîchissement sans bloquer
            plt.show(block=False)  

        if iters % 5 == 0:  
            print(f"Iter {iters:3d}, f={f:.4e}, ||g||={np.linalg.norm(g):.2e}, RMSE={rmse:.4e}, alpha={alpha:.2e}")

    def check_gradient(x0, eps=1e-4, n_tests=10):
        """
        Compare autograd gradient vs finite-difference gradient
        """
        x0 = x0.astype(np.float64)
        fx, gx = fg(x0.astype(np.float32))  # gradient autograd en float32
        gx = gx.ravel()
        print("Autograd gradient norm:", np.linalg.norm(gx))
        N = gx.size
        idxs = np.random.choice(N, size=n_tests, replace=False)
        for i in idxs:
            x_plus = x0.copy().ravel()
            x_minus = x0.copy().ravel()
            x_plus[i] += eps
            x_minus[i] -= eps
            # Pour la FD, on garde float64
            f_plus, _ = fg(x_plus.reshape(x0.shape).astype(np.float32))
            f_minus, _ = fg(x_minus.reshape(x0.shape).astype(np.float32))
            g_fd = (f_plus - f_minus) / (2 * eps)
            g_auto = gx[i]
            print(f"i={i:6d} | autograd={g_auto:+.6e} | FD={g_fd:+.6e} | ratio={g_auto/g_fd if g_fd!=0 else np.nan}")

    # ------------------------------------------------------------
    # 5. Grid search over regularization parameters
    # ------------------------------------------------------------

    s1 = 1; s2 = 1 # number of values for mu_smooth and mu_sparse
    n_smooth = 4; n_sparse = 4 # starting exponents for mu_smooth and mu_sparse
    RMSE = np.zeros((s1,s2)); crit = np.zeros((s1,s2))
    x_disk_store = np.empty((s1,s2), dtype=object)    
    for k in range(s1):
        for j in range(s2):
            exomild = ExoMILD(**cfg_model).to(device)
            print(f"\nIterations {k+1,j+1} / {s1,s2} with mu_smooth=1e{(k+n_smooth)} and mu_sparse=1e{(j+n_sparse)}")
            xdisc_iter = np.zeros((C,H,W))

            # BFGS optimization scheme
            def fg(x_disc):
                x_tensor = torch.tensor(x_disc, dtype=torch.float32, device=device, requires_grad=True)
                im = forward(x_tensor)
                diff = y - im
                params = exomild.fit_params(diff, lbda) 
                log_likelihood = exomild.get_log_likelihood(diff, lbda, params)
                phi = torch.stack([torch.sum(-ll) for ll in log_likelihood]).mean()
                mu_smooth = 10.0**(k+n_smooth); mu_sparse = 10.0**(j+n_sparse)
                loss = phi + mu_sparse*l2_l1_sparse_2d(x_tensor) + mu_smooth*l2_l1_edge_preserving_2d(x_tensor)
                loss.backward() 
                fx = loss.detach().cpu().numpy().astype(np.float32)
                gx = x_tensor.grad.detach().cpu().numpy().astype(np.float32)
                return fx, gx
            # check_gradient(xdisc_iter, eps=1e-5, n_tests=10)
            (xdisc_iter, fx, gx, status) = optm.vmlmb(fg, xdisc_iter, verb=1, lower=0, maxiter=3000, observer=None)

            # Compute RMSE
            diffgt = xdisc_iter - x_gt_numpy
            RMSE[k,j] = np.sqrt(np.mean(diffgt.flatten()**2))
            # crit[k,j] = np.sqrt(np.mean(fx.flatten()**2))
            x_disk_store[k, j] = xdisc_iter

            # Clean GPU memory
            del exomild
            torch.cuda.empty_cache()

    plt.ioff


    # ------------------------------------------------------------
    # 6. Save results and visualize
    # ------------------------------------------------------------

    y_numpy = y.detach().cpu().numpy()
    np.savez("/scratch2/clear/chalucas/codes/DiscRecCopy/results/results_hierarchic_ellipse1em5_grid_40frames.npz", x=x_disk_store, x_gt=x_gt_numpy, rmse=RMSE, n_smooth=n_smooth, n_sparse=n_sparse)

    ## Find best parameters
    idx_best = np.unravel_index(np.argmin(RMSE), RMSE.shape)
    k_best, j_best = idx_best
    best_x_disk = x_disk_store[k_best, j_best]
    best_mse = RMSE[k_best, j_best]

    ##  Visualization
    # Plot grid search results
    plt.figure(2)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(np.squeeze(best_x_disk.mean(axis=0, keepdims=True))); plt.title(r"$\mathbf{x}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(np.squeeze(x_gt_numpy[0,:,:])); plt.title(r"$\mathbf{x_{\rm GT}}$"); plt.colorbar()
    plt.subplot(2,2,4); plt.imshow(RMSE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{RMSE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
8