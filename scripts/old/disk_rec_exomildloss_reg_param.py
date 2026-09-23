import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import hydra
import numpy as np
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import time
t = time.time()

from inference.inference import load_folder

from models.exomild.exomild import ExoMILD
from utils.viz import cube_3d_viewer
from disk.debris_disk import Disk
from utils.rotation import BatchRotationOperator
from collections import defaultdict
from torch.cuda.amp import autocast

from astropy.io import fits
from scipy.ndimage import zoom

from matplotlib.colors import LogNorm


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Exomild configuration
    cfg_model = cfg.model
    cfg_model.repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0] # first scale is coarse, last scale is fine
    # cfg_model.repeats = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None
    exomild = ExoMILD(**cfg_model).to(device)

    ## Load data and preprocess    
    # path_folder = ("/scratch/vasher/tbodrito/exo/data/real_data/HIP_60074/2015-04-08")

    # path_folder = ("/scratch/vasher/tbodrito/exo/data/real_data/HR_4796/2015-02-03")
    # path_folder = ("/scratch2/clear/oflasseu/DISKS_IRDIS_CHARLES/PDS_70/2018-02-24/IRDIS/data")
    # path_frame = "/scratch2/clear/oflasseu/DISKS_IRDIS_CHARLES/PDS_70/2018-02-24/IRDIS/frame_selection_vector"
    # path_folder = ("/scratch2/clear/oflasseu/DISKS_IRDIS_CHARLES") # real data folder
    
    # Load only selected frames
    path_folder = ("/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/DataAssessment/HIP_72192/2015-06-11/IRDIS/data")
    path_frame = "/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/DataAssessment/HIP_72192/2015-06-11/IRDIS/frame_selection_vector/"
    hdul = fits.open(path_frame+"ird_sortframes_vector_dc-IRD_FRAME_SELECTION_VECTOR-frame_selection_vector.fits")
    frames = hdul[0].data

    # Indices of frames to keep
    frames = np.asarray(frames).reshape(-1)
    idx = np.where(frames == 1)[0]

    # Load data
    inputs = load_folder(path_folder=path_folder, use_centered=False, channel_sortframes=0, channel_idx=None,)
    y = inputs["y"].astype(np.float32)[:,idx,:,:] # (C, T, H, W) # y = y[:,idx,:,:]
    C, T, H, W = y.shape
    lbda = inputs["lbdas"].astype(np.float32)
    print(f"{y.shape=}")
    rot = inputs["rot"].astype(np.float32)[idx] # rot = rot[idx,:]
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

    # Initialize rotation operator
    batch_rotation = BatchRotationOperator(device=device, in_size=H, out_size=H, mode="bicubic", zero_init=True,)

    ## Forward model
    def forward(x_disc):
        im = x_disc.view(1, 1, H, W).expand(-1, T, -1, -1)  # (b, T, H, W)
        im = batch_rotation.forward(x=im, rot=rot)  # (b, T, H, W)
        im = im.view(T, 1, H, W)    # (b * T, 1, H, W)
        im = im.expand(-1, C, -1, -1)   # (b * T, C, H, W)
        im = F.conv2d(im, weight=psf_crop, padding="same", groups=2)    # (b * T, C, H, W)
        im = im.view(1, T, C, H, W) # (1, b * T, C, H, W)
        im = im.permute(0, 2, 1, 3, 4)  # (1, C, b * T, H, W)
        return im

    ## Regularization term
    def l2_l1_edge_preserving_2d(x, epsilon=1e-7):
        dx = x[:-1, 1:] - x[:-1, :-1]   # (H-1, W-1)
        dy = x[1:, :-1] - x[:-1, :-1]   # (H-1, W-1)
        return torch.sum(torch.sqrt(dx**2 + dy**2 + epsilon))
    
    def l2_l1_sparse_2d(x, epsilon=1e-7):
        # return torch.sum(torch.sqrt(x**2 + epsilon))
        return torch.sum(x)

    ## Load disk
    path_disk = "/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/DataAssessment/medium_ellipse/"
    hdul = fits.open(path_disk+"hid_fake_disk_image_medium_ellipse_0degrees.fits")
    data = hdul[0].data

    #data = fits.getdata("/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/hid_fake_disk_image_ellipse_180degrees.fits")
    # Ensure the byte order is native
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128]
    with torch.no_grad():
        x_gt = 1e-5*torch.tensor(datadisk, dtype=torch.float32, device=device) 
        y = forward(x_gt) + y

    ## Grid search over regularization parameters
    s1 = 1; s2 = 1 # number of values for mu_smooth and mu_sparse
    nk = 0; nj = 1  # starting exponents for mu_smooth and mu_sparse
    RMSE = np.zeros((s1,s2))
    n_iter = 100
    x_disk_store = torch.zeros((s1,s2))
    for k in range(s1):
        for j in range(s2):
            exomild = ExoMILD(**cfg_model).to(device)
            # x_disc = torch.ones(H,W, dtype=torch.float32, requires_grad=True, device=device)
            # x_disc = torch.empty(H, W, dtype=torch.float32, device=device).uniform_(1e-7, 1e-4)
            x_disc = torch.zeros(H, W, dtype=torch.float32, device=device).detach()
            # min_val, max_val = 1e-2, 0.1 # valeurs positives petites
            # x_disc = torch.rand(H, W, dtype=torch.float32, device=device) * (max_val - min_val) + min_val
            x_disc.requires_grad_(True)
            optimizer = optim.LBFGS([x_disc], lr=0.01, max_iter=20, line_search_fn="strong_wolfe")
            # optimizer = optim.AdamW([x_disc], lr=.01)

            # Optimization scheme
            print(f"\nIterations {k+1,j+1} / {s1,s2} with mu_smooth=1e{(k+nk)} and mu_sparse=1e{(j+nj)}")
            # for i in range(n_iter):
            #     optimizer.zero_grad()
            #     im = forward(x_disc)
            #     diff = y - im
            #     with torch.no_grad():
            #         exomild.fit_params(diff, lbda)
            #     log_likelihood = exomild.get_log_likelihood(diff, lbda)
            #     phi = torch.stack([torch.mean(-ll) for ll in log_likelihood]).sum()  # loss = torch.mean(diff**2)
            #     mu_smooth = 10**3
            #     mu_sparse = 10**0
            #     epsilon = 1e-7
            #     loss = phi/(T*1) + mu_sparse*x_disc.flatten().sum() + mu_smooth*l2_l1_edge_preserving_2d(x_disc, epsilon)
            #     loss.backward()
            #     optimizer.step()
            #     with torch.no_grad():
            #         x_disc.clamp_(min=1e-16)
            #     loss_iter[i] = loss.item()
            #     print(f"Iteration {i}: loss={loss.item()}")
\
            for i in range(n_iter):
                def closure():
                    im = forward(x_disc)
                    diff = y - im
                    mu_smooth = 10.0**(k+nk); mu_sparse = 10.0**(j+nj); epsilon = 1e-7
                    exomild.fit_params(diff, lbda)
                    log_likelihood = exomild.get_log_likelihood(diff, lbda)
                    phi = torch.stack([torch.mean(-ll) for ll in log_likelihood]).sum()  # loss = torch.mean(diff**2)
                    #loss = phi/(T*1) + mu_sparse*x_disc.flatten().sum() + mu_smooth*l2_l1_edge_preserving_2d(x_disc, epsilon)
                    loss = phi/(T*1) + mu_sparse*l2_l1_sparse_2d(x_disc, epsilon) + mu_smooth*l2_l1_edge_preserving_2d(x_disc, epsilon)
                    # with torch.no_grad():
                    #     print("phi:", phi.item(), "reg_sparse:", mu_sparse*l2_l1_sparse_2d(x_disc, epsilon).item(), "reg_smooth:", mu_smooth*l2_l1_edge_preserving_2d(x_disc, epsilon).item())
                    loss.backward()
                    return loss

                optimizer.zero_grad()
                optimizer.step(closure)
                with torch.no_grad():
                    x_disc.clamp_(min=1e-16)
                grad_norm = torch.nn.utils.clip_grad_norm_([x_disc], max_norm=1.0)
                if i % 10 == 0:
                    print(f"Iteration {i}: loss={closure().item()}")
                    print(f"Gradient norm: {grad_norm:.6f}")
            
            print(f"{closure().item()=}")
            diffgt = x_disc-x_gt
            RMSE[k,j] = torch.sqrt(torch.mean(diffgt.flatten()**2))
            x_disk_store[k, j] = x_disc.detach().cpu().numpy()

    print(RMSE) 
    
    ## Find best parameters
    idx = np.unravel_index(np.argmin(RMSE), RMSE.shape)
    k_best, j_best = idx
    best_x_disk = x_disk_store[k_best, j_best]
    best_mse = RMSE[k_best, j_best]

    ## Visualization
    # Plot grid search results
    plt.figure(1)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(best_x_disk); plt.title(r"$\mathbf{x}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(np.squeeze(x_gt.detach().cpu().numpy())); plt.title(r"$\mathbf{x_{\rm GT}}$"); plt.colorbar()
    plt.subplot(2,2,4); # plt.imshow(MSE); 
    plt.imshow(RMSE, cmap="viridis", norm=LogNorm()); # plt.colorbar(label="Valeur (échelle log)")
    plt.title(r"$\mathrm{RMSE}$"); plt.colorbar()
    exponents = np.arange(s1) + nk; tick_values = np.arange(s1); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + nj; tick_values = np.arange(s2); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.savefig("/scratch2/clear/chalucas/codes/DiscRec/figures/ellipse_0degrees-gridsearch.jpeg", dpi=300)
    plt.show()

    print(best_mse)

    # Plot loss vs iterations
    plt.figure(2)
    iterations = np.arange(len(loss_iter))
    plt.plot(iterations, loss_iter, marker='o', linestyle='-', color='blue', label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss progression over iterations')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
