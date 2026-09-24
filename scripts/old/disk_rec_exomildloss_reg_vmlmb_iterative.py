import sys, pathlib, os

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

from matplotlib.colors import LogNorm
import torch
import hydra
import numpy as np
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils.optm as optm

from inference.inference import load_folder

from models.exomild.exomild import ExoMILD
from utils.viz import cube_3d_viewer
# from disk.debris_disk import Disk
from utils.rotation import BatchRotationOperator
from collections import defaultdict
from astropy.io import fits

from DDiT import Disk
from hydra.utils import to_absolute_path
from models.exomild.spatial_term import SpatialTerm

from torch.autograd import gradcheck


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Exomild configuration
    cfg_model = cfg.model
    cfg_model.repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # cfg_model.repeats = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None

    # Load only selected frames
    path_folder = ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data"
    path_frame = ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/frame_selection_vector/"
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

    # Load coronograph mask
    path_coronograph = ROOT / "data/coronograph"
    k1k2_path = os.path.join(path_coronograph,"sphere_irdis_k1_k2_coronagraph_transmission_map.fits")
    with fits.open(k1k2_path, memmap=False) as hdul:
            mask = np.array(hdul[0].data, dtype=np.float32)
    deltaH = (mask.shape[1] - H) // 2; deltaW = (mask.shape[2] - W) // 2; 
    mask = mask[:, deltaH:deltaH+H, deltaW:deltaW+W] # (C, H, W)
    mask = torch.tensor(mask, device=device) # (C, H, W)

    ## Forward model
    def forward(x_disc):
        # x_disc : (C, H, W)
        # Add batch dimension and time dimension
        im = x_disc.unsqueeze(0)   # (1, C, H, W)
        im = im.expand(T, -1, -1, -1)        # (T, C, H, W)
        # Merge batch and time for rotation
        im = im.permute(1, 0, 2, 3)          # (C, T, H, W)
        # Rotate each channel independently
        im = batch_rotation.forward(x=im, rot=rot.expand(C,-1))  # (C, T, H, W)
        im = im.permute(1, 0, 2, 3)   
        # Convolution: groups=C means depthwise conv
        im = F.conv2d(im, weight=psf_crop, padding="same", groups=C)  # (C, T, H, W)
        # Apply coronograph mask
        im = im.permute(1, 0, 2, 3)   
        im = im * mask.unsqueeze(1)  # mask: (1, 1, H, W)
        return im.unsqueeze(0) # (1, C, T, H, W)

    ## Regularization term
    def l2_l1_edge_preserving_2d(x, epsilon=1e-6): 
        # x : (C, H, W) 
        dx = x[:, :-1, 1:] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        dy = x[:, 1:, :-1] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        grad_sq = dx**2 + dy**2  # (C, H-1, W-1) 
        grad_sq = grad_sq.mean(dim=0)  # (H-1, W-1) 
        return torch.sum(torch.sqrt(grad_sq + epsilon**2))
    
    def l2_l1_sparse_2d(x, epsilon=1e-7):
        return torch.sum(torch.abs(x))

    ## Load disk
    path_disk = ROOT / "data/synthetic_disks/medium_ellipse/"
    hdul = fits.open(path_disk / "hid_fake_disk_image_medium_ellipse_0degrees.fits")
    data = hdul[0].data

    # Ensure the byte order is native
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128]
    with torch.no_grad():
        x_gt = 1e-5*torch.tensor(datadisk, dtype=torch.float32, device=device) 
        x_gt = x_gt.unsqueeze(0).repeat(C, 1, 1)
        speckle = y
        y = forward(x_gt) + y
    x_gt_numpy = x_gt.detach().cpu().numpy() # for testing convergence to the correct solution

    # ##  Visualization
    # # Plot grid search results
    # plt.figure(2)
    # plt.subplot(2,3,1); plt.imshow(np.squeeze(forward(x_gt).detach().cpu().numpy()[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$forward(\mathbf{x})_0$"); plt.colorbar()
    # plt.subplot(2,3,2); plt.imshow(np.squeeze(speckle.detach().cpu().numpy()[0,0,0,:,:])); plt.title(r"$\mathbf{speckle}_0$"); plt.colorbar()
    # plt.subplot(2,3,3); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,0,:,:])); plt.title(r"$y_0$"); plt.colorbar()
    # plt.subplot(2,3,4); plt.imshow(np.squeeze(forward(x_gt).detach().cpu().numpy()[0,1,0,:,:]),vmin=0,vmax=50); plt.title(r"$forward(\mathbf{x})_1$"); plt.colorbar()
    # plt.subplot(2,3,5); plt.imshow(np.squeeze(speckle.detach().cpu().numpy()[0,1,0,:,:])); plt.title(r"$\mathbf{speckle}_1$"); plt.colorbar()
    # plt.subplot(2,3,6); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,1,0,:,:])); plt.title(r"$y_1$"); plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    
    # Observer for vmlmb monitoring
    history = {'f':[], 'grad_norm':[], 'rmse':[], 'alpha':[]}
    def observer(iters, evals, rejects, t, x, f, g, pgnorm, alpha, fg):
        history['f'].append(f)
        history['grad_norm'].append(np.linalg.norm(g))
        history['alpha'].append(alpha)
        rmse = np.sqrt(np.mean((x.flatten() - x_gt_numpy.flatten())**2))
        history['rmse'].append(rmse)

        if iters % 5 == 0:  
            print(f"Iter {iters:3d}, f={f:.4e}, ||g||={np.linalg.norm(g):.2e}, RMSE={rmse:.4e}, alpha={alpha:.2e}")

    ## Grid search over regularization parameters
    s1 = 5; s2 = 5 # number of values for mu_smooth and mu_sparse
    n_smooth = 3; n_sparse = 3 # starting exponents for mu_smooth and mu_sparse
    n_iter = 20000
    RMSE = np.zeros((s1,s2)); crit = np.zeros((s1,s2))
    x_disk_store = np.empty((s1,s2), dtype=object)
    loss_values_store = np.empty((s1,s2), dtype=object)
    for k in range(s1):
        for j in range(s2):
            print(f"\nIterations {k+1,j+1} / {s1,s2} with mu_smooth=1e{(k+n_smooth)} and mu_sparse=1e{(j+n_sparse)}")
            xdisc_iter = np.zeros((C,H,W))
            xdisc_current = np.ones((C,H,W))
            exomild = ExoMILD(**cfg_model).to(device)

            # plt.ion()  # mode interactif
            # fig, ax = plt.subplots(figsize=(7,4))
            # line, = ax.plot([], [], '-b')
            # ax.set_xlabel(f"Iteration")
            # ax.set_ylabel(f"Stop criterion")
            # ax.set_title(f"Iterative algorithm")
            # ax.grid(True)

            # fig2, ax2 = plt.subplots(figsize=(5,5))
            # im2 = ax2.imshow(np.zeros((H, W)))
            # ax2.set_title("Estimate (live)")
            # plt.tight_layout()
            # plt.colorbar(im2, ax=ax2)

            # fig3, ax3 = plt.subplots(figsize=(5,5))
            # im3 = ax3.imshow(np.zeros((H, W)))
            # ax3.set_title("Estimate (live)")
            # plt.tight_layout()
            # plt.colorbar(im3, ax=ax3)

            loss_values = []

            iter = 0
            while np.linalg.norm(xdisc_current-xdisc_iter) > 1e-6*np.linalg.norm(xdisc_iter):
                iter = iter + 1 
                xdisc_current = xdisc_iter
                x_tensor = torch.tensor(xdisc_iter, dtype=torch.float32, device=device, requires_grad=True)
                im = forward(x_tensor)
                diff = y - im
                with torch.no_grad():
                    # exomild.fit_params(diff, lbda)
                    params = exomild.fit_params(diff, lbda)
                # print(diff[0,0,:,:,:].mean())
                # print(diff[0,1,:,:,:].mean())

                # BFGS optimization scheme
                def fg(x_disc):
                    x_tensor = torch.tensor(x_disc, dtype=torch.float32, device=device, requires_grad=True)
                    im = forward(x_tensor)
                    diff = y - im
                    log_likelihood = exomild.get_log_likelihood(diff, lbda, params)
                    # log_likelihood = exomild.get_log_likelihood(diff, lbda)
                    phi = torch.stack([torch.sum(-ll) for ll in log_likelihood]).mean()
                    mu_smooth = 10.0**(k+n_smooth); mu_sparse = 10.0**(j+n_sparse)
                    # mu_smooth = 0; mu_sparse = 0; epsilon = 1e-7
                    loss = phi + mu_sparse*l2_l1_sparse_2d(x_tensor) + mu_smooth*l2_l1_edge_preserving_2d(x_tensor)
                    loss.backward()
                    fx = loss.detach().cpu().numpy().astype(np.float32)
                    gx = x_tensor.grad.detach().cpu().numpy().astype(np.float32)
                    # print(x_tensor.grad[0].mean())
                    # print(x_tensor.grad[1].mean())
                    del loss, fx, gx, im, diff, log_likelihood, x_tensor
                    torch.cuda.empty_cache()
                    return fx, gx

                (xdisc_iter, fx, gx, status) = optm.vmlmb(fg, xdisc_iter, verb=1, lower=0, observer=None, maxiter=10000)

                loss_values.append(np.linalg.norm(xdisc_current-xdisc_iter)/np.linalg.norm(xdisc_iter))

                # # Mise à jour du plot
                # line.set_xdata(range(len(loss_values)))
                # line.set_ydata(loss_values)
                # ax.relim()
                # ax.autoscale_view()

                # plt.draw()
                # plt.pause(0.001)  # rafraîchissement sans bloquer
                # plt.show(block=False)     

                # x_img = np.array(xdisc_iter, dtype=float, copy=True).reshape(C, H, W)[0]
                # im2.set_clim(vmin=x_img.min(), vmax=x_img.max())
                # im2.set_data(x_img)
                # ax2.set_title(f"Channel 1 ")
                # fig2.canvas.draw_idle()
                # plt.pause(0.001)   # rafraîchissement sans bloquer
                # plt.show(block=False)  

                # x_img = np.array(xdisc_iter, dtype=float, copy=True).reshape(C, H, W)[1]
                # im3.set_clim(vmin=x_img.min(), vmax=x_img.max())
                # im3.set_data(x_img)
                # ax3.set_title(f"Channel 2")
                # fig2.canvas.draw_idle()
                # plt.pause(0.001)   # rafraîchissement sans bloquer
                # plt.show(block=False)          

                if iter > n_iter:
                    break

            # Compute RMSE
            diffgt = xdisc_iter - x_gt_numpy
            RMSE[k,j] = np.sqrt(np.mean(diffgt.flatten()**2))
            crit[k,j] = np.sqrt(np.mean(fx.flatten()**2))
            x_disk_store[k,j] = xdisc_iter

            loss_values_store[k,j] = loss_values

    # plt.ioff()  

    np.savez(ROOT / "results/results_iterative_ellipse1em5_grid_iter10000.npz", x=x_disk_store, x_gt=x_gt_numpy, rmse=RMSE, n_smooth=n_smooth, n_sparse=n_sparse, loss_values_store=loss_values_store)

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