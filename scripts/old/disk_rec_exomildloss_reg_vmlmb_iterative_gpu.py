import gc
import sys, pathlib, os
import argparse

# Parse a few command-line options before Hydra consumes the rest.
# This allows launching multiple processes (one per GPU) with different parameters.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mu-smooth", type=float, default=None, help="Regularization weight smoothness (overrides grid search)")
parser.add_argument("--mu-sparse", type=float, default=None, help="Regularization weight sparsity (overrides grid search)")
parser.add_argument("--out", type=str, default=None, help="Output NPZ path (overrides default naming)")
args, remaining = parser.parse_known_args()
# Remove parsed args so Hydra doesn't complain.
sys.argv = [sys.argv[0]] + remaining

MU_SMOOTH = args.mu_smooth
MU_SPARSE = args.mu_sparse
OUTPUT_PATH = args.out

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]

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
from disk.debris_disk import Disk
from utils.rotation import BatchRotationOperator
from collections import defaultdict
from astropy.io import fits

from DDiT import Disk
from hydra.utils import to_absolute_path
from models.exomild.spatial_term import SpatialTerm

from torch.autograd import gradcheck


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Exomild configuration
    cfg_model = cfg.model
    cfg_model.repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # cfg_model.repeats = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None

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
    
    def l2_l1_sparse_2d(x):
        return torch.sum(x)

    ## Load disk
    path_disk = ROOT / "SyntheticDisks/DataAssessment/medium_ellipse/"
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

    # If the user specified parameters on the command line, run only that pair.
    if MU_SMOOTH is not None and MU_SPARSE is not None:
        mu_smooth = MU_SMOOTH
        mu_sparse = MU_SPARSE
        print(f"Running single combination: mu_smooth={mu_smooth}, mu_sparse={mu_sparse}")
        
    n_iter = 20000
    xdisc_iter = np.zeros((C,H,W))
    xdisc_current = np.ones((C,H,W))
    exomild = ExoMILD(**cfg_model).to(device)

    loss_values = []

    iter = 0
    while np.linalg.norm(xdisc_current-xdisc_iter) > 1e-6*np.linalg.norm(xdisc_iter):
        iter = iter + 1 
        xdisc_current = xdisc_iter
        x_tensor = torch.tensor(xdisc_iter, dtype=torch.float32, device=device, requires_grad=True)
        im = forward(x_tensor)
        diff = y - im
        with torch.no_grad():
            params = exomild.fit_params(diff, lbda)

        # BFGS optimization scheme
        def fg(x_disc):
            x_tensor = torch.tensor(x_disc, dtype=torch.float32, device=device, requires_grad=True)
            im = forward(x_tensor)
            diff = y - im
            log_likelihood = exomild.get_log_likelihood(diff, lbda, params)
            # log_likelihood = exomild.get_log_likelihood(diff, lbda)
            phi = torch.stack([torch.sum(-ll) for ll in log_likelihood]).mean()
            loss = phi + mu_sparse*l2_l1_sparse_2d(x_tensor) + mu_smooth*l2_l1_edge_preserving_2d(x_tensor)
            loss.backward()
            fx = loss.detach().cpu().numpy().astype(np.float32)
            gx = x_tensor.grad.detach().cpu().numpy().astype(np.float32)
            del loss, im, diff, log_likelihood, x_tensor
            torch.cuda.empty_cache()
            return fx, gx

        (xdisc_iter, fx, gx, status) = optm.vmlmb(fg, xdisc_iter, verb=1, lower=0, observer=None, maxiter=10000)

        loss_values.append(np.linalg.norm(xdisc_current-xdisc_iter)/np.linalg.norm(xdisc_iter)) 
        
        diffgt = xdisc_iter - x_gt_numpy
        rmse = np.sqrt(np.mean(diffgt.flatten()**2))     

        if iter > n_iter:
            break

    # ------------------------------------------------------------
    # 7. Save results and visualize
    # ------------------------------------------------------------

    y_numpy = y.detach().cpu().numpy()
    x_gt_numpy = x_gt.detach().cpu().numpy()

    # Determine output path: allow overriding via CLI, else use a deterministic file name.
    if OUTPUT_PATH is not None:
        out_path = pathlib.Path(OUTPUT_PATH)
    else:
        out_dir = ROOT / "results/iterative"
        out_dir.mkdir(exist_ok=True, parents=True)
        if MU_SMOOTH is not None and MU_SPARSE is not None:
            out_path = out_dir / f"mu_smooth{MU_SMOOTH}_mu_sparse{MU_SPARSE}.npz"

    np.savez(out_path, y=y_numpy, x=xdisc_iter, x_gt=x_gt_numpy, mse=rmse, mu_smooth=MU_SMOOTH, mu_sparse=MU_SPARSE)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()