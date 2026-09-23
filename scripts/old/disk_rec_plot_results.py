import sys, pathlib, os
# Configure GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]

from matplotlib.colors import LogNorm
import torch
import hydra
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from inference.inference import load_folder
from utils.rotation import BatchRotationOperator
from astropy.io import fits
from hydra.utils import to_absolute_path

from torch.autograd import gradcheck


@hydra.main(config_path="../conf", config_name="config")
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
    path_coronograph = ROOT / "/scratch2/clear/chalucas/codes/DiscRec/data/coronograph"
    k1k2_path = os.path.join(path_coronograph,"sphere_irdis_k1_k2_coronagraph_transmission_map.fits")
    with fits.open(k1k2_path, memmap=False) as hdul:
            mask = np.array(hdul[0].data, dtype=np.float32)
    deltaH = (mask.shape[1] - H) // 2; deltaW = (mask.shape[2] - W) // 2; 
    mask = mask[:, deltaH:deltaH+H, deltaW:deltaW+W] # (C, H, W)
    mask = torch.tensor(mask, device=device) # (C, H, W)

    ## Forward model
    # def forward(x_disc):
    #     # Rotate disk, convolve with psf and apply coronograph mask
    #     im = x_disc.view(1, 1, H, W).expand(-1, T, -1, -1)  # (b, T, H, W)
    #     im = batch_rotation.forward(x=im, rot=rot)  # (b, T, H, W)
    #     im = im.view(T, 1, H, W)    # (b * T, 1, H, W)
    #     im = im.expand(-1, C, -1, -1)   # (b * T, C, H, W)
    #     im = F.conv2d(im, weight=psf_crop, padding="same", groups=C)    # (b * T, C, H, W)
    #     im = im.view(1, T, C, H, W) # (1, b * T, C, H, W)
    #     im = im.permute(0, 2, 1, 3, 4)  # (1, C, b * T, H, W)
    #     im = im * mask.unsqueeze(0) # (1, C, b * T, H, W)
    #     return im
    
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

    ## Load disk
    path_disk = "/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/DataAssessment/medium_ellipse/"
    hdul = fits.open(path_disk+"hid_fake_disk_image_medium_ellipse_0degrees.fits")
    data = hdul[0].data

    # Ensure the byte order is native
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128]
    with torch.no_grad():
        x_gt = 1e-5*torch.tensor(datadisk, dtype=torch.float32, device=device) 
        x_gt = x_gt.unsqueeze(0).repeat(C, 1, 1)
        y = forward(x_gt) + y
    x_gt_numpy = x_gt.detach().cpu().numpy() # for testing convergence to the correct solution

    # data = np.load(ROOT / "results/results_hierarchic_ellipse1em5_grid2.npz", allow_pickle=True) 
    # data = np.load(ROOT / "results/results_hierarchic_ellipse1em5_grid_pretrained_40frames.npz", allow_pickle=True) 
    data = np.load(ROOT / "results/results_hierarchic_ellipse1em5_grid_40frames.npz", allow_pickle=True) 
    # data = np.load(ROOT / "results/results_iterative_ellipse1em5_grid_iter5000.npz", allow_pickle=True) 
    x_disk_store = data["x"] 
    x_gt= data["x_gt"]
    RMSE = data["rmse"]
    n_sparse = data["n_sparse"]
    n_smooth = data["n_smooth"]

    ## Find best parameters
    idx_best = np.unravel_index(np.argmin(RMSE), RMSE.shape)
    k_best, j_best = idx_best
    best_x_disk = x_disk_store[k_best, j_best]
    best_mse = RMSE[k_best, j_best]
    s1, s2 = RMSE.shape

    ##  Visualization
    # Plot grid search results
    plt.figure(1)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(np.squeeze(best_x_disk[0,:,:])); plt.title(r"$\mathbf{x}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(np.squeeze(x_gt[0,:,:])); plt.title(r"$\mathbf{x_{\rm GT}}$"); plt.colorbar()
    plt.subplot(2,2,4); plt.imshow(RMSE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{RMSE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()