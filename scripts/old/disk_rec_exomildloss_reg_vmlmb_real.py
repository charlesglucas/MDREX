import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

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


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Exomild configuration
    cfg_model = cfg.model
    cfg_model.repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # cfg_model.repeats = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None

    ## load data
    path_folder = (
        "/scratch/vasher/tbodrito/exo/data/real_data/HR_4796/2015-02-03"
    )

    inputs = load_folder(
        path_folder=path_folder,
        use_centered=False,
        channel_sortframes=0,
        channel_idx=None,
    )

    y = inputs["y"].astype(np.float32) # (C, T, H, W)
    C, T, H, W = y.shape
    lbda = inputs["lbdas"].astype(np.float32)
    print(f"{y.shape=}")
    rot = inputs["rot"].astype(np.float32)
    print(f"{y.shape=}")
    psf = inputs["psf"].astype(np.float32)
    print(f"{psf.shape=}")

    y = torch.tensor(y, device=device).unsqueeze(0) # (b, C, T, H, W)
    lbda = torch.tensor(lbda, device=device).unsqueeze(0) # (b, C)
    rot = torch.tensor(rot, device=device).unsqueeze(0)

    psf = torch.tensor(psf, device=device)
    print(f"{y.shape=}")
    print(f"{lbda.shape=}")
    print(f"{rot.shape=}")
    print(f"{psf.shape=}")

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
        return torch.sum(torch.sqrt(x**2 + epsilon))
        # return torch.sum(x)

    # ## Load disk
    # path_disk = "/scratch2/clear/chalucas/codes/DiscRec/SyntheticDisks/DataAssessment/medium_ellipse/"
    # hdul = fits.open(path_disk+"hid_fake_disk_image_medium_ellipse_0degrees.fits")
    # data = hdul[0].data

    # disk = Disk()
    # disk.compute_model(a = 1, incl = 0, e = 0.2, omega = 0, pa = 90, pin = 10, pout = -10, gsca = 0.2, gpol = 0, opang = 0.2)
    # im = disk.intensity
    # x_min = im.min(); x_max = im.max(); data = (im - x_min) / (x_max - x_min)

    # # Ensure the byte order is native
    # if data.dtype.byteorder not in ('=', '|'):
    #     data = data.byteswap().view(data.dtype.newbyteorder('='))
    # datadisk = data[513-128:513+128, 513-128:513+128]
    # with torch.no_grad():
    #     x_gt = 1e-5*torch.tensor(datadisk, dtype=torch.float32, device=device) 
    #     y = forward(x_gt) + y

    # x0_disc = np.ones((H,W)) * 1e-5
    # x0_disc = np.zeros((H,W))
    # x0_disc = np.abs(np.random.randn(H,W)) * 1e-5
    # x_gt = x_gt.detach().cpu().numpy()
    # x0_disc = x_gt

    ## Regularization scheme
    x0_disc = np.ones((H,W)) * 1e-5 
    exomild = ExoMILD(**cfg_model).to(device)
    mu_smooth = 10.0**2; mu_sparse = 10.0**3; epsilon = 1e-7

    # BFGS optimization scheme
    def fg(x_disc):
        x_tensor = torch.tensor(x_disc, dtype=torch.float32, device=device, requires_grad=True)
        im = forward(x_tensor)
        diff = y - im
        diff_ng = diff.detach().clone()
        with torch.no_grad():
            exomild.fit_params(diff_ng, lbda)
        log_likelihood = exomild.get_log_likelihood(diff, lbda)
        phi = torch.stack([torch.mean(-ll) for ll in log_likelihood]).mean()
        loss = phi/(T*1) + mu_sparse*l2_l1_sparse_2d(x_tensor, epsilon) + mu_smooth*l2_l1_edge_preserving_2d(x_tensor, epsilon)
        loss.backward()
        fx = loss.detach().cpu().numpy()
        gx = x_tensor.grad.detach().cpu().numpy()
        return fx, gx

    (x_disc, fx, gx, status) = optm.vmlmb(fg, x0_disc, verb=1, lower=0, ftol=1e-8, gtol=1e-8, epsilon=1e-8, maxiter=100)

    x_tensor = torch.tensor(x_disc, dtype=torch.float32, device=device)
    diff = y - forward(x_tensor)

    ##  Visualization
    # Plot grid search results
    plt.figure(1)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(x_disc); plt.title(r"$\mathbf{x}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(forward(x_tensor).detach().cpu().numpy()[0,0,0,:,:]); plt.title(r"$\mathbf{x_{\rm GT}}$"); plt.colorbar()
    plt.subplot(2,2,4); plt.imshow(diff.detach().cpu().numpy()[0,0,0,:,:]);  plt.title(r"$\mathbf{x - x_{\rm GT}}$"); plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
