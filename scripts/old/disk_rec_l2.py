import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import hydra
import numpy as np
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from inference.inference import load_folder

from models.exomild.exomild import ExoMILD
from utils.viz import cube_3d_viewer
from disk.debris_disk import Disk
from utils.rotation import BatchRotationOperator
from collections import defaultdict


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Exomild configuration
    cfg_model = cfg.model
    cfg_model.repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None
    exomild = ExoMILD(**cfg_model).to(device)

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

    y = inputs["y"].astype(np.float32)
    # (C, T, H, W)

    C, T, H, W = y.shape

    lbda = inputs["lbdas"].astype(np.float32)
    print(f"{y.shape=}")

    rot = inputs["rot"].astype(np.float32)
    print(f"{y.shape=}")

    psf = inputs["psf"].astype(np.float32)
    print(f"{psf.shape=}")

    y = torch.tensor(y, device=device).unsqueeze(0)
    # (b, C, T, H, W)

    lbda = torch.tensor(lbda, device=device).unsqueeze(0)
    # (b, C)

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

    psf_crop = psf_crop.view(2, 1, crop_size, crop_size)
    # (1, 1, crop, crop)
    
    device = y.device

    def forward(x_disc):
        im = x_disc.view(1, 1, H, W).expand(-1, T, -1, -1)  # (b, T, H, W)
        im = batch_rotation.forward(x=im, rot=rot)  # (b, T, H, W)
        im = im.view(T, 1, H, W)    # (b * T, 1, H, W)
        im = im.expand(-1, C, -1, -1)   # (b * T, C, H, W)
        im = F.conv2d(im, weight=psf_crop, padding="same", groups=2)    # (b * T, C, H, W)
        im = im.view(1, T, C, H, W) # (1, b * T, C, H, W)
        im = im.permute(0, 2, 1, 3, 4)  # (1, C, b * T, H, W)
        return im
    
    ## Initialize Optimization parameters
    with torch.no_grad():
        exomild.fit_params(y, lbda)

    batch_rotation = BatchRotationOperator(
        device=device,
        in_size=H,
        out_size=H,
        mode="bicubic",
        zero_init=True,
    )

    lr = 1e-10
    x_disc = torch.zeros(
        H,W,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    optimizer = optim.SGD([x_disc], lr=lr)

    ## Optimization scheme
    all_losses = []
    all_im = defaultdict(list)
    with torch.autograd.detect_anomaly(check_nan=True):
        for i in range(100):
            # for i in range(5):
            print(f"\nIteration {i}")
            def closure():
                optimizer.zero_grad()
                im = forward(x_disc)
                diff = y - im      
                loss = torch.mean(diff**2)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure()
            # print(f"losses={[l.item() for l in losses]}")
            print(f"{loss.item()=}")
            grad_norm = torch.nn.utils.clip_grad_norm_([x_disc], max_norm=1.0)
            if i % 10 == 0:
                print(f"Gradient norm: {grad_norm:.6f}")
            all_losses.append(loss.item())

    im = forward(x_disc)
    diff = y - im 

    plt.figure(1)
    plt.plot(all_losses); plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.yscale("log"); plt.show()

    plt.figure(2)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,0,:,:])); plt.title("$y$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(np.squeeze(x_disc.detach().cpu().numpy())); plt.title("$x$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(np.squeeze(im.detach().cpu().numpy()[0,0,0,:,:])); plt.title("$Ax$"); plt.colorbar()
    plt.subplot(2,2,4); plt.imshow(np.squeeze(diff.detach().cpu().numpy()[0,0,0,:,:])); plt.title("$\Vert y-Ax \Vert^2$"); plt.colorbar()
    plt.show()

    plt.figure(3)
    for j, _im in all_im.items():
        print(f"{j=}")
        _im = torch.stack(_im).cpu().numpy()
        cube_3d_viewer(_im)

if __name__ == "__main__":
    main()
