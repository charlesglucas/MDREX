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

    cfg_model = cfg.model
    cfg_model.repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None
    exomild = ExoMILD(**cfg_model).to(device)

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

    disk = Disk(z_max=0.2, n_layers=1)

    device = y.device
    theta_u = torch.tensor(
        [0.3, 0.5, 0.2],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    g = torch.tensor(
        0.3, dtype=torch.float32, requires_grad=True, device=device
    )
    ecc = torch.tensor(
        0.0, dtype=torch.float32, requires_grad=True, device=device
    )
    radius = torch.tensor(
        # 0.0,
        -1,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    alpha = torch.tensor(
        10.0,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    amplitude = torch.tensor(
        [-15, -15],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    dist = 2
    dampening = 0.001
    h = 0.1

    with torch.no_grad():
        exomild.fit_params(y, lbda)

    batch_rotation = BatchRotationOperator(
        device=device,
        in_size=H,
        out_size=H,
        mode="bicubic",
        zero_init=True,
    )

    lr = 1e-5
    optimizer = optim.SGD([amplitude, g, ecc, radius, theta_u], lr=lr)

    all_losses = []
    all_im = defaultdict(list)
    with torch.autograd.detect_anomaly(check_nan=True):
        # for i in range(200):
        for i in range(100):
            # for i in range(5):
            print(f"\nIteration {i}")
            print(f"{amplitude=}")
            optimizer.zero_grad()
            im = disk.forward(
                radius=radius,
                theta_u=theta_u,
                dist=dist,
                h=h,
                alpha=alpha,
                ecc=ecc,
                g=g,
                dampening=dampening,
            )

            # forward model
            im = im.view(1, 1, H, H).expand(-1, T, -1, -1)
            # (b, T, H, W)

            im = batch_rotation.forward(x=im, rot=rot)
            # (b, T, H, W)

            im = im.view(T, 1, H, W)
            # (b * T, 1, H, W)

            im = im * torch.exp(amplitude).view(1, 2, 1, 1)
            # (b * T, C, H, W)

            im = F.conv2d(im, weight=psf_crop, padding="same", groups=2)
            # (b * T, C, H, W)

            im = im.view(1, T, C, H, W)
            # (1, T, C, H, W)

            im = im.permute(0, 2, 1, 3, 4)
            # (1, C, T, H, W)

            diff = y - im
            log_likelihood = exomild.get_log_likelihood(diff, lbda)
            # (C, T, H', W')
            if i % 5 == 0:
                for j in range(len(log_likelihood)):
                    all_im[j].append(-log_likelihood[j][0, 0].detach())
            losses = [torch.mean(-ll) for ll in log_likelihood]
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            print(f"losses={[l.item() for l in losses]}")
            print(f"{loss.item()=}")
            all_losses.append(loss.item())

    plt.plot(all_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.show()

    plt.imshow(im)


    for j, _im in all_im.items():
        print(f"{j=}")
        _im = torch.stack(_im).cpu().numpy()
        cube_3d_viewer(_im)


if __name__ == "__main__":
    main()


# sum(sum(sum(sum(sum(exomild.all_terms[0].buffered_params['C_inv'])))))
# sum(sum(sum(sum(exomild.all_terms[0].buffered_params['mean']))))