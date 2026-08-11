import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import hydra
from torch import optim
import matplotlib.pyplot as plt

from utils.viz import cube_3d_viewer
from disk.debris_disk import Disk
from utils.rotation import BatchRotationOperator


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    C, T, H, W = 1, 64, 256, 256

    y = torch.zeros((C, T, H, W), device=device)

    rot = torch.linspace(0, 45, T, device=device)

    print(f"{y.shape=}")

    disk = Disk(z_max=0.2, n_layers=1)

    device = y.device
    theta_u_gt = torch.tensor(
        [0.3, 0.5, 0.2],
        dtype=torch.float32,
        device=device,
    )
    g_gt = torch.tensor(0.3, dtype=torch.float32, device=device)
    ecc_gt = torch.tensor(0.0, dtype=torch.float32, device=device)
    radius_gt = torch.tensor(
        -1,
        dtype=torch.float32,
        device=device,
    )

    alpha_gt = torch.tensor(
        10.0,
        dtype=torch.float32,
        device=device,
    )

    dist = 2
    dampening = 0.001
    h = 0.1

    im_gt = disk.forward(
        radius=radius_gt,
        theta_u=theta_u_gt,
        dist=dist,
        h=h,
        alpha=alpha_gt,
        ecc=ecc_gt,
        g=g_gt,
        dampening=dampening,
    )
    # (H, W)

    # forward model
    im_gt = im_gt.view(1, 1, H, H).expand(-1, T, -1, -1)
    # (b, T, H, W)

    rot = rot.view(1, T)
    # (b, T)

    batch_rotation = BatchRotationOperator(
        device=device,
        in_size=H,
        out_size=H,
        mode="bicubic",
        zero_init=True,
    )

    im_gt = batch_rotation.forward(x=im_gt, rot=rot)
    # (b, T, H, W)

    theta_u = torch.tensor(
        [0.1, 0.8, 0.9],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    g = torch.tensor(
        0.3, dtype=torch.float32, requires_grad=True, device=device
    )
    ecc = torch.tensor(
        0.2, dtype=torch.float32, requires_grad=True, device=device
    )
    radius = torch.tensor(
        # 0.0,
        # -1.5,
        -0.5,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    alpha = torch.tensor(
        5.0,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    amplitude = torch.tensor(
        [-10],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    lr = 5e-5
    optimizer = optim.SGD([amplitude, g, ecc, radius, theta_u, alpha], lr=lr)

    all_im = []
    all_losses = []
    with torch.autograd.detect_anomaly(check_nan=True):
        for i in range(200):
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
            # (H, W)

            # forward model
            im_rot = im.view(1, 1, H, H).expand(-1, T, -1, -1)
            # (b, T, H, W)

            im_rot = batch_rotation.forward(x=im_rot, rot=rot)
            # (b, T, H, W)

            im_rot = im_rot.view(1, T, H, W)
            # (b, T, H, W)

            diff = im_gt - im_rot
            # (b, T, H, W)
            loss = torch.mean(diff**2)
            loss.backward()
            optimizer.step()
            print(f"{i=:3d} {loss.item()=:.4f} {alpha.item()=:.4f}")
            with torch.no_grad():
                all_im.append(diff[0, 0].detach().cpu())
                all_losses.append(loss.detach().cpu().item())

    all_im = torch.stack(all_im)
    plt.plot(all_losses)
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.show()
    cube_3d_viewer(all_im.cpu().numpy(), quantile=0)


if __name__ == "__main__":
    main()
