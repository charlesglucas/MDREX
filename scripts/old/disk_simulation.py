import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import hydra
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

from utils.viz import cube_3d_viewer
from disk.debris_disk import Disk
from utils.rotation import BatchRotationOperator


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disk = Disk(z_max=100, n_layers=10001)

    dist =  1
    dampening = 1
    h = 1

    theta_u = torch.tensor(
        #[-np.pi/8, -np.pi/5, -np.pi/5],
        [np.pi/4, 0, 0],
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    g = torch.tensor(
        0, dtype=torch.float32, requires_grad=True, device=device
    )
    ecc = torch.tensor(
        .99, dtype=torch.float32, requires_grad=True, device=device
    )
    radius = torch.tensor(
        # 0.0,
        # -1.5,
        0,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    alpha = torch.tensor(
        4,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

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
    
    x_min = im.min(); x_max = im.max(); x_norm = (im - x_min) / (x_max - x_min)
    
    plt.figure(1)
    plt.imshow(np.squeeze(x_norm.detach().cpu().numpy())); plt.title("Disk"); plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
