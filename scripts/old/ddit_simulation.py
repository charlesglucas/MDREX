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

from DDiT import Disk

@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    disk = Disk()
    disk.compute_model(a = 1, incl = 0, e = 0.2, omega = 0, pa = 90, pin = 10, pout = -10, gsca = 0.2, gpol = 0, opang = 0.2)
    #   - a: reference semi-major axis in arcseconds
    #   - incl: inclination in degrees
    #   - pa: position angle in degrees
    #   - pin: inner slope
    #   - pout: outer slope
    #   - gsca: HG coefficient for total intensity
    #   - gpol: HG coefficient for polarized intensity
    #   - e: eccentricity
    #   - omega: argument of pericenter in degrees
    #   - opang: opening angle of the disk
    #   - s11: the scattered light phase function
    #   - s12: the polarized light phase function
    #  disk.compute_model(e = 0.2, incl = 70.1, pa = 110., a = 0.89, gsca = 0.4, gpol = 0.6, omega = 80., opang = 0.035, pin = 20.0, pout = -5.5, pmid = 0.5, da = 0.5)
    # disk.plot()

    im = disk.intensity
    x_min = im.min(); x_max = im.max(); x_norm = (im - x_min) / (x_max - x_min)

    plt.figure(2)
    plt.imshow(x_norm); plt.title("Disk"); plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
