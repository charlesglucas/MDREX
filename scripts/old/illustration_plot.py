import sys, pathlib, os
# Configure GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

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

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HR_4796/2015-02-03/IRDIS/data/"
    # path_folder = "/scratch/vasher/tbodrito/exo/data/real_data/HD_95086/2015-05-05"
    inputs = load_folder(path_folder=path_folder, use_centered=False, channel_sortframes=0, channel_idx=None,)
    y = inputs["y"].astype(np.float32) # (C, T, H, W) 
    C, T, H, W = y.shape
    print(f"{y.shape=}")
    lbda = inputs["lbdas"].astype(np.float32)
    print(f"{lbda=}")
    rot = inputs["rot"].astype(np.float32)
    print(f"{rot.shape=}")
    psf = inputs["psf"].astype(np.float32)
    print(f"{psf.shape=}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_rotation = BatchRotationOperator(device=device, in_size=H, out_size=H, mode="bicubic", zero_init=True,)  
    y_tensor = torch.tensor(y, device=device)
    rot_tensor = torch.tensor(rot, device=device) 

    y_res = y_tensor - y_tensor.mean(dim=1, keepdim=True)
    disk = batch_rotation.forward(x=y_res, rot=-rot_tensor.expand(C,-1))  # (C, T, h, w)
    disk = disk.mean(dim=1, keepdim=False) # (C, h, w)

    ##  Visualization
    plt.figure(1)
    plt.imshow(np.squeeze(y[0,0]), cmap='inferno',vmin=-2,vmax=100)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

    plt.figure(2)
    plt.imshow(np.squeeze(psf[0]), cmap='inferno')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

    plt.figure(3)
    plt.imshow(np.squeeze(disk.detach().cpu().numpy()[0]), cmap='inferno', vmin=0, vmax=5)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

if __name__ == "__main__":
    main()