import sys, pathlib, os

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

from matplotlib.colors import LogNorm
import torch
import hydra
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    
    data = np.load(ROOT / f"results/realdata/PDS_70.npz", allow_pickle=True)
    y = data["y"]
    x_disk_store = data["x"]
    nsmooth = data["n_smooth"]
    nsparse = data["n_sparse"]

    k,j = 1,2
    plt.figure(1)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(y[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0^{(1)}$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(np.squeeze(y[0,1,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0^{(2)}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(np.squeeze(x_disk_store[k,j][0,:,:])); plt.title(r"$x$"); plt.colorbar()
    plt.subplot(2,2,4); plt.imshow(np.squeeze(x_disk_store[k,j][1,:,:])); plt.title(r"$x$"); plt.colorbar()
    plt.tight_layout()
    plt.show()
    # plt.savefig("gridsearch.jpg", dpi=300)


    plt.figure(2)
    for k in range(3):
        for j in range(3):
            h=k*4+j+1
            sm = nsmooth+k; 
            sp = nsparse+j
            plt.subplot(4,4,h); plt.imshow(np.squeeze(np.mean(x_disk_store[k,j], axis=0))); plt.title(rf"$(10^{{{sm}}},\, 10^{{{sp}}})$"); plt.colorbar()
            plt.suptitle(r"$\mathbf{x}$");
            plt.tight_layout()
            plt.show()

    plt.figure(2)
    for k in range(3):
        for j in range(3):
            h=k*4+j+1
            sm = nsmooth+k; 
            sp = nsparse+j
            plt.subplot(1,4,h); plt.imshow(np.squeeze(np.mean(x_disk_store[0,2], axis=0))); plt.title(rf"$(10^{{{sm}}},\, 10^{{{sp}}})$"); plt.colorbar()
            plt.suptitle(r"$\mathbf{x}$");
            plt.tight_layout()
            plt.show()

    plt.figure(3)
    plt.imshow(np.squeeze(np.mean(x_disk_store[0,3], axis=0)), cmap='inferno'); plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.axis('off')
    plt.savefig("figures/reconstruction.jpg", dpi=300, bbox_inches='tight', pad_inches=0)

    j=1
    plt.figure(3)
    plt.subplot(1,2,1);plt.imshow(x_disk_store[0,0][0], cmap='inferno'); plt.colorbar()
    plt.subplot(1,2,2);plt.imshow(x_disk_store[0,0][1], cmap='inferno'); plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.axis('off')
    plt.savefig("figures/reconstruction.jpg", dpi=300, bbox_inches='tight', pad_inches=0)

    from astropy.io import fits
    hdu = fits.PrimaryHDU(data=x_disk_store[1,0]);
    hdu.writeto("results/HR4796.fits", overwrite=True);

if __name__ == "__main__":
    main()
