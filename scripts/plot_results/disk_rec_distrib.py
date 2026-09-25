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
from scipy.ndimage import affine_transform

import astropy.io.fits as fits


def derotate(img, angle_deg, center=(126.5, 126.5)):
    """Rotate a disk back by its angle (bicubic). The synthetic disks rotate around pixel
    (511.5, 511.5) of the 1024x1024 images, i.e. (126.5, 126.5) in the [513-128:513+128] crop
    (checked on the ground truths: exact at 180 deg)."""
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c = np.array(center)
    return affine_transform(img, R, offset=c - R @ c, order=3)


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):

    # Load ground truth
    shape = "medium_ellipse"
    flux = 5e-6
    suffix = "5em6"
    x_gt_store = np.zeros((6, 10, 256, 256), dtype=np.float32)
    for d in range(6):
        for (a, angle) in enumerate(range(0, 325, 36)):
            path_disk = ROOT / f"data/synthetic_disks/{shape}/"
            filename = f"hid_fake_disk_image_{shape}_{angle}degrees.fits"
            with fits.open(path_disk / filename) as hdul:
                data = hdul[0].data
            if data.dtype.byteorder not in ('=', '|'):
                data = data.byteswap().view(data.dtype.newbyteorder('='))
            x_gt = flux*data[513-128:513+128, 513-128:513+128]
            x_gt_store[d, a, :, :] = x_gt

    mdrex_dir = "musmooth1e6_musparse1e6"
    path = ROOT / f"results/distributions_5em6/{shape}_{mdrex_dir}/x_opt.fits"
    with fits.open(path) as hdul:
        x_mdrex = hdul[0].data
   
    NMSE_mdrex = np.zeros((6, 10), dtype=np.float32)
    NMSEsupp_mdrex = np.zeros((6, 10), dtype=np.float32)
    for d in range(6):
        for (a, angle) in enumerate(range(0, 325, 36)):
            NMSE_mdrex[d,a] = np.sqrt(np.mean((x_gt_store[d,a] - x_mdrex[d,a])**2)) / np.sqrt(np.mean(x_gt_store[d,a]**2))
            mask = x_gt_store[d, a] != 0
            NMSEsupp_mdrex[d,a] = np.sqrt(np.mean((x_gt_store[d,a][mask] - x_mdrex[d,a][mask])**2)/np.mean((x_gt_store[d,a][mask]**2)))
    

    # ------------------------------------------------------------
    # 2. Table plots
    # ------------------------------------------------------------

    # Whole image
    mean_vals_whole = np.mean(-20*np.log10(NMSE_mdrex), 1)
    std_vals_whole  = np.std(-20*np.log10(NMSE_mdrex), 1)

    mean_whole = mean_vals_whole.reshape(2, 3)
    std_whole  = std_vals_whole.reshape(2, 3)

    # Disk support
    mean_vals_supp = np.mean(-20*np.log10(NMSEsupp_mdrex), 1)
    std_vals_supp  = np.std(-20*np.log10(NMSEsupp_mdrex), 1)

    mean_supp = mean_vals_supp.reshape(2, 3)
    std_supp  = std_vals_supp.reshape(2, 3)

    rows = ["1", "4"]   # N_res = 1 resolution, 4 resolutions
    cols = ["$1$-folded", "$2$-folded", "$4$-folded"]

    print("\\begin{table*}[h!]")
    print("\\caption{{\\bf Multi-Distribution assessment.} "
        "N-RMSE averaged over parallactic angles of MD-REX reconstruction "
        "for the elliptic disk with contrast $\\alpha = 5\\cdot 10^{-6}$ "
        "using different patch distribution sets.}")
    print("\\label{table:1}")
    print("\\centering")
    print("\\begin{tabular}{c c c c}")
    print("\\hline")
    print("$N_{\\rm res}$ & " + " & ".join(cols) + " \\\\")
    print("\\hline")

    # Whole image block
    print("\\multicolumn{4}{c}{Whole image} \\\\")
    for i, r in enumerate(rows):
        line = " & ".join([f"${mean_whole[i,j]:.2f} \\pm {std_whole[i,j]:.2f}$" 
                        for j in range(3)])
        print(f"{r} & {line} \\\\")

    print("\\hline")

    # Disk support block
    print("\\multicolumn{4}{c}{Disk support} \\\\")
    for i, r in enumerate(rows):
        line = " & ".join([f"${mean_supp[i,j]:.2f} \\pm {std_supp[i,j]:.2f}$" 
                        for j in range(3)])
        print(f"{r} & {line} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table*}")


    ## Plot reconstruction error, averaged over the disk angles
    # each reconstruction is derotated by its angle before averaging. The ground truths are derotated
    # and averaged the same way, so that the interpolation blur is the same on both sides.
    angles = list(range(0, 325, 36))
    x_mdrex_mean = np.stack([
        np.mean([derotate(x_mdrex[k][a], angle) for (a, angle) in enumerate(angles)], axis=0)
        for k in range(6)
    ])
    x_gt_mean = np.stack([
        np.mean([derotate(x_gt_store[k][a], angle) for (a, angle) in enumerate(angles)], axis=0)
        for k in range(6)
    ])
    titles = ["1-folded", "2-folded", "4-folded"]
    row_labels = ["1 resolution", "4 resolutions"]

    fig, axs = plt.subplots(2, 3, figsize=(9, 5.4))
    fig.subplots_adjust(wspace=0.05, hspace=0.05, right=0.9)

    ims = np.empty((2, 3), dtype=object)
    for k in range(6):
        i, c = divmod(k, 3)
        ims[i, c] = axs[i, c].imshow(np.abs(x_mdrex_mean[k] - x_gt_mean[k])/flux, cmap='gray', vmin=0, vmax=.15)
        axs[i, c].set_xticks([])
        axs[i, c].set_yticks([])
        for spine in axs[i, c].spines.values():
            spine.set_visible(False)

    for c in range(3):
        axs[0, c].set_title(titles[c])

    for i in range(2):
        axs[i, 0].set_ylabel(row_labels[i])
        fig.colorbar(ims[i, 2], ax=axs[i, :], location='right', pad=0.02, shrink=0.9)

    plt.savefig(
        f"figures/distrib_{shape}_{suffix}_{mdrex_dir}.pdf",
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.05,
    )
    plt.show()


if __name__ == "__main__":
    main()