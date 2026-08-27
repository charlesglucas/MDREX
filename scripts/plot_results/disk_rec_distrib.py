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

import astropy.io.fits as fits


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):

    # Load ground truth
    shape = "medium_ellipse"
    flux = 5e-6
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

    path = ROOT / f"results/distributions_5em6/musmooth1e6_musparse1e6/x_opt.fits"
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


    # ## Whole MSE
    # mean_vals = np.mean(NMSE_mdrex, 1)
    # std_vals  = np.std(NMSE_mdrex, 1)

    # mean_mat = mean_vals.reshape(2, 3)
    # std_mat  = std_vals.reshape(2, 3)

    # rows = ["1 resolution", "All resolutions"]
    # cols = ["$1$-folded", "$2$-folded", "$4$-folded"]

    # print("\\begin{table}[ht!]")
    # print("\\caption{{\\bf Multi-Distribution assessment.} N-RMSE averaged over parallactic angles of MD-REX reconstruction for the elliptic disk with contrast $5.10^{-6}$ using different patch distribution sets.}")
    # print("\\label{table:1}")
    # print("\\centering")
    # print("\\begin{tabular}{c c c c}")
    # print("\\hline\\hline")
    # print(" & " + " & ".join(cols) + " \\\\")
    # print("\\hline")

    # for i, row in enumerate(rows):
    #     line = " & ".join([f"${mean_mat[i,j]:.4f} \\pm {std_mat[i,j]:.4f}$" for j in range(3)])
    #     print(f"{row} & {line} \\\\")

    # print("\\hline")
    # print("\\end{tabular}")
    # print("\\end{table}")

    # ## Support MSE
    # mean_vals = np.mean(NMSEsupp_mdrex, 1)
    # std_vals  = np.std(NMSEsupp_mdrex, 1)

    # mean_mat = mean_vals.reshape(2, 3)
    # std_mat  = std_vals.reshape(2, 3)

    # rows = ["1 resolution", "All resolutions"]
    # cols = ["$1$-folded", "$2$-folded", "$4$-folded"]

    # print("\\begin{table}[ht!]")
    # print("\\caption{{\\bf Multi-Distribution assessment.} N-RMSE on support averaged over parallactic angles of MD-REX reconstruction for the elliptic disk with contrast $5.10^{-6}$ using different patch distribution sets.}")
    # print("\\label{table:1}")
    # print("\\centering")
    # print("\\begin{tabular}{c c c c}")
    # print("\\hline\\hline")
    # print(" & " + " & ".join(cols) + " \\\\")
    # print("\\hline")

    # for i, row in enumerate(rows):
    #     line = " & ".join([f"${mean_mat[i,j]:.4f} \\pm {std_mat[i,j]:.4f}$" for j in range(3)])
    #     print(f"{row} & {line} \\\\")

    # print("\\hline")
    # print("\\end{tabular}")
    # print("\\end{table}")

    ## Plot example reconstructions
    d = 5
    a = 0
    plt.figure(1)
    plt.subplot(1,2,1); plt.imshow(x_gt_store[d,a]); plt.title(r"$xgt$"); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(x_mdrex[d,a]);  plt.title(r"MD-REX"); plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()