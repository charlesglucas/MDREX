import sys, pathlib, os

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
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
    x_gt_store = np.zeros((3, 10, 3, 256, 256), dtype=np.float32)
    for (s, shape) in enumerate(["medium_ellipse", "spiral", "circle"]):
        for (a, angle) in enumerate(range(0, 325, 36)):
            path_disk = ROOT / f"data/synthetic_disks/{shape}/"
            filename = f"hid_fake_disk_image_{shape}_{angle}degrees.fits"
            with fits.open(path_disk / filename) as hdul:
                data = hdul[0].data
            if data.dtype.byteorder not in ('=', '|'):
                data = data.byteswap().view(data.dtype.newbyteorder('='))
            datadisk = data[513-128:513+128, 513-128:513+128]
            for (f, flux) in enumerate([1e-6, 5e-6, 1e-5]):
                datadisk = data[513-128:513+128, 513-128:513+128]
                x_gt = flux*datadisk
                x_gt_store[f, a, s, :, :] = x_gt

    # Load results
    path = ROOT / "results/Rexpaco_results/x_opt.fits"
    with fits.open(path) as hdul:
        x_rexpaco = hdul[0].data[:,:,:,251-128:251+128, 251-128:251+128]

    path = ROOT / "results/mdrex_results/musmooth1e6_musparse1e6/x_opt.fits"
    with fits.open(path) as hdul:
        x_mdrex = hdul[0].data
   
    NMSE_rexpaco = np.zeros((3, 10, 3), dtype=np.float32)
    NMSE_mdrex = np.zeros((3, 10, 3), dtype=np.float32)
    NMSEsupp_rexpaco = np.zeros((3, 10, 3), dtype=np.float32)
    NMSEsupp_mdrex = np.zeros((3, 10, 3), dtype=np.float32)
    for (s, shape) in enumerate(["medium_ellipse", "spiral", "circle"]):
        for (a, angle) in enumerate(range(0, 325, 36)):
            for (f, flux) in enumerate([1e-6, 5e-6, 1e-5]):
                NMSE_rexpaco[f,a,s] = np.sqrt(np.sum((x_gt_store[f,a,s] - x_rexpaco[f,a,s])**2)) / np.sqrt(np.sum(x_gt_store[f,a,s]**2))
                NMSE_mdrex[f,a,s] = np.sqrt(np.sum((x_gt_store[f,a,s] - x_mdrex[f,a,s])**2)) / np.sqrt(np.sum(x_gt_store[f,a,s]**2))
                mask = x_gt_store[f, a, s] > 2e-7
                NMSEsupp_rexpaco[f,a,s] = np.sqrt(np.sum((x_gt_store[f,a,s][mask] - x_rexpaco[f,a,s][mask])**2)/np.sum((x_gt_store[f,a,s][mask]**2)))
                NMSEsupp_mdrex[f,a,s] = np.sqrt(np.sum((x_gt_store[f,a,s][mask] - x_mdrex[f,a,s][mask])**2)/np.sum((x_gt_store[f,a,s][mask]**2)))
   
    # ------------------------------------------------------------
    # 2. Table plots
    # ------------------------------------------------------------

    # Whole MSE
    mean_mdrex      = np.mean(-20*np.log10(NMSE_mdrex), 1)
    std_mdrex       = np.std(-20*np.log10(NMSE_mdrex), 1)
    mean_rexpaco    = np.mean(-20*np.log10(NMSE_rexpaco), 1)
    std_rexpaco     = np.std(-20*np.log10(NMSE_rexpaco), 1)

    # Support MSE
    mean_mdrex_supp   = np.mean(-20*np.log10(NMSEsupp_mdrex), 1)
    std_mdrex_supp    = np.std(-20*np.log10(NMSEsupp_mdrex), 1)
    mean_rexpaco_supp = np.mean(-20*np.log10(NMSEsupp_rexpaco), 1)
    std_rexpaco_supp  = np.std(-20*np.log10(NMSEsupp_rexpaco), 1)

    shapes = ["Ellipse", "Spiral", "Circle"]
    contrasts = ["$\\alpha = 1\\cdot 10^{-6}$", "$\\alpha = 5\\cdot 10^{-6}$", "$\\alpha = 1\\cdot10^{-5}$"]

    print("\\begin{table*}[h!]")
    print("\\centering")
    print("\\caption{{\\bf Comparison of performances.} PSNR (whole image and support) averaged over parallactic angles for MD-REX and REXPACO reconstructions.}")
    print("\\begin{tabular}{llcccccc}")
    print("\\hline")
    print(" & & \\multicolumn{3}{c}{Whole image} & \\multicolumn{3}{c}{Support} \\\\")
    print(" & & " + " & ".join(contrasts) + " & " + " & ".join(contrasts) + " \\\\")
    print("\\hline")

    for i, c in enumerate(shapes):

        # REXPACO
        row_rexpaco_whole = " & ".join([f"${mean_rexpaco[j,i]:.2f} \\pm {std_rexpaco[j,i]:.2f}$" 
                                        for j in range(3)])
        row_rexpaco_supp  = " & ".join([f"${mean_rexpaco_supp[j,i]:.2f} \\pm {std_rexpaco_supp[j,i]:.2f}$" 
                                        for j in range(3)])
        print(f"\\multirow{{2}}{{*}}{{{c}}} & REXPACO & {row_rexpaco_whole} & {row_rexpaco_supp} \\\\")

        # MD-REX
        row_mdrex_whole = " & ".join([f"${mean_mdrex[j,i]:.2f} \\pm {std_mdrex[j,i]:.2f}$" 
                                    for j in range(3)])
        row_mdrex_supp  = " & ".join([f"${mean_mdrex_supp[j,i]:.2f} \\pm {std_mdrex_supp[j,i]:.2f}$" 
                                    for j in range(3)])
        print(f" & MD-REX & {row_mdrex_whole} & {row_mdrex_supp} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table*}")

    # ## Whole MSE
    # mean_mdrex = np.mean(NMSE_mdrex, 1)
    # std_mdrex  = np.std(NMSE_mdrex, 1)

    # mean_rexpaco = np.mean(NMSE_rexpaco, 1)
    # std_rexpaco  = np.std(NMSE_rexpaco, 1)

    # shapes = ["Ellipse", "Spiral", "Circle"]
    # contrasts = ["$10^{-6}$", "$5\\cdot 10^{-6}$", "$10^{-5}$"]

    # print("\\begin{table}[h!]")
    # print("\\centering")
    # print("\\caption{{\\bf Comparison of performances.} N-RMSE averaged over parallactic angles of MD-REX and REXPACO reconstructions for the elliptic disk with different contrasts using all the patch distributions.}")
    # print("\\begin{tabular}{llccc}")
    # print("\\hline")
    # print(" & & " + " & ".join(contrasts) + " \\\\")
    # print("\\hline")

    # for i, c in enumerate(shapes):
    #     # REXPACO
    #     row_rexpaco = " & ".join([f"${mean_rexpaco[j,i]:.2f} \\pm {std_rexpaco[j,i]:.2f}$" 
    #                             for j in range(3)])
    #     print(f"\\multirow{{2}}{{*}}{{{c}}} & REXPACO & {row_rexpaco} \\\\")
        
    #     # MD-REX
    #     row_mdrex = " & ".join([f"${mean_mdrex[j,i]:.2f} \\pm {std_mdrex[j,i]:.2f}$" 
    #                             for j in range(3)])
    #     print(f" & MD-REX & {row_mdrex} \\\\")

    # print("\\hline")
    # print("\\end{tabular}")
    # print("\\end{table}")

    # ## Support MSE
    # mean_mdrex = np.mean(NMSEsupp_mdrex, 1)
    # std_mdrex  = np.std(NMSEsupp_mdrex, 1)

    # mean_rexpaco = np.mean(NMSEsupp_rexpaco, 1)
    # std_rexpaco  = np.std(NMSEsupp_rexpaco, 1)

    # shapes = ["Ellipse", "Spiral", "Circle"]
    # contrasts = ["$10^{-6}$", "$5\\cdot 10^{-6}$", "$10^{-5}$"]

    # print("\\begin{table}[h!]")
    # print("\\centering")
    # print("\\caption{{\\bf Comparison of performances.} N-RMSE on support averaged over parallactic angles of MD-REX and REXPACO reconstructions for the elliptic disk with different contrasts using all the patch distributions.}")
    # print("\\begin{tabular}{llccc}")
    # print("\\hline")
    # print(" & & " + " & ".join(contrasts) + " \\\\")
    # print("\\hline")

    # for i, c in enumerate(shapes):
    #     # REXPACO
    #     row_rexpaco = " & ".join([f"${mean_rexpaco[j,i]:.2f} \\pm {std_rexpaco[j,i]:.2f}$" 
    #                             for j in range(3)])
    #     print(f"\\multirow{{2}}{{*}}{{{c}}} & REXPACO & {row_rexpaco} \\\\")
        
    #     # MD-REX
    #     row_mdrex = " & ".join([f"${mean_mdrex[j,i]:.2f} \\pm {std_mdrex[j,i]:.2f}$" 
    #                             for j in range(3)])
    #     print(f" & MD-REX & {row_mdrex} \\\\")

    # print("\\hline")
    # print("\\end{tabular}")
    # print("\\end{table}")


    ## Plot example reconstructions
    f = 0
    a = 0
    s = 1
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.7), constrained_layout=True)
    im0 = axes[0].imshow(x_gt_store[f, a, s])
    axes[0].set_title(r"$Ground Truth$")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar0.ax.yaxis.set_offset_position('right')
    cbar0.ax.yaxis.get_offset_text().set_visible(True)
    cbar0.ax.yaxis.get_offset_text().set_horizontalalignment('left')
    cbar0.ax.yaxis.get_offset_text().set_verticalalignment('bottom')
    cbar0.ax.yaxis.get_offset_text().set_fontsize(12)

    im1 = axes[1].imshow(x_rexpaco[f, a, s])
    axes[1].set_title(r"REXPACO")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar1.ax.yaxis.set_offset_position('right')
    cbar1.ax.yaxis.get_offset_text().set_visible(True)
    cbar1.ax.yaxis.get_offset_text().set_horizontalalignment('left')
    cbar1.ax.yaxis.get_offset_text().set_verticalalignment('bottom')
    cbar1.ax.yaxis.get_offset_text().set_fontsize(12)

    im2 = axes[2].imshow(x_mdrex[f, a, s])
    axes[2].set_title(r"MD-REX")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar2.ax.yaxis.set_offset_position('right')
    cbar2.ax.yaxis.get_offset_text().set_visible(True)
    cbar2.ax.yaxis.get_offset_text().set_horizontalalignment('left')
    cbar2.ax.yaxis.get_offset_text().set_verticalalignment('bottom')
    cbar2.ax.yaxis.get_offset_text().set_fontsize(12)
    plt.show()

    plt.subplots(1,3, figsize=(5, 2))
    plt.subplot(1,3,1); plt.imshow(x_gt_store[0,0,0], cmap='hot'); plt.title(r"$Ellipse$"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(x_gt_store[0,0,1], cmap='hot');  plt.title(r"Spiral"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(x_gt_store[0,0,2], cmap='hot');  plt.title(r"Circle"); plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1)
    plt.savefig("figures/shapes.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    fig, axs = plt.subplots(1,2, figsize=(6,2), gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(wspace=0.03, right=0.84, top=0.84)
    data1 = x_rexpaco[0,0,0] - x_gt_store[0,0,0]
    data2 = x_mdrex[0,0,0] - x_gt_store[0,0,0]
    vmin = np.min([data1.min(), data2.min()])
    vmax = np.max([data1.max(), data2.max()])
    vmax = np.max([np.abs(vmin), np.abs(vmax)])
    vmin = -vmax
    im1 = axs[0].imshow(data1, cmap='bwr', vmin=vmin, vmax=vmax)
    im2 = axs[1].imshow(data2, cmap='bwr', vmin=vmin, vmax=vmax)
    for ax in axs:
        ax.set_title(ax.get_title() or "")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
    axs[0].set_title(r"$\mathrm{REXPACO}$")
    axs[1].set_title(r"$\mathrm{MD-REX}$")
    cbar = fig.colorbar(im2, ax=axs, location='right', pad=0.05, shrink=1)
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.set_offset_position('right')
    cbar.ax.yaxis.get_offset_text().set_visible(True)
    cbar.ax.yaxis.get_offset_text().set_horizontalalignment('left')
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')
    cbar.ax.yaxis.get_offset_text().set_fontsize(12)
    plt.savefig(
        "figures/comp_medium_ellipse_1em6.pdf",
        dpi=300,
        bbox_inches='tight',
        bbox_extra_artists=[cbar.ax.yaxis.get_offset_text()],
        pad_inches=0.05,
    )
    plt.show()

    fig, axs = plt.subplots(1,2, figsize=(6,2), gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(wspace=0.03, right=0.84, top=0.84)
    data1 = x_rexpaco[2,0,0] - x_gt_store[2,0,0]
    data2 = x_mdrex[2,0,0] - x_gt_store[2,0,0]
    vmin = np.min([data1.min(), data2.min()])
    vmax = np.max([data1.max(), data2.max()])
    vmax = np.max([np.abs(vmin), np.abs(vmax)])
    vmin = -vmax
    im1 = axs[0].imshow(data1, cmap='bwr', vmin=vmin, vmax=vmax)
    im2 = axs[1].imshow(data2, cmap='bwr', vmin=vmin, vmax=vmax)
    for ax in axs:
        ax.set_title(ax.get_title() or "")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
    axs[0].set_title(r"$\mathrm{REXPACO}$")
    axs[1].set_title(r"$\mathrm{MD-REX}$")
    cbar = fig.colorbar(im2, ax=axs, location='right', pad=0.05, shrink=1)
    cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    cbar.ax.yaxis.set_offset_position('right')
    cbar.ax.yaxis.get_offset_text().set_visible(True)
    cbar.ax.yaxis.get_offset_text().set_horizontalalignment('left')
    cbar.ax.yaxis.get_offset_text().set_verticalalignment('bottom')
    cbar.ax.yaxis.get_offset_text().set_fontsize(12)
    plt.savefig(
        "figures/comp_medium_ellipse_1em5.pdf",
        dpi=300,
        bbox_inches='tight',
        bbox_extra_artists=[cbar.ax.yaxis.get_offset_text()],
        pad_inches=0.05,
    )
    plt.show()


if __name__ == "__main__":
    main()