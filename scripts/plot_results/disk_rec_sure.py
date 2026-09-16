import sys, pathlib, os

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

from matplotlib.colors import LogNorm, SymLogNorm, TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
import torch
import hydra
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):

    # Load results
    shape = "medium_ellipse"
    flux = "1em5"
    path = Path(ROOT / f"results/grids_111111111111/grid_sure_{shape}_alpha{flux}")
    # path = Path(ROOT / f"results/grids_100100100100/grid_sure_{shape}_alpha{flux}")
    pattern = re.compile(r"musmooth([0-9.]+)_musparse([0-9.]+)\.npz")

    print("root =", path.resolve())
    print("cwd  =", Path().resolve())

    # --- collect all files and extract the (mu_smooth, mu_sparse) values ---
    files = []
    mu_smooth_list = []
    mu_sparse_list = []

    for f in path.glob("musmooth*_musparse*.npz"):
        m = pattern.match(f.name)
        if m:
            ms = float(m.group(1))   # extract mu_smooth
            mp = float(m.group(2))   # extract mu_sparse
            files.append((f, ms, mp))
            mu_smooth_list.append(ms)
            mu_sparse_list.append(mp)

    # --- build sorted unique vectors of mu_smooth and mu_sparse ---
    mu_smooth_vals = sorted(set(mu_smooth_list))
    mu_sparse_vals = sorted(set(mu_sparse_list))

    # fast lookup tables: value → index
    idx_smooth = {v: i for i, v in enumerate(mu_smooth_vals)}
    idx_sparse = {v: j for j, v in enumerate(mu_sparse_vals)}

    # # --- determine the shape of x by loading one sample ---
    shape_x = np.load(files[0][0], allow_pickle=True)["x"].shape

    # allocate the storage array: shape = (K, J, ...) where K,J are # of mus
    x_disk_store = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)) + shape_x)
    x_gt = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)) + shape_x)
    MSE = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))
    PSNR = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))
    SURE = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))
    DA = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))
    DE = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))

    # --- fill the storage array ---
    for f, ms, mp in files:
        k = idx_smooth[ms]   # index for mu_smooth
        j = idx_sparse[mp]   # index for mu_sparse
        data = np.load(f, allow_pickle=True)
        x_disk_store[k, j] = data["x"]
        x_gt[k, j] = data["x_gt"]
        MSE[k, j] = data["mse"]
        PSNR[k, j] = -20*np.log10(np.sqrt(np.mean((x_disk_store[k, j] - x_gt[k, j])**2)) / np.sqrt(np.mean(x_gt[k, j]**2)))
        SURE[k, j] = data["sure"]
        DA[k, j] = data["data_term"]
        DE[k, j] = data["div_est"]

    n_sparse = np.log10(mu_sparse_vals[0]).astype(int)
    n_smooth = np.log10(mu_smooth_vals[0]).astype(int)

    # Find best parameters
    idx_best = np.unravel_index(np.argmin(MSE), MSE.shape)
    k_mse_best, j_mse_best = idx_best
    best_x_disk_mse = x_disk_store[k_mse_best, j_mse_best]
    idx_best = np.unravel_index(np.argmin(SURE), SURE.shape)
    k_sure_best, j_sure_best = idx_best
    best_x_disk_sure = x_disk_store[k_sure_best, j_sure_best]

    # SURE can be negative, so center the color map at zero when possible.
    # If all values are positive or all are negative, fall back to a standard norm.
    finite_sure = SURE[np.isfinite(SURE)]
    if finite_sure.size:
        vmin, vmax = np.nanmin(finite_sure), np.nanmax(finite_sure)
        if vmin < 0 < vmax:
            sure_norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
        else:
            sure_norm = None
    else:
        sure_norm = None

    # Compute N-RMSE of SURE solution
    diffgt = best_x_disk_sure - x_gt
    NRMSE = np.sqrt(np.sum(diffgt.flatten()**2))/np.sqrt(np.sum(x_gt.flatten()**2))
    print(f"N-RMSE of SURE solution: {NRMSE:.4f}")  

    ##  Visualization
    # Plot grid search results
    s1, s2 = MSE.shape
    plt.figure(1)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(best_x_disk_sure.mean(axis=0, keepdims=True))); plt.title(r"$\mathbf{x}_{\rm SURE}$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(np.squeeze(best_x_disk_mse.mean(axis=0, keepdims=True))); plt.title(r"$\mathbf{x}_{\rm MSE}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(SURE, cmap="RdBu_r", norm=sure_norm);  plt.title(r"$\mathrm{SURE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.plot(j_sure_best, k_sure_best, "rx", markersize=12);
    plt.subplot(2,2,4); plt.imshow(MSE, cmap="RdBu_r", norm=LogNorm());  plt.title(r"$\mathrm{MSE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.plot(j_mse_best, k_mse_best, "rx", markersize=12);
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.subplot(1,2,1); plt.imshow(DA, cmap="viridis");  plt.title(r"$\mathrm{Data Term}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.subplot(1,2,2); plt.imshow(DE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{Divergence Estimate}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.show()

    # plt.subplots(1,2, figsize=(7, 3))
    # plt.subplot(1,2,1); plt.imshow(SURE, cmap="RdBu_r", norm=sure_norm);  plt.title(r"$\mathrm{SURE}$"); plt.colorbar(shrink=1)
    # exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    # exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    # plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    # plt.plot(j_sure_best, k_sure_best, "rx", markersize=12);
    # plt.subplot(1,2,2); plt.imshow(MSE, cmap="RdBu_r", norm=LogNorm());  plt.title(r"$\mathrm{MSE}$"); plt.colorbar(shrink=1)
    # exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    # exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    # plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    # plt.plot(j_mse_best, k_mse_best, "rx", markersize=12);
    # plt.tight_layout()
    # plt.savefig(f"figures/grids_{shape}_{flux}.pdf")
    # plt.show()

    fig = plt.figure(figsize=(5, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    im1 = ax1.imshow(MSE, cmap="RdBu_r", norm=LogNorm())
    ax1.set_title(r"$\mathrm{MSE}$", fontsize=18)
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=1)
    cbar1_formatter = ScalarFormatter(useMathText=True)
    cbar1_formatter.set_scientific(True)
    cbar1_formatter.set_powerlimits((0, 0))
    cbar1.ax.yaxis.set_major_formatter(cbar1_formatter)
    cbar1.ax.yaxis.set_offset_position('right')
    cbar1.ax.yaxis.get_offset_text().set_visible(True)
    cbar1.ax.yaxis.get_offset_text().set_horizontalalignment('left')
    cbar1.ax.yaxis.get_offset_text().set_verticalalignment('bottom')
    cbar1.ax.yaxis.get_offset_text().set_fontsize(12)
    cbar1.ax.tick_params(labelsize=12)
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); ax1.set_xticks(tick_values); ax1.set_xticklabels([f"$10^{{{e}}}$" for e in exponents], fontsize=12)
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); ax1.set_yticks(tick_values); ax1.set_yticklabels([f"$10^{{{e}}}$" for e in exponents], fontsize=12)
    ax1.set_ylabel(r"$\mu_{\rm smooth}$", fontsize=14); ax1.set_xlabel(r"$\mu_{\rm sparse}$", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.plot(j_mse_best, k_mse_best, "rx", markersize=12, markeredgewidth=2)
    ax1.plot(j_sure_best, k_sure_best, "m+", markersize=12, markeredgewidth=2)
    ax1.set_aspect('equal', adjustable='box')

    ax2 = fig.add_subplot(2, 1, 2)
    im2 = ax2.imshow(SURE, cmap="RdBu_r", norm=sure_norm)
    ax2.set_title(r"$\mathrm{MC-SURE}$", fontsize=18)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=1)
    cbar2_formatter = ScalarFormatter(useMathText=True)
    cbar2_formatter.set_scientific(True)
    cbar2_formatter.set_powerlimits((0, 0))
    cbar2.ax.yaxis.set_major_formatter(cbar2_formatter)
    cbar2.ax.yaxis.set_offset_position('right')
    cbar2.ax.yaxis.get_offset_text().set_visible(True)
    cbar2.ax.yaxis.get_offset_text().set_horizontalalignment('left')
    cbar2.ax.yaxis.get_offset_text().set_verticalalignment('bottom')
    cbar2.ax.yaxis.get_offset_text().set_fontsize(12)
    cbar2.ax.tick_params(labelsize=12)
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); ax2.set_xticks(tick_values); ax2.set_xticklabels([f"$10^{{{e}}}$" for e in exponents], fontsize=12)
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); ax2.set_yticks(tick_values); ax2.set_yticklabels([f"$10^{{{e}}}$" for e in exponents], fontsize=12)
    ax2.set_ylabel(r"$\mu_{\rm smooth}$", fontsize=14); ax2.set_xlabel(r"$\mu_{\rm sparse}$", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.plot(j_mse_best, k_mse_best, "rx", markersize=12, markeredgewidth=2)
    ax2.plot(j_sure_best, k_sure_best, "m+", markersize=12, markeredgewidth=2)
    ax2.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(f"figures/grids_{shape}_{flux}.pdf")
    plt.show()


if __name__ == "__main__":
    main()