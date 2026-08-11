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

    # Load results
    shape = "medium_ellipse"
    flux = "1em6"
    path = Path(ROOT / f"results/grids_111111111111/grid_sure_{shape}_alpha{flux}")
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
    plt.subplot(2,2,3); plt.imshow(SURE, cmap="viridis");  plt.title(r"$\mathrm{SURE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.plot(j_sure_best, k_sure_best, "rx", markersize=12);
    plt.subplot(2,2,4); plt.imshow(MSE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{MSE}$"); plt.colorbar()
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

    # fig, ax = plt.figure(3)
    plt.subplots(1,2, figsize=(7, 3))
    plt.subplot(1,2,1); plt.imshow(SURE, cmap="viridis");  plt.title(r"$\mathrm{SURE}$"); plt.colorbar(shrink=1)
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.plot(j_sure_best, k_sure_best, "rx", markersize=12);
    plt.subplot(1,2,2); plt.imshow(MSE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{MSE}$"); plt.colorbar(shrink=1)
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.plot(j_mse_best, k_mse_best, "rx", markersize=12);
    plt.tight_layout()
    plt.savefig(f"figures/grids_{shape}_{flux}.pdf")
    plt.show()


if __name__ == "__main__":
    main()