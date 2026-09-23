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
    path = Path(ROOT / "results/iterative")
    pattern = re.compile(r"mu_smooth([0-9.]+)_mu_sparse([0-9.]+)\.npz")

    # --- collect all files and extract the (mu_smooth, mu_sparse) values ---
    files = []
    mu_smooth_list = []
    mu_sparse_list = []

    for f in path.glob("mu_smooth*_mu_sparse*.npz"):
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
    shape_y = np.load(files[0][0], allow_pickle=True)["y"].shape

    # allocate the storage array: shape = (K, J, ...) where K,J are # of mus
    x_disk_store = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)) + shape_x)
    x_gt = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)) + shape_x)
    y = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)) + shape_y)
    RMSE = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))

    # --- fill the storage array ---
    for f, ms, mp in files:
        k = idx_smooth[ms]   # index for mu_smooth
        j = idx_sparse[mp]   # index for mu_sparse
        data = np.load(f, allow_pickle=True)
        x_disk_store[k, j] = data["x"]
        RMSE[k, j] = data["mse"]

    x_gt = data["x_gt"]
    y = data["y"]
    n_sparse = len(mu_sparse_vals)
    n_smooth = len(mu_smooth_vals)

    # Find best parameters
    idx_best = np.unravel_index(np.argmin(RMSE), RMSE.shape)
    k_best, j_best = idx_best
    best_x_disk = x_disk_store[k_best, j_best]
    best_mse = RMSE[k_best, j_best]
    s1, s2 = RMSE.shape 

    ##  Visualization
    plt.figure(1)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(y[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(np.squeeze(best_x_disk[0,:,:])); plt.title(r"$\mathbf{x}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(np.squeeze(x_gt[0,:,:])); plt.title(r"$\mathbf{x_{\rm GT}}$"); plt.colorbar()
    plt.subplot(2,2,4); plt.imshow(RMSE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{RMSE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()