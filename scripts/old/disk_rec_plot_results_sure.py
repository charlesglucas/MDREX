import sys, pathlib, os

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]

from matplotlib.colors import LogNorm
import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load results from grid search
    data = np.load(ROOT / "results/results_hierarchic_ellipse1em5_grid_sure.npz", allow_pickle=True) 
    # data = np.load("/scratch2/clear/chalucas/codes/DiscRecCopy/results/results_hierarchic_ellipse1em6_grid2.npz", allow_pickle=True) 
    # data = np.load("/scratch2/clear/chalucas/codes/DiscRecCopy/results/results_iterative_ellipse1em5_grid_iter1000.npz", allow_pickle=True) 
    x_disk_store = data["x"] 
    x_gt = data["x_gt"]
    MSE = data["mse"]
    SURE = data["sure"]
    n_sparse = data["n_sparse"]
    n_smooth = data["n_smooth"]

    # Find best parameters
    idx_best = np.unravel_index(np.argmin(MSE), MSE.shape)
    k_best, j_best = idx_best
    best_x_disk_mse = x_disk_store[k_best, j_best]
    idx_best = np.unravel_index(np.argmin(SURE), SURE.shape)
    k_best, j_best = idx_best
    best_x_disk_sure = x_disk_store[k_best, j_best]

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
    plt.subplot(2,2,3); plt.imshow(SURE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{SURE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.subplot(2,2,4); plt.imshow(MSE, cmap="viridis", norm=LogNorm());  plt.title(r"$\mathrm{MSE}$"); plt.colorbar()
    exponents = np.arange(s1) + n_smooth; tick_values = np.arange(s1); plt.xticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    exponents = np.arange(s2) + n_sparse; tick_values = np.arange(s2); plt.yticks(tick_values, [f"$10^{{{e}}}$" for e in exponents])
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.show()

    # # Plot solution and ground truth
    # plt.figure(2)
    # plt.subplot(1,2,1); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0$"); plt.colorbar()
    # plt.subplot(1,2,2); plt.imshow(np.squeeze(x_gt[0,:,:])); plt.title(r"$\mathbf{x_{\rm GT}}$"); plt.colorbar()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()