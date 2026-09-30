import sys, pathlib, os

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import torch
import torch.nn.functional as F
import hydra
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
from astropy.io import fits

from utils.rotation import BatchRotationOperator

# Same as disk_rec_sure.py, except that the SURE data term and the Mahalanobis MSE are recomputed
# with a FIXED metric W = C_inv, estimated once per (shape, flux) instead of on each residual
# y - A x_hat (compute_mc_sure). With W re-estimated on each residual, the data term normalizes
# itself and SURE selects x = 0. The divergence saved in the result files (div_est) is reused as is.
#   W_REF = "y"  : C_inv estimated on y (residual at x = 0), usable on real data
#   W_REF = "gt" : C_inv estimated on y - A x_gt (nuisance only), oracle
#   W_REF = "pilot": C_inv estimated on y - A x_hat(mu_0), where mu_0 minimizes the SURE computed with
#                    W_REF = "y" (the disk is removed before estimating the covariance), usable on real data
# The MSE (reference) is always computed with the oracle metric W_gt.
W_REF = "pilot"


# ------------------------------------------------------------
# Forward model and patch statistics (same as reconstruction/mdrex.py)
# ------------------------------------------------------------

def load_forward_model(device):
    path_folder = ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data"
    rot = fits.getdata(path_folder / "ird_convert_recenter_dc_IRD_SCIENCE_PARA_ROTATION_CUBE_rotnth.fits").flatten()
    psf = fits.getdata(path_folder / "ird_sortframes_dc-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits")
    infrared_filter = fits.getheader(path_folder / "ird_sortframes_dc-IRD_SCIENCE_LAMBDA_INFO-lam.fits").get("HIERARCH ESO INS COMB IFLT")
    lbdas = {"DB_K12": [2.110, 2.251], "DB_H23": [1.593, 1.667]}[infrared_filter]
    rot = torch.tensor(rot.astype(np.float32), device=device)
    psf = torch.tensor(psf.astype(np.float32), device=device)
    lbda = torch.tensor([lbdas], dtype=torch.float32, device=device)  # (1, C)

    # PSF preprocessing (same as syntheticdata_sure_gpu.py)
    psf_size = psf.shape[-1]
    border = 15
    mask_out = torch.ones_like(psf)[0].bool()
    mask_out[border : psf_size - border, border : psf_size - border] = 0
    psf[0] = psf[0] - psf[0, mask_out].mean()
    psf[1] = psf[1] - psf[1, mask_out].mean()
    crop_size = 13
    psf_crop = psf[
        :,
        1 + (psf_size - crop_size) // 2 : 1 + (psf_size + crop_size) // 2,
        1 + (psf_size - crop_size) // 2 : 1 + (psf_size + crop_size) // 2,
    ].reshape(2, 1, crop_size, crop_size)

    C, T, H = 2, rot.numel(), 256
    k1k2_path = ROOT / "data/coronograph/sphere_irdis_k1_k2_coronagraph_transmission_map.fits"
    mask = fits.getdata(k1k2_path).astype(np.float32)
    deltaH = (mask.shape[1] - H) // 2
    mask = torch.tensor(mask[:, deltaH:deltaH+H, deltaH:deltaH+H], device=device)  # (C, H, W)
    batch_rotation = BatchRotationOperator(device=device, in_size=H, out_size=H, mode="bicubic", zero_init=True)
    rot = rot.expand(C, -1)  # (C, T)

    def forward_model(x):  # (C, H, W) -> (1, C, T, H, W)
        im = x.unsqueeze(0).expand(T, -1, -1, -1).permute(1, 0, 2, 3)  # (C, T, H, W)
        im = batch_rotation.forward(x=im, rot=rot).permute(1, 0, 2, 3)  # (T, C, H, W)
        im = F.conv2d(im, weight=psf_crop, padding="same", groups=C).permute(1, 0, 2, 3)
        return (im * mask.unsqueeze(1)).unsqueeze(0)

    def to_patches(x):  # (1, C, T, H, W) -> (L, C*T, 64), wavelength-aligned 8x8 patches (PatchesHandler)
        coeff = (lbda / lbda.max()).view(C, 1, 1, 1)
        yyxx = torch.tensor(2 * np.mgrid[:H, :H] / (H - 1) - 1, dtype=torch.float32, device=device)[None].permute(0, 2, 3, 1)
        offset = 1 / (H - 1)
        grid = offset + (yyxx - offset) * coeff
        x = F.grid_sample(x[0], grid=grid, align_corners=True, mode="bicubic", padding_mode="border")
        x = x.permute(0, 1, 3, 2).reshape(1, C * T, H, H)
        return F.unfold(x, kernel_size=8, stride=8).permute(0, 2, 1).reshape(-1, C * T, 64)

    def centered(p):  # remove the per-channel temporal mean (theta_hat = A x + mean_t(y - A x))
        q = p.view(p.shape[0], C, T, p.shape[-1])
        return (q - q.mean(dim=2, keepdim=True)).view_as(p)

    return forward_model, to_patches, centered


def shrinkage_cov_inv(pc):
    """Inverse of the shrinkage covariance of ParamsGaussianHandler; pc: centered patches (L, T, f)."""
    L, T, f = pc.shape
    S = pc.transpose(1, 2) @ pc / T
    tr_S2 = torch.diagonal(S**2, dim1=-2, dim2=-1).sum(dim=-1)
    tr2_S = torch.diagonal(S, dim1=-2, dim2=-1).sum(dim=-1)**2
    tr_SS = torch.diagonal(S @ S, dim1=-2, dim2=-1).sum(dim=-1)
    rho = torch.clip((tr_SS + tr2_S - 2 * tr_S2) / ((T // 2 + 1) * (tr_SS - tr_S2)), 0, 1).view(L, 1, 1)
    C_hat = (1 - rho) * S + rho * torch.diag_embed(torch.diagonal(S, dim1=-2, dim2=-1))
    return torch.linalg.inv(C_hat)


def mahalanobis(pc, C_inv):
    return torch.einsum("ltf,lfg,ltg->", pc, C_inv, pc).item()


@torch.no_grad()
def fixed_metric_terms(path, files, mu_smooth_vals, mu_sparse_vals, device, DE=None, w_ref=None):
    """Data term with fixed metric w_ref (default W_REF) and MSE with fixed W_gt, cached in the results
    folder. DE (2 * divergence) is needed for w_ref = "pilot" (SURE with C^y to find mu_0)."""
    w_ref = W_REF if w_ref is None else w_ref
    cache = path / f"sure_fixedW_{w_ref}.npz"
    if cache.exists():
        c = np.load(cache)
        if list(c["mu_smooth"]) == mu_smooth_vals and list(c["mu_sparse"]) == mu_sparse_vals:
            print(f"Loading cached fixed-W terms from {cache}")
            return c["data_term"], c["mse"], int(c["N"])
    if w_ref == "pilot":
        # 1st pass: SURE with C^y, its minimizer mu_0 gives the pilot reconstruction
        DA_y, _, N = fixed_metric_terms(path, files, mu_smooth_vals, mu_sparse_vals, device, w_ref="y")
        k0, j0 = np.unravel_index(np.argmin(DA_y - N + DE), DA_y.shape)
        f0 = [f for f, ms, mp in files if (ms, mp) == (mu_smooth_vals[k0], mu_sparse_vals[j0])][0]
        print(f"Pilot: mu_0 = ({mu_smooth_vals[k0]:.0e}, {mu_sparse_vals[j0]:.0e})")

    forward_model, to_patches, centered = load_forward_model(device)
    data = np.load(path / "data.npz")
    y = torch.tensor(data["y"], device=device)
    x_gt = torch.tensor(data["x_gt"], device=device)
    Ax_gt = forward_model(x_gt)
    C_inv_gt = shrinkage_cov_inv(centered(to_patches(y - Ax_gt)))
    if w_ref == "gt":
        C_inv = C_inv_gt
    elif w_ref == "pilot":  # covariance of the residual of the pilot reconstruction (disk removed)
        x0 = torch.tensor(np.load(f0)["x"], dtype=torch.float32, device=device)
        C_inv = shrinkage_cov_inv(centered(to_patches(y - forward_model(x0))))
    else:
        C_inv = shrinkage_cov_inv(centered(to_patches(y)))

    DA = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))
    MSE = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))
    for n, (f, ms, mp) in enumerate(files):
        k, j = mu_smooth_vals.index(ms), mu_sparse_vals.index(mp)
        x = torch.tensor(np.load(f)["x"], dtype=torch.float32, device=device)
        Ax = forward_model(x)
        DA[k, j] = mahalanobis(centered(to_patches(y - Ax)), C_inv)
        MSE[k, j] = mahalanobis(centered(to_patches(Ax_gt - Ax)), C_inv_gt)
        print(f"\r fixed-W terms {n+1}/{len(files)}", end="", flush=True)
    print()
    np.savez(cache, data_term=DA, mse=MSE, N=y.numel(), mu_smooth=mu_smooth_vals, mu_sparse=mu_sparse_vals)
    return DA, MSE, y.numel()


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    x_gt = np.load(path / "data.npz")["x_gt"]
    DE = np.zeros((len(mu_smooth_vals), len(mu_sparse_vals)))

    # --- fill the storage array ---
    for f, ms, mp in files:
        k = idx_smooth[ms]   # index for mu_smooth
        j = idx_sparse[mp]   # index for mu_sparse
        data = np.load(f, allow_pickle=True)
        x_disk_store[k, j] = data["x"]
        DE[k, j] = data["div_est"]   # already 2 * divergence

    # --- data term and MSE with a fixed metric, then SURE = data term - N + 2 div ---
    DA, MSE, N = fixed_metric_terms(path, files, mu_smooth_vals, mu_sparse_vals, device, DE=DE)
    SURE = DA - N + DE

    # Tick labels from the actual grid values: rows (axis 0, y) = mu_smooth, columns (axis 1, x) = mu_sparse
    labels_smooth = [f"$10^{{{np.log10(v):g}}}$" for v in mu_smooth_vals]
    labels_sparse = [f"$10^{{{np.log10(v):g}}}$" for v in mu_sparse_vals]

    # Find best parameters
    idx_best = np.unravel_index(np.argmin(MSE), MSE.shape)
    k_mse_best, j_mse_best = idx_best
    best_x_disk_mse = x_disk_store[k_mse_best, j_mse_best]
    idx_best = np.unravel_index(np.argmin(SURE), SURE.shape)
    k_sure_best, j_sure_best = idx_best
    best_x_disk_sure = x_disk_store[k_sure_best, j_sure_best]
    print(f"W_REF={W_REF}: argmin MSE  mu_smooth={mu_smooth_vals[k_mse_best]:.0e} mu_sparse={mu_sparse_vals[j_mse_best]:.0e}")
    print(f"W_REF={W_REF}: argmin SURE mu_smooth={mu_smooth_vals[k_sure_best]:.0e} mu_sparse={mu_sparse_vals[j_sure_best]:.0e}")

    # SURE is defined up to a constant and its variations are small compared to its level:
    # display SURE - min(SURE) (+1 for the log scale)
    SURE_disp = SURE # - SURE.min() + 1

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
    plt.subplot(2,2,3); plt.imshow(SURE_disp, cmap="RdBu_r");  plt.title(r"$\mathrm{SURE}$"); plt.colorbar()
    plt.xticks(np.arange(s2), labels_sparse)
    plt.yticks(np.arange(s1), labels_smooth)
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.plot(j_sure_best, k_sure_best, "rx", markersize=12);
    plt.subplot(2,2,4); plt.imshow(MSE, cmap="RdBu_r");  plt.title(r"$\mathrm{MSE}$"); plt.colorbar()
    plt.xticks(np.arange(s2), labels_sparse)
    plt.yticks(np.arange(s1), labels_smooth)
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.plot(j_mse_best, k_mse_best, "rx", markersize=12);
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.subplot(1,2,1); plt.imshow(DA, cmap="viridis");  plt.title(r"$\mathrm{Data Term}$"); plt.colorbar()
    plt.xticks(np.arange(s2), labels_sparse)
    plt.yticks(np.arange(s1), labels_smooth)
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.subplot(1,2,2); plt.imshow(DE, cmap="viridis");  plt.title(r"$\mathrm{Divergence Estimate}$"); plt.colorbar()
    plt.xticks(np.arange(s2), labels_sparse)
    plt.yticks(np.arange(s1), labels_smooth)
    plt.ylabel(r"$\mu_{\rm smooth}$"); plt.xlabel(r"$\mu_{\rm sparse}$")
    plt.tight_layout()
    plt.show()

    s1, s2 = MSE.shape
    fig = plt.figure(figsize=(5, 8))
    ax1 = fig.add_subplot(2, 1, 1)
    im1 = ax1.imshow(MSE, cmap="RdBu_r")
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
    ax1.set_xticks(np.arange(s2)); ax1.set_xticklabels(labels_sparse, fontsize=12)
    ax1.set_yticks(np.arange(s1)); ax1.set_yticklabels(labels_smooth, fontsize=12)
    ax1.set_ylabel(r"$\mu_{\rm smooth}$", fontsize=14); ax1.set_xlabel(r"$\mu_{\rm sparse}$", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.plot(j_mse_best, k_mse_best, "rx", markersize=12, markeredgewidth=2)
    ax1.plot(j_sure_best, k_sure_best, "m+", markersize=12, markeredgewidth=2)
    ax1.set_aspect('equal', adjustable='box')

    ax2 = fig.add_subplot(2, 1, 2)
    im2 = ax2.imshow(SURE_disp, cmap="RdBu_r")
    ax2.set_title(r"$\mathrm{MC-SURE}$", fontsize=18)
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=1)
    cbar2.ax.tick_params(labelsize=12)
    ax2.set_xticks(np.arange(s2)); ax2.set_xticklabels(labels_sparse, fontsize=12)
    ax2.set_yticks(np.arange(s1)); ax2.set_yticklabels(labels_smooth, fontsize=12)
    ax2.set_ylabel(r"$\mu_{\rm smooth}$", fontsize=14); ax2.set_xlabel(r"$\mu_{\rm sparse}$", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.plot(j_mse_best, k_mse_best, "rx", markersize=12, markeredgewidth=2)
    ax2.plot(j_sure_best, k_sure_best, "m+", markersize=12, markeredgewidth=2)
    ax2.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(f"figures/grids_{shape}_{flux}_fixedC{W_REF}.pdf")
    plt.show()

    if W_REF == "pilot":
        # refined MC-SURE alone (no MSE row), same layout as the bottom panel above, to be placed under
        # the MSE / MC-SURE (C^y) figure in the paper
        fig_r = plt.figure(figsize=(5, 4))
        ax3 = fig_r.add_subplot(1, 1, 1)
        im3 = ax3.imshow(SURE_disp, cmap="RdBu_r")
        ax3.set_title(r"$\mathrm{Refined\ MC-SURE}$", fontsize=18)
        cbar3 = fig_r.colorbar(im3, ax=ax3, shrink=1)
        cbar3.ax.tick_params(labelsize=12)
        ax3.set_xticks(np.arange(s2)); ax3.set_xticklabels(labels_sparse, fontsize=12)
        ax3.set_yticks(np.arange(s1)); ax3.set_yticklabels(labels_smooth, fontsize=12)
        ax3.set_ylabel(r"$\mu_{\rm smooth}$", fontsize=14); ax3.set_xlabel(r"$\mu_{\rm sparse}$", fontsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3.plot(j_mse_best, k_mse_best, "rx", markersize=12, markeredgewidth=2)
        ax3.plot(j_sure_best, k_sure_best, "m+", markersize=12, markeredgewidth=2)
        ax3.set_aspect('equal', adjustable='box')
        fig_r.tight_layout()
        fig_r.savefig(f"figures/grids_{shape}_{flux}_refinedSURE.pdf")
        plt.show()


if __name__ == "__main__":
    main()
