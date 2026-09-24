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

import astropy.io.fits as fits


# ======================================================================
# HELPERS: PSF loading and convolution
# ======================================================================

def load_psf_synthetic():
    """PSF used for the synthetic data (HIP_72192), same preprocessing as syntheticdata_all.py,
    averaged over channels and normalized to unit sum (keeps the flux, so the same colour
    scales / error normalization as for x can be used)."""
    path_folder = ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data"
    psf_file = [f for f in path_folder.glob("*") if "psf_master_cube" in f.name.lower()]
    assert len(psf_file) == 1, f"{psf_file=}"
    psf = torch.tensor(fits.getdata(psf_file[0]).astype(np.float32))
    if psf.ndim == 4:
        psf = psf[:, 0]
    psf_size = psf.shape[-1]
    border = 15
    mask_out = torch.ones_like(psf)[0].bool()
    mask_out[border : psf_size - border, border : psf_size - border] = 0
    for c in range(psf.shape[0]):
        psf[c] = psf[c] - psf[c, mask_out].mean()
    crop_size = 13
    psf_crop = psf[
        :,
        1 + (psf_size - crop_size) // 2 : 1 + (psf_size + crop_size) // 2,
        1 + (psf_size - crop_size) // 2 : 1 + (psf_size + crop_size) // 2,
    ]
    psf_mean = psf_crop.mean(dim=0)
    return (psf_mean / psf_mean.sum()).view(1, 1, crop_size, crop_size)


def convolve_stack(x, psf):
    """x (..., H, W) convolved image by image with psf (1, 1, k, k)."""
    shape = x.shape
    x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(-1, 1, *shape[-2:])
    return F.conv2d(x_t, weight=psf, padding="same").reshape(shape).numpy()


def compute_psnr(x_gt, x_rexpaco, x_mdrex, support_threshold=2e-7):
    """PSNR = -20 log10(N-RMSE) on the whole image and on the support (x_gt > threshold),
    for every (flux, angle, shape). Returns a dict of (3, 10, 3) arrays."""
    nrmse = {k: np.zeros((3, 10, 3), dtype=np.float32)
             for k in ["rexpaco", "mdrex", "rexpaco_supp", "mdrex_supp"]}
    for s in range(3):
        for a in range(10):
            for f in range(3):
                gt = x_gt[f, a, s]
                mask = gt > support_threshold
                for name, x in [("rexpaco", x_rexpaco[f, a, s]), ("mdrex", x_mdrex[f, a, s])]:
                    nrmse[name][f, a, s] = np.sqrt(np.sum((gt - x)**2) / np.sum(gt**2))
                    nrmse[name + "_supp"][f, a, s] = np.sqrt(np.sum((gt[mask] - x[mask])**2) / np.sum(gt[mask]**2))
    return {k: -20 * np.log10(v) for k, v in nrmse.items()}


def print_psnr_table(psnr, caption):
    """LaTeX table: PSNR mean +- std over parallactic angles, whole image and support."""
    mean = {k: np.mean(v, 1) for k, v in psnr.items()}
    std = {k: np.std(v, 1) for k, v in psnr.items()}
    shapes = ["Ellipse", "Spiral", "Circle"]
    contrasts = ["$\\alpha = 1\\cdot 10^{-6}$", "$\\alpha = 5\\cdot 10^{-6}$", "$\\alpha = 1\\cdot10^{-5}$"]

    def cells(k, i):
        return " & ".join([f"${mean[k][j,i]:.2f} \\pm {std[k][j,i]:.2f}$" for j in range(3)])

    print("\\begin{table*}[h!]")
    print("\\centering")
    print(f"\\caption{{{caption}}}")
    print("\\begin{tabular}{llcccccc}")
    print("\\hline")
    print(" & & \\multicolumn{3}{c}{Whole image} & \\multicolumn{3}{c}{Support} \\\\")
    print(" & & " + " & ".join(contrasts) + " & " + " & ".join(contrasts) + " \\\\")
    print("\\hline")
    for i, c in enumerate(shapes):
        print(f"\\multirow{{2}}{{*}}{{{c}}} & REXPACO & {cells('rexpaco', i)} & {cells('rexpaco_supp', i)} \\\\")
        print(f" & MD-REX & {cells('mdrex', i)} & {cells('mdrex_supp', i)} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table*}")
    print()


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):

    # ======================================================================
    # 1. GROUND TRUTH (synthetic disks)
    # ======================================================================
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


    # ======================================================================
    # 2. RECONSTRUCTIONS (REXPACO and MD-REX)
    # ======================================================================
    path = ROOT / "results/Rexpaco_results/x_opt.fits"
    with fits.open(path) as hdul:
        x_rexpaco = hdul[0].data[:,:,:,251-128:251+128, 251-128:251+128]

    mdrex_dir = "musmooth1e7_musparse1e5"
    path = ROOT / f"results/mdrex_results/{mdrex_dir}/x_opt.fits"
    with fits.open(path) as hdul:
        x_mdrex = hdul[0].data
   
    # ======================================================================
    # 3. PSF CONVOLUTION of the ground truth and of the reconstructions
    # ======================================================================
    psf = load_psf_synthetic()
    x_gt_conv = convolve_stack(x_gt_store, psf)
    x_rexpaco_conv = convolve_stack(x_rexpaco, psf)
    x_mdrex_conv = convolve_stack(x_mdrex, psf)

    # ======================================================================
    # 4. PSNR TABLES (LaTeX)
    # ======================================================================
    # Table 1: disks x
    psnr = compute_psnr(x_gt_store, x_rexpaco, x_mdrex)
    print_psnr_table(psnr, "{\\bf Comparison of performances.} PSNR (whole image and support) averaged over "
                           "parallactic angles for MD-REX and REXPACO reconstructions.")

    # Table 2: disks convolved by the PSF (support: convolved ground truth > threshold)
    psnr_conv = compute_psnr(x_gt_conv, x_rexpaco_conv, x_mdrex_conv)
    print_psnr_table(psnr_conv, "{\\bf Comparison of performances on PSF-convolved disks.} PSNR (whole image and "
                                "support) averaged over parallactic angles for MD-REX and REXPACO reconstructions "
                                "convolved by the PSF.")

    # ======================================================================
    # 5. GROUND-TRUTH SHAPES
    # ======================================================================

    plt.subplots(1,3, figsize=(6, 2.7))
    plt.subplot(1,3,1); plt.imshow(x_gt_store[0,0,0], cmap='hot'); plt.title(r"$Ellipse$"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(x_gt_store[0,0,1], cmap='hot');  plt.title(r"Spiral"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(x_gt_store[0,0,2], cmap='hot');  plt.title(r"Circle"); plt.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1)
    plt.savefig("figures/shapes.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    # ======================================================================
    # 6. COMPARISON FIGURES (on x, then on PSF-convolved x)
    # ======================================================================
    shape_ids = ["medium_ellipse", "spiral", "circle"]
    fluxes = [1e-6, 5e-6, 1e-5]
    suffixes = ["1em6", "5em6", "1em5"]

    # Same comparison on x (tag "") and on x convolved by the PSF (tag "_conv")
    datasets = [
        (x_gt_store, x_rexpaco, x_mdrex, ""),
        (x_gt_conv, x_rexpaco_conv, x_mdrex_conv, "_conv"),
    ]
    for (gt_all, rex_all, md_all, tag) in datasets:
        for s, shape_id in enumerate(shape_ids):
            for f, (flux, suffix) in enumerate(zip(fluxes, suffixes)):
                fig = plt.figure(figsize=(6, 6))
                gs = fig.add_gridspec(3, 2, wspace=0.03, hspace=0.25, right=0.84, top=0.92)
                ax_gt = fig.add_subplot(gs[0, :])
                axs = np.array([[fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])],
                                [fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])]])

                # --- ligne du haut : vérité terrain centrée ---
                vmin_top = min(rex_all[f,0,s].min(), md_all[f,0,s].min())
                im0 = ax_gt.imshow(gt_all[f,0,s], cmap='hot', vmin=vmin_top, vmax=2*flux)
                ax_gt.set_title("Ground Truth")

                # --- ligne du milieu : plage commune ---
                im1 = axs[0,0].imshow(rex_all[f,0,s], cmap='hot', vmin=vmin_top, vmax=2*flux)
                im2 = axs[0,1].imshow(md_all[f,0,s], cmap='hot', vmin=vmin_top, vmax=2*flux)

                # --- ligne du bas : plage symétrique commune ---
                data1 = np.abs(rex_all[f,0,s] - gt_all[f,0,s])/flux
                data2 = np.abs(md_all[f,0,s]   - gt_all[f,0,s])/flux
                im3 = axs[1,0].imshow(data1, cmap='gray', vmin=0, vmax=1)
                im4 = axs[1,1].imshow(data2, cmap='gray', vmin=0, vmax=1)

                # --- cosmétique ---
                for ax in [ax_gt, *axs.flat]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_frame_on(True)
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.5)

                axs[0,0].set_title(r"$\mathrm{REXPACO}$")
                axs[0,1].set_title(r"$\mathrm{MD-REX}$")
                axs[0,0].set_ylabel(r"Reconstruction")
                axs[1,0].set_ylabel(r"Relative error")

                # --- colorbar ligne du milieu ---
                cbar1 = fig.colorbar(im1, ax=axs[0,:], location='right', pad=0.05, shrink=1)
                cbar1.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                cbar1.ax.yaxis.set_offset_position('right')
                offset1 = cbar1.ax.yaxis.get_offset_text()
                offset1.set_visible(True)
                offset1.set_fontsize(10)
                offset1.set_horizontalalignment('center')
                offset1.set_verticalalignment('bottom')
                offset1.set_position((0.5, 1.02))

                # --- colorbar ligne du bas ---
                cbar2 = fig.colorbar(im3, ax=axs[1,:], location='right', pad=0.05, shrink=1)
                cbar2.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                cbar2.ax.yaxis.set_offset_position('right')
                cbar2.ax.yaxis.get_offset_text().set_visible(True)
                cbar2.ax.yaxis.get_offset_text().set_fontsize(10)

                # --- colorbar ligne du haut (placée à part pour garder ax_gt centré) ---
                # shift the top image slightly to the left
                gt_pos = ax_gt.get_position()
                ax_gt.set_position([gt_pos.x0 - 0.03, gt_pos.y0, gt_pos.width, gt_pos.height])
                gt_pos = ax_gt.get_position()
                cb_width = 0.046 * gt_pos.width
                cb_pad = 0.13 * gt_pos.width
                cax0 = fig.add_axes([gt_pos.x1 + cb_pad, gt_pos.y0, cb_width, gt_pos.height])
                cbar0 = fig.colorbar(im0, cax=cax0)
                cbar0.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                cbar0.ax.yaxis.set_offset_position('right')
                offset0 = cbar0.ax.yaxis.get_offset_text()
                offset0.set_visible(True)
                offset0.set_fontsize(10)
                offset0.set_horizontalalignment('center')
                offset0.set_verticalalignment('bottom')
                offset0.set_position((0.5, 1.02))

                plt.savefig(
                    f"figures/comp{tag}_{shape_id}_{suffix}_{mdrex_dir}.pdf",
                    dpi=300,
                    bbox_inches='tight',
                    bbox_extra_artists=[cbar0.ax.yaxis.get_offset_text(),
                                        cbar1.ax.yaxis.get_offset_text(),
                                        cbar2.ax.yaxis.get_offset_text()],
                    pad_inches=0.05,
                )
                plt.show()


if __name__ == "__main__":
    main()