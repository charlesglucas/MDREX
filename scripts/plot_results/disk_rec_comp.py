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
from scipy.ndimage import affine_transform


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


def derotate(img, angle_deg, center=(126.5, 126.5)):
    """Rotate a disk back by its angle (bicubic), as in disk_rec_distrib.py: the synthetic disks rotate
    around pixel (126.5, 126.5) of the 256x256 crop."""
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    c = np.array(center)
    return affine_transform(img, R, offset=c - R @ c, order=3)


def compute_psnr(x_gt, x_rexpaco, x_mdrex, support_threshold=0.04):
    """PSNR = -20 log10(N-RMSE) on the whole image and on the support, for every (flux, angle, shape).
    Support: x_gt > support_threshold * max(x_gt), a threshold RELATIVE to the maximum of the disk, so that
    the support is the same for the three contrasts (0.04 = former absolute threshold 2e-7 at alpha = 5e-6).
    Returns a dict of (3, 10, 3) arrays."""
    nrmse = {k: np.zeros((3, 10, 3), dtype=np.float32)
             for k in ["rexpaco", "mdrex", "rexpaco_supp", "mdrex_supp"]}
    for s in range(3):
        for a in range(10):
            for f in range(3):
                gt = x_gt[f, a, s]
                mask = gt > support_threshold * gt.max()
                for name, x in [("rexpaco", x_rexpaco[f, a, s]), ("mdrex", x_mdrex[f, a, s])]:
                    nrmse[name][f, a, s] = np.sqrt(np.sum((gt - x)**2) / np.sum(gt**2))
                    nrmse[name + "_supp"][f, a, s] = np.sqrt(np.sum((gt[mask] - x[mask])**2) / np.sum(gt[mask]**2))
    return {k: -20 * np.log10(v) for k, v in nrmse.items()}


def print_psnr_table(psnr, psnr_conv, caption, label="table:PSNR"):
    """LaTeX table: PSNR mean +- std over parallactic angles, whole image and support, computed on the
    reconstructions x (psnr) and on the reconstructions convolved by the PSF (psnr_conv). For each shape:
    REXPACO / MD-REX on x, then on H * x; the best method of each column is in bold."""
    shapes = ["Ellipse", "Spiral", "Circle"]
    contrasts = ["$\\alpha = 1\\cdot 10^{-6}$", "$\\alpha = 5\\cdot 10^{-6}$", "$\\alpha = 1\\cdot10^{-5}$"]
    quantities = [(psnr, "$\\widehat{\\mathbf{x}}$"), (psnr_conv, "$\\mathbf{H} * \\widehat{\\mathbf{x}}$")]

    def cells(p, method, i):
        """REXPACO or MD-REX row: whole image then support, best of the two methods in bold."""
        out = []
        for key in ["", "_supp"]:
            for j in range(3):
                mean = {m: np.round(np.mean(p[m + key], 1)[j, i], 2) for m in ["rexpaco", "mdrex"]}
                std = np.std(p[method + key], 1)[j, i]
                other = "mdrex" if method == "rexpaco" else "rexpaco"
                value = f"{mean[method]:.2f}"
                if mean[method] > mean[other]:
                    value = f"\\mathbf{{{value}}}"
                out.append(f"${value} \\pm {std:.2f}$")
        return " & ".join(out)

    print("\\begin{table*}[h!]")
    print("\\centering")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")
    print("\\begin{tabular}{lllcccccc}")
    print("\\hline")
    print(" & & & \\multicolumn{3}{c}{Whole image} & \\multicolumn{3}{c}{Support} \\\\")
    print(" & & & " + " & ".join(contrasts) + " & " + " & ".join(contrasts) + " \\\\")
    print("\\hline")
    for i, c in enumerate(shapes):
        for k, (p, quantity) in enumerate(quantities):
            first = f"\\multirow{{4}}{{*}}{{{c}}} " if k == 0 else " "
            print(f"{first}& \\multirow{{2}}{{*}}{{{quantity}}} & REXPACO & {cells(p, 'rexpaco', i)} \\\\")
            print(f" & & MD-REX & {cells(p, 'mdrex', i)} \\\\")
            if k == 0:
                print("\\cline{2-9}")
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

    # Hyperparameters (mu_smooth, mu_sparse) adapted to each shape (SURE grids)
    mdrex_dirs = {"medium_ellipse": "musmooth1e6_musparse1e6",
                  "spiral": "musmooth5e6_musparse1e5_spiral",
                  "circle": "musmooth5e6_musparse1e5_circle"}
    x_mdrex = np.zeros_like(x_gt_store)
    for (s, shape) in enumerate(["medium_ellipse", "spiral", "circle"]):
        path = ROOT / f"results/mdrex_results/{mdrex_dirs[shape]}/x_opt.fits"
        with fits.open(path) as hdul:
            x_mdrex[:, :, s] = hdul[0].data[:, :, s]

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
    # PSNR on the disks x and on the disks convolved by the PSF (support: convolved ground truth > threshold)
    psnr = compute_psnr(x_gt_store, x_rexpaco, x_mdrex)
    psnr_conv = compute_psnr(x_gt_conv, x_rexpaco_conv, x_mdrex_conv)
    def mu_latex(mdrex_dir):  # "musmooth1e7_musparse5e5" -> "(10^7, 5 \cdot 10^5)"
        vals = []
        for m, e in re.findall(r"mu(?:smooth|sparse)([0-9]+)e([0-9]+)", mdrex_dir):
            vals.append(f"10^{{{e}}}" if m == "1" else f"{m}\\cdot 10^{{{e}}}")
        return f"({vals[0]}, {vals[1]})"
    names = {"medium_ellipse": "ellipse", "spiral": "spiral", "circle": "circle"}
    mus = [f"$\\boldsymbol{{\\mu}} = {mu_latex(d)}$ for the {names[s]}" for s, d in mdrex_dirs.items()]
    mu_caption = "MD-REX uses " + ", ".join(mus[:-1]) + " and " + mus[-1] + "."
    # Single table: reconstructions x and reconstructions convolved by the PSF
    print_psnr_table(psnr, psnr_conv,
                     "{\\bf Comparison of performances.} PSNR (whole image and support) averaged over parallactic "
                     "angles for MD-REX and REXPACO, computed on the reconstructions $\\widehat{\\mathbf{x}}$ and on "
                     "the reconstructions convolved by the PSF $\\mathbf{H} * \\widehat{\\mathbf{x}}$; the support is the set of "
                     "pixels where the ground truth exceeds 4\\% of its maximum. " + mu_caption
                     + " The best method is in bold.")

    # Same table with a single couple (mu_smooth, mu_sparse) for all the geometries and contrasts, as REXPACO
    fixed_dir = "musmooth1e7_musparse1e5"
    with fits.open(ROOT / f"results/mdrex_results/{fixed_dir}/x_opt.fits") as hdul:
        x_mdrex_fixed = hdul[0].data
    psnr_fixed = compute_psnr(x_gt_store, x_rexpaco, x_mdrex_fixed)
    psnr_conv_fixed = compute_psnr(x_gt_conv, x_rexpaco_conv, convolve_stack(x_mdrex_fixed, psf))
    print_psnr_table(psnr_fixed, psnr_conv_fixed,
                     "{\\bf Comparison of performances with fixed hyperparameters.} Same as "
                     "Table~\\ref{table:PSNR} with the same hyperparameters "
                     f"$\\boldsymbol{{\\mu}} = {mu_latex(fixed_dir)}$ for all the geometries and contrasts. "
                     "The best method is in bold.", label="table:PSNR-fixed")

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

    # COMP_ANGLE = 0: first parallactic angle; COMP_ANGLE = "mean": reconstructions and ground truths derotated
    # by their angle and averaged over the 10 angles (as in disk_rec_distrib.py), files suffixed "_mean"
    COMP_ANGLE = "mean"
    angles = list(range(0, 325, 36))

    def select(arr):  # (flux, angle, shape, H, W) -> (flux, shape, H, W)
        if COMP_ANGLE == "mean":
            return np.mean([np.stack([np.stack([derotate(arr[f, a, s], ang) for s in range(3)]) for f in range(3)])
                            for a, ang in enumerate(angles)], axis=0)
        return arr[:, angles.index(COMP_ANGLE)]
    mean_tag = "_mean" if COMP_ANGLE == "mean" else ""

    # Same comparison on x (tag "") and on x convolved by the PSF (tag "_conv")
    datasets = [
        (select(x_gt_store), select(x_rexpaco), select(x_mdrex), ""),
        (select(x_gt_conv), select(x_rexpaco_conv), select(x_mdrex_conv), "_conv"),
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
                vmin_top = min(rex_all[f,s].min(), md_all[f,s].min())
                im0 = ax_gt.imshow(gt_all[f,s], cmap='hot', vmin=vmin_top, vmax=2*flux)
                ax_gt.set_title("Ground Truth")

                # --- ligne du milieu : plage commune ---
                im1 = axs[0,0].imshow(rex_all[f,s], cmap='hot', vmin=vmin_top, vmax=2*flux)
                im2 = axs[0,1].imshow(md_all[f,s], cmap='hot', vmin=vmin_top, vmax=2*flux)

                # --- ligne du bas : plage symétrique commune ---
                data1 = np.abs(rex_all[f,s] - gt_all[f,s])/flux
                data2 = np.abs(md_all[f,s]   - gt_all[f,s])/flux
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
                    f"figures/comp{tag}_{shape_id}_{suffix}_{mdrex_dirs[shape_id]}{mean_tag}.pdf",
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