"""PSNR of the PCA ADI / ASDI images on the ground truth convolved by the PSF (rows "PCA ASDI" of the H_lambda x_lambda
blocks of Table 1), with the conventions of disk_rec_comp.py (normalized channel-mean PSF, support: convolved ground
truth > 4% of its maximum, mean +- std over the ten orientations).

Input: selected_pca_<mode>_detailed_alpha_unc_aligned.fits, array (geometry, flux, orientation, channel, 1448, 1448) in
ADU, geometries (medium_ellipse, spiral, circle), fluxes (1e-6, 5e-6, 1e-5), orientations 0, 36, ..., 324 deg aligned on
the orientation 0 (ROTREF = 0). The PCA images are:
  - converted to contrast by the flux of the off-axis PSF used for the injection (13 x 13 crop, background subtracted,
    as in the forward model of MD-REX), channel by channel, then averaged over the two channels, without throughput
    correction;
  - cropped to the 256 x 256 frame of the reconstructions (the 1024 x 1024 ground truth is centered in the 1448 x 1448
    frame: offset 212 pixels);
  - compared with the ground truth at orientation 0, convolved by the PSF (the PCA images being at the resolution of the
    telescope, they are not convolved again).

    python scripts/plot_results/pca_psnr.py [--pca-dir <folder of the fits files>] [--modes asdi adi]
"""
import argparse
import pathlib
import sys

import numpy as np
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/plot_results"))
import disk_rec_comp as R   # noqa: E402  (load_psf_synthetic, convolve_stack, compute_psnr of Table 1)

SHAPES = ["medium_ellipse", "spiral", "circle"]
FLUXES = [1e-6, 5e-6, 1e-5]
OFFSET = (1448 - 1024) // 2


def psf_flux():
    """Flux (ADU per unit contrast) of the PSF of each channel used to inject the disks in the synthetic data."""
    raw = fits.getdata(ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data/"
                       "ird_sortframes_dc-IRD_SCIENCE_PSF_MASTER_CUBE-median_unsat.fits").astype(np.float32)
    n, border, cs = raw.shape[-1], 15, 13
    out = np.ones((n, n), bool)
    out[border:n - border, border:n - border] = 0
    raw = np.array([p - p[out].mean() for p in raw])
    return raw[:, 1 + (n - cs) // 2:1 + (n + cs) // 2, 1 + (n - cs) // 2:1 + (n + cs) // 2].sum((1, 2))


def ground_truth_conv():
    """Ground truth at orientation 0 convolved by the PSF, (flux, orientation, shape, 256, 256) (same for all the
    orientations, the PCA images being aligned)."""
    gt = np.zeros((3, 10, 3, 256, 256), np.float32)
    for s, shape in enumerate(SHAPES):
        d = fits.getdata(ROOT / f"data/synthetic_disks/{shape}/hid_fake_disk_image_{shape}_0degrees.fits").astype(np.float32)
        for f, flux in enumerate(FLUXES):
            gt[f, :, s] = flux * d[513 - 128:513 + 128, 513 - 128:513 + 128]
    return R.convolve_stack(gt, R.load_psf_synthetic())


def pca_images(path, psum):
    """PCA images in contrast, averaged over the channels, (flux, orientation, shape, 256, 256)."""
    cube = fits.open(path, memmap=True)[0].data
    c0 = OFFSET + 513 - 128
    x = np.zeros((3, 10, 3, 256, 256), np.float32)
    for s in range(3):
        for f in range(3):
            for a in range(10):
                im = np.array(cube[s, f, a, :, c0:c0 + 256, c0:c0 + 256], dtype=np.float64)
                x[f, a, s] = np.mean(im / psum[:, None, None], 0)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-dir", type=pathlib.Path, default=ROOT.parent / "code-astro")
    parser.add_argument("--modes", nargs="+", default=["asdi", "adi"])
    args = parser.parse_args()
    psum = psf_flux()
    gt_conv = ground_truth_conv()
    for mode in args.modes:
        x = pca_images(args.pca_dir / f"selected_pca_{mode}_detailed_alpha_unc_aligned.fits", psum)
        p = R.compute_psnr(gt_conv, x, x)   # the two method slots hold the same images
        print(f"\n% PCA {mode.upper()}, H x: whole image then support, alpha = 1e-6, 5e-6, 1e-5")
        for s, shape in enumerate(SHAPES):
            cells = " & ".join(f"\\modif{{${np.mean(p[k][j, :, s]):.2f} \\pm {np.std(p[k][j, :, s]):.2f}$}}"
                               for k in ["rexpaco", "rexpaco_supp"] for j in range(3))
            print(f"% {shape}\n & & \\modif{{PCA {mode.upper()}}} & {cells} \\\\")


if __name__ == "__main__":
    main()
