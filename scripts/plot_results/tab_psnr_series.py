"""LaTeX PSNR table (disk_rec_comp.compute_psnr / print_psnr_table) of an MD-REX series against REXPACO.
  python scripts/plot_results/tab_psnr_series.py <series>
<series>: key of SERIES below; each entry gives, per shape and contrast (1e-6, 5e-6, 1e-5), the result folder
results/mdrex_results/musmooth<ms>_musparse<mp>_<shape>_alpha<flux>_ftol0_<tag> as "<ms>_<mp>_<tag>" (None: missing).
"""
import os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/plot_results"))
import numpy as np
import astropy.io.fits as fits
from scipy.ndimage import correlate
import disk_rec_comp as D

SERIES = {
    "K12fpsnrmax": {"medium_ellipse": ["3e7_1e6_K12fpsnrmax", "1e6_1e6_K12fpsnrmax", "1e6_1e6_K12fpsnrmax"],
                    "spiral": ["3e7_3e5_K12fpsnrmax", "3e6_1e5_K12fpsnrmax", "3e6_1e5_K12fpsnrmax"],
                    "circle": ["3e7_3e5_K12fpsnrmax", "1e7_1e5_K12fpsnrmax", "3e6_3e4_K12fpsnrmax"]},
    "K12ffixedpoint": {"medium_ellipse": ["2.95e7_2.03e6_K12ffixedpoint", None, None],
                       "spiral": ["1.72e8_6.2e5_K12ffixedpoint", "1.5e7_3.34e5_K12ffixedpoint", None],
                       "circle": ["2.33e7_1.44e6_K12ffixedpoint", None, None]},
}
mode = sys.argv[1]
DIRS = SERIES[mode]
shapes = ["medium_ellipse", "spiral", "circle"]; ft = ["1em6", "5em6", "1em5"]
gt = np.zeros((3, 10, 3, 256, 256), np.float32)
for s, sh in enumerate(shapes):
    for a, ang in enumerate(range(0, 325, 36)):
        d = fits.getdata(ROOT / f"data/synthetic_disks/{sh}/hid_fake_disk_image_{sh}_{ang}degrees.fits").astype(np.float32)
        for f, fl in enumerate([1e-6, 5e-6, 1e-5]):
            gt[f, a, s] = fl * d[513-128:513+128, 513-128:513+128]
xr = fits.getdata(ROOT / "results/Rexpaco_results/x_opt.fits")[:, :, :, 251-128:251+128, 251-128:251+128].astype(np.float32)
xm = np.zeros_like(gt)
for s, sh in enumerate(shapes):
    for f, m in enumerate(DIRS[sh]):
        if m is None:
            print(f"% {sh} {ft[f]}: missing"); continue
        sm, sp, tag = m.split("_")
        p = ROOT / f"results/mdrex_results/musmooth{sm}_musparse{sp}_{sh}_alpha{ft[f]}_ftol0_{tag}/x_opt.fits"
        if not p.exists():
            print(f"% {p.parent.name}: MISSING"); continue
        print(f"% {p.parent.name}")
        xm[f, :, s] = fits.getdata(p)[f, :, s]
psf = D.load_psf_synthetic()
def conv_np(x, psf):   # = F.conv2d(x, psf, padding="same") (cross-correlation, zero padding), without torch
    k = psf[0, 0].numpy().astype(np.float64); sh = x.shape; xs = x.reshape(-1, *sh[-2:])
    return np.stack([correlate(im.astype(np.float64), k, mode="constant") for im in xs]).reshape(sh).astype(np.float32)
D.convolve_stack = conv_np
p = D.compute_psnr(gt, xr, xm)
pc = D.compute_psnr(D.convolve_stack(gt, psf), D.convolve_stack(xr, psf), D.convolve_stack(xm, psf))
D.print_psnr_table(p, pc, mode)
