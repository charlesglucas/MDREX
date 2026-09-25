"""High-resolution square panels of PDS 70 for the Genoa Science Festival 2026
Scientific Image Exhibition (printed 50 x 50 cm).

- star_observation.png/.pdf: one north-aligned observed frame y (measurement)
- circumstellar_disk_reconstruction.png/.pdf: MD-REX reconstruction x (result)

Same north alignment and blue/orange two-channel rendering as
disk_rec_realdata_x_y.py, without colorbars, with a physical scale bar.

Usage:
    python scripts/plot_results/genoa_exhibition.py [--out_dir ...] [--dpi 300]
"""
import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
ROOT = pathlib.Path(__file__).resolve().parents[2]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import PowerNorm

from disk_rec_realdata_x_y import north_align_disk

PIXEL_SCALE_ARCSEC = 0.01227  # IRDIS plate scale (PIXSCAL in the FITS headers)
DISTANCE_PC = 112.4           # PDS 70 (Gaia DR3)
AU_M = 1.496e11
PANEL_CM = 50


def rgb_composite(img, vmin, vmax):
    vmax = img.max() if vmax is None else vmax
    norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    d = np.clip(norm(np.clip(img, vmin, None)), 0.0, 1.0)  # (2, H, W)
    color_map = np.array([[0.0, 0.5, 1.0],
                          [1.0, 0.5, 0.0]], dtype=np.float32)
    return np.clip(d[0][..., None] * color_map[0] + d[1][..., None] * color_map[1], 0.0, 1.0)


def save_panel(rgb, path, dpi, bar_au):
    size_in = PANEL_CM / 2.54
    fig = plt.figure(figsize=(size_in, size_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin="lower", interpolation="lanczos")
    ax.axis("off")

    # scale bar in the lower-left corner
    n = rgb.shape[0]
    au_per_pixel = PIXEL_SCALE_ARCSEC * DISTANCE_PC
    bar_px = bar_au / au_per_pixel
    x0, y0 = 0.06 * n, 0.07 * n
    ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=size_in * 0.8,
            solid_capstyle="butt")
    exponent = int(np.floor(np.log10(bar_au * AU_M)))
    mantissa = bar_au * AU_M / 10 ** exponent
    ax.text(x0 + bar_px / 2, y0 + 0.025 * n, rf"${mantissa:.1f}\times10^{{{exponent}}}$ m",
            color="white", ha="center", va="bottom", fontsize=size_in * 3.2)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=dpi)
        print(f"saved {path.with_suffix(f'.{ext}')}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=ROOT / "results/realdata_grids/PDS_70.npz")
    parser.add_argument("--path_data", default=ROOT / "data/DISKS_IRDIS_CHARLES/PDS_70/2018-02-24/IRDIS/data")
    parser.add_argument("--out_dir", default=ROOT.parent / "Genoa_Science_Festival_2026/Lucas")
    parser.add_argument("--k", type=int, default=3, help="mu_smooth index: 10^(n_smooth + k)")
    parser.add_argument("--j", type=int, default=4, help="mu_sparse index: 10^(n_sparse + j)")
    parser.add_argument("--frame", type=int, default=0, help="frame of y to display")
    parser.add_argument("--crop", type=int, default=192, help="side of the central crop in pixels")
    parser.add_argument("--y_vmax", type=float, default=None)
    parser.add_argument("--bar_au", type=float, default=100.0, help="scale bar length in au")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    data = np.load(args.results, allow_pickle=True)
    y = data["y"]  # (1, 2, T, H, W)
    x = data["x"][args.k, args.j].astype(np.float32)  # (2, H, W)
    parallactic_angles = np.asarray(
        fits.getdata(pathlib.Path(args.path_data)
                     / "ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits"),
        dtype=np.float32,
    ).reshape(-1)

    x = north_align_disk(x, -float(parallactic_angles[0]))
    y_frame = north_align_disk(y[0, :, args.frame], -float(parallactic_angles[args.frame]))

    # central crop hides the rotated image corners
    n = x.shape[-1]
    lo = (n - args.crop) // 2
    crop = slice(lo, lo + args.crop)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, img, vmax in [
        ("star_observation", y_frame, args.y_vmax),
        ("circumstellar_disk_reconstruction", x, None),
    ]:
        rgb = rgb_composite(img[:, crop, crop], 0.0, vmax)
        save_panel(rgb, out_dir / name, args.dpi, args.bar_au)

    fov_au = args.crop * PIXEL_SCALE_ARCSEC * DISTANCE_PC
    print(f"field of view: {fov_au:.0f} au = {fov_au * AU_M:.2e} m")


if __name__ == "__main__":
    main()
