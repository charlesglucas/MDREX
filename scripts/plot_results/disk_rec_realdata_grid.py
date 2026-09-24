import sys, pathlib, os
import argparse
import re

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits
from utils.rotation import BatchRotationOperator

# Reads the per-hyperparameter files written by run_results/realdata_grid_split.py:
#   results/realdata/<result>/musmooth<mu>_musparse<mu>.npz  (e.g. musmooth1e3_musparse5e4.npz)
#
# Figure 1: all the (mu_smooth, mu_sparse) results available for one star, to choose the couple.
# Figure 2: the couple chosen by hand (MU_SMOOTH, MU_SPARSE), x and x convolved by the PSF, in RGB.

# ------------------------------------------------------------
# Settings (can also be overridden from the command line)
# ------------------------------------------------------------
RESULT = "HD_169142-2019-05-19"            # folder in results/realdata (= --datares of the run)
DATA = "HD_169142/2019-05-19/IRDIS/data"    # real data folder (parallactic angles, wavelengths)
MU_SMOOTH = None                          # chosen couple, e.g. 1e6 (None: only figure 1)
MU_SPARSE = None                          # chosen couple, e.g. 5e6
CROP = False                              # zoom on the central half of the image

parser = argparse.ArgumentParser()
parser.add_argument("--result", type=str, default=RESULT)
parser.add_argument("--data", type=str, default=DATA)
parser.add_argument("--mu-smooth", type=float, default=MU_SMOOTH)
parser.add_argument("--mu-sparse", type=float, default=MU_SPARSE)
parser.add_argument("--crop", action="store_true", default=CROP)
parser.add_argument("--no-north", action="store_true", help="Do not north-align the images")
args, _ = parser.parse_known_args()

display_titles = {
    "HR_4796": "HR 4796A",
    "HR_4796A-2015-02-03": "HR 4796A",
    "RY_lup": "RY Lupi",
    "PDS_70": "PDS 70",
    "HD_169142": "HD 169142",
    "SAO_206462": "SAO 206462",
    "RX_J161533255": "RX J1615.3-3255",
    "HD_106906-2016-03-28-h2_h3": "HD 106906",
    "HD_106906-2016-03-28-k1_k2": "HD 106906",
}


def north_align_disk(disk, rotation_deg):
    channels, height, width = disk.shape
    if height != width:
        raise ValueError(f"Expected square disk images, got {disk.shape}")
    operator = BatchRotationOperator(
        device=torch.device("cpu"),
        in_size=height,
        out_size=height,
        mode="bicubic",
        # x is a static reconstruction, so apply the absolute reference
        # parallactic angle here; zero_init belongs to the frame sequence.
        zero_init=False,
    )
    disk_tensor = torch.from_numpy(disk.astype(np.float32))[None]
    angle = torch.full((1, channels), float(rotation_deg))
    aligned = operator.forward(x=disk_tensor, rot=angle)[0].numpy()
    return aligned


def available_results():
    results_root = ROOT / "results" / "realdata"
    if not results_root.exists():
        return []
    return sorted(
        d.name for d in results_root.iterdir()
        if d.is_dir() and any(d.glob("musmooth*_musparse*.npz"))
    )


def load_grid(result_dir):
    pattern = re.compile(r"musmooth(.+?)_musparse(.+)\.npz$")
    entries = {}
    for f in result_dir.glob("musmooth*_musparse*.npz"):
        m = pattern.search(f.name)
        if m:
            entries[(float(m.group(1)), float(m.group(2)))] = f
    if not entries:
        raise FileNotFoundError(
            f"No result file found in {result_dir}\n"
            f"Available --result values: {available_results()}"
        )
    mu_smooth = sorted({k[0] for k in entries})
    mu_sparse = sorted({k[1] for k in entries})
    return entries, mu_smooth, mu_sparse


def tex_mu(mu):
    """1e3 -> '10^{3}', 5e4 -> '5\\cdot 10^{4}'."""
    mant, exp = f"{mu:e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    if mant == "1":
        return rf"10^{{{int(exp)}}}"
    return rf"{mant}\cdot 10^{{{int(exp)}}}"


def fmt_mu(mu):
    """1e6 -> '1e6', 5e6 -> '5e6' (same format as the result file names)."""
    mant, exp = f"{mu:e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


def get_title(result):
    title = display_titles.get(result)
    if title is None:
        # fall back on the star name (part before the date), e.g. HD_169142-2019-05-19 -> HD_169142
        matches = [v for k, v in display_titles.items() if result.startswith(k.split("-")[0])]
        title = matches[0] if matches else result.replace("_", " ")
    return title


def data_root():
    # cluster: data/real_data/DISKS_IRDIS_CHARLES, local copy: data/DISKS_IRDIS_CHARLES
    for d in (ROOT / "data/real_data/DISKS_IRDIS_CHARLES", ROOT / "data/DISKS_IRDIS_CHARLES"):
        if d.exists():
            return d
    return ROOT / "data/real_data/DISKS_IRDIS_CHARLES"


def get_rotation(data):
    if args.no_north:
        return None
    if data is None:
        print("Warning: no --data given, images are NOT north-aligned")
        return None
    path_rot = (data_root() / data
                / "ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits")
    if not path_rot.exists():
        print(f"Warning: {path_rot} not found, images are NOT north-aligned "
              f"(check --data, or use --no-north)")
        return None
    parallactic_angles = np.asarray(fits.getdata(path_rot), dtype=np.float32).reshape(-1)
    return -float(parallactic_angles[0])


def get_wavelengths(data):
    if data is not None:
        lambda_files = list((data_root() / data).glob("**/*-lam.fits"))
        if len(lambda_files) == 1:
            return np.asarray(fits.getdata(lambda_files[0])).reshape(-1)
    print("Warning: '*-lam.fits' not found, colorbars labelled by channel index")
    return None


def get_psf(data):
    """PSF (C, 1, k, k) with the same preprocessing as in run_results/realdata_grid*.py."""
    if data is None:
        return None
    psf_files = [f for f in (data_root() / data).glob("*") if "psf_master_cube" in f.name.lower()]
    if len(psf_files) != 1:
        print(f"Warning: PSF ('*psf_master_cube*') not found in {data_root() / data}")
        return None
    psf = fits.getdata(psf_files[0]).astype(np.float32)
    if psf.ndim == 4:
        psf = psf[:, 0]
    psf = torch.tensor(psf)
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
    return psf_crop.reshape(psf.shape[0], 1, crop_size, crop_size)


def convolve(x, psf):
    """x (C, H, W) convolved channel by channel with psf (C, 1, k, k)."""
    x_t = torch.from_numpy(x)[None]
    return F.conv2d(x_t, weight=psf, padding="same", groups=x.shape[0])[0].numpy()


def load_images(f, rotation_deg, psf=None):
    """Returns x (C, H, W) and x convolved by the PSF (C, H, W, None if no PSF),
    north-aligned if possible (convolution done before the rotation, in the detector frame)."""
    d = np.load(f, allow_pickle=True)
    x = d["x"].astype(np.float32)
    x_conv = convolve(x, psf) if psf is not None else None
    if rotation_deg is not None:
        x = north_align_disk(x, rotation_deg)
        if x_conv is not None:
            x_conv = north_align_disk(x_conv, rotation_deg)
    return x, x_conv


def set_crop(ax, shape):
    if args.crop:
        h, w = shape[:2]
        ax.set_xlim(w / 4, 3 * w / 4)
        ax.set_ylim(h / 4, 3 * h / 4)


# ------------------------------------------------------------
# Figure 1: all available couples for one star
# ------------------------------------------------------------
def plot_grid(result, entries, e_smooth, e_sparse, rotation_deg, title):
    n_rows, n_cols = len(e_smooth), len(e_sparse)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.6 * n_rows), squeeze=False)
    for r, sm in enumerate(e_smooth):
        for c, sp in enumerate(e_sparse):
            ax = axs[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(rf"$\mu_{{smooth}}={tex_mu(sm)}$")
            if r == n_rows - 1:
                ax.set_xlabel(rf"$\mu_{{sparse}}={tex_mu(sp)}$")
            if (sm, sp) not in entries:
                ax.set_facecolor("lightgray")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            x, _ = load_images(entries[(sm, sp)], rotation_deg)
            img = np.mean(x, axis=0)
            im = ax.imshow(img, origin="lower", cmap="inferno")
            set_crop(ax, img.shape)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(rf"$({tex_mu(sm)},\, {tex_mu(sp)})$", fontsize=10)
    fig.suptitle(rf"{title} – $\widehat{{\mathbf{{x}}}}$ (mean over channels)")
    fig.tight_layout()
    outfile = ROOT / "figures" / f"{result}_grid.pdf"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved {outfile}")
    return fig


# ------------------------------------------------------------
# Figure 2: chosen couple, x and PSF-convolved x in RGB (as in disk_rec_realdata.py)
# ------------------------------------------------------------
def to_rgb(data1, data2):
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())
    d1 = (data1 - vmin) / (vmax - vmin)
    d2 = (data2 - vmin) / (vmax - vmin)
    color_map = np.array([[0.0, 0.5, 1.0],
                          [1.0, 0.5, 0.0]], dtype=np.float32)
    rgb = d1[..., None] * color_map[0] + d2[..., None] * color_map[1]
    return np.clip(rgb, 0.0, 1.0), Normalize(vmin=vmin, vmax=vmax)


def plot_couple(result, f, mu_smooth, mu_sparse, rotation_deg, wavelengths, title, psf):
    x, x_conv = load_images(f, rotation_deg, psf)
    if wavelengths is not None:
        labels = [rf"$\lambda = {wavelengths[i]:.3f}\,\mu\mathrm{{m}}$" for i in range(2)]
    else:
        labels = ["channel 1", "channel 2"]
    blue_map = LinearSegmentedColormap.from_list("black_blue", ["black", "blue"])
    orange_map = LinearSegmentedColormap.from_list("black_orange", ["black", "orange"])

    rows = [(x, r"$\widehat{\mathbf{x}}_{\lambda}$")]
    if x_conv is not None:
        rows.append((x_conv, r"$\mathbf{h}_{\lambda} * \widehat{\mathbf{x}}_{\lambda}$"))
    fig = plt.figure(figsize=(4.8, 4.5 * len(rows)))
    fig.subplots_adjust(left=0.12, right=0.82, bottom=0.12, top=0.92, hspace=0.23)
    grid = fig.add_gridspec(len(rows), 1)
    for row, (img, annotation) in enumerate(rows):
        rgb, norm = to_rgb(img[0], img[1])
        ax = fig.add_subplot(grid[row, 0])
        ax.imshow(rgb, origin="lower")
        set_crop(ax, rgb.shape)
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
                ha="left", va="top", color="white", fontsize=18)
        ax.axis("off")
        if row == 0:
            ax.set_title(title, fontsize=18, pad=10)
        divider = make_axes_locatable(ax)
        blue_cax = divider.append_axes("bottom", size="4%", pad=0.10)
        orange_cax = divider.append_axes("right", size="4%", pad=0.10)
        for cmap, cax, orientation, label in [
            (blue_map, blue_cax, "horizontal", labels[0]),
            (orange_map, orange_cax, "vertical", labels[1]),
        ]:
            cbar = fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation=orientation)
            cbar.formatter = ticker.ScalarFormatter(useMathText=True)
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label(label, fontsize=13)
    fig.suptitle(rf"$\mu_{{smooth}}={tex_mu(mu_smooth)},\ \mu_{{sparse}}={tex_mu(mu_sparse)}$", fontsize=12, y=0.99)
    outfile = ROOT / "figures" / f"{result}_musmooth{fmt_mu(mu_smooth)}_musparse{fmt_mu(mu_sparse)}_RGB.pdf"
    fig.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {outfile}")
    return fig


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    result = args.result
    title = get_title(result)
    entries, e_smooth, e_sparse = load_grid(ROOT / "results" / "realdata" / result)
    print(f"{result}: {len(entries)} files, mu_smooth {[f'{m:g}' for m in e_smooth]}, "
          f"mu_sparse {[f'{m:g}' for m in e_sparse]}")
    # Angle / wavelengths saved with the results (recent runs), else read from the raw data
    saved = np.load(next(iter(entries.values())), allow_pickle=True)
    if "rotation_deg" in saved.files and np.isfinite(saved["rotation_deg"]) and not args.no_north:
        rotation_deg = float(saved["rotation_deg"])
        print(f"Using saved rotation_deg = {rotation_deg:.3f}")
    else:
        rotation_deg = get_rotation(args.data)
    saved_wavelengths = saved["wavelengths"] if "wavelengths" in saved.files else None

    # Figure 1: choose the couple
    plot_grid(result, entries, e_smooth, e_sparse, rotation_deg, title)

    # Figure 2: couple chosen by hand
    if args.mu_smooth is not None and args.mu_sparse is not None:
        key = (args.mu_smooth, args.mu_sparse)
        if key not in entries:
            raise KeyError(
                f"No result for mu_smooth={args.mu_smooth:g}, mu_sparse={args.mu_sparse:g}. "
                f"Available couples: {[(f'{a:g}', f'{b:g}') for a, b in sorted(entries)]}"
            )
        wavelengths = saved_wavelengths if saved_wavelengths is not None else get_wavelengths(args.data)
        psf = get_psf(args.data)
        plot_couple(result, entries[key], *key, rotation_deg, wavelengths, title, psf)
    else:
        print("Set MU_SMOOTH / MU_SPARSE (or --mu-smooth / --mu-sparse) to plot the chosen couple")

    plt.show()


if __name__ == "__main__":
    main()
