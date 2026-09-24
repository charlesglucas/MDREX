import sys, pathlib, os
import argparse
import re

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

import torch
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from utils.rotation import BatchRotationOperator

# Reads the per-hyperparameter files written by run_results/realdata_grid_split.py:
#   results/realdata/<result>/musmooth<mu>_musparse<mu>.npz  (e.g. musmooth1e3_musparse5e4.npz)
# and displays the (mu_smooth, mu_sparse) grid of reconstructions.

parser = argparse.ArgumentParser()
parser.add_argument("--result", type=str, default="HD_106906-2016-03-28-k1_k2",
                    help="Result folder name in results/realdata (= --datares of the run)")
parser.add_argument("--data", type=str, default="HD_106906/2016-03-28/IRDIS/k1_k2/data",
                    help="Real data relative folder name (for parallactic angles)")
parser.add_argument("--quantity", type=str, default="x", choices=["x", "Ax"],
                    help="Show reconstructions x or forward model A_0 x")
parser.add_argument("--channel", type=str, default="mean",
                    help="'mean' over channels, or channel index (0, 1)")
parser.add_argument("--crop", action="store_true", help="Zoom on the central half of the image")
parser.add_argument("--no-north", action="store_true", help="Do not north-align the images")
args = parser.parse_args()

display_titles = {
    "HR_4796": "HR 4796A",
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


def load_grid(result_dir):
    pattern = re.compile(r"musmooth(.+?)_musparse(.+)\.npz$")
    entries = {}
    for f in result_dir.glob("musmooth*_musparse*.npz"):
        m = pattern.search(f.name)
        if m:
            entries[(float(m.group(1)), float(m.group(2)))] = f
    if not entries:
        raise FileNotFoundError(f"No result file found in {result_dir}")
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


def main():
    result_dir = ROOT / "results" / "realdata" / args.result
    title = display_titles.get(args.result, args.result.replace("_", " "))
    entries, e_smooth, e_sparse = load_grid(result_dir)
    print(f"Found {len(entries)} files: mu_smooth {[f'{m:g}' for m in e_smooth]}, mu_sparse {[f'{m:g}' for m in e_sparse]}")

    rotation_deg = None
    if not args.no_north:
        path_data = ROOT / "data/real_data/DISKS_IRDIS_CHARLES" / args.data
        parallactic_angles = np.asarray(
            fits.getdata(
                path_data
                / "ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits"
            ),
            dtype=np.float32,
        ).reshape(-1)
        rotation_deg = -float(np.median(parallactic_angles[0]))

    def get_image(f):
        d = np.load(f, allow_pickle=True)
        if args.quantity == "x":
            img = d["x"].astype(np.float32)  # (C, H, W)
        else:
            img = d["Ax"][0, :, 0].astype(np.float32)  # (C, H, W), frame 0
        if rotation_deg is not None:
            img = north_align_disk(img, rotation_deg)
        if args.channel == "mean":
            return np.mean(img, axis=0)
        return img[int(args.channel)]

    n_rows, n_cols = len(e_smooth), len(e_sparse)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.4 * n_rows), squeeze=False)
    for r, sm in enumerate(e_smooth):
        for c, sp in enumerate(e_sparse):
            ax = axs[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if (sm, sp) not in entries:
                ax.set_facecolor("lightgray")
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                continue
            img = get_image(entries[(sm, sp)])
            im = ax.imshow(img, origin="lower", cmap="inferno")
            if args.crop:
                h, w = img.shape
                ax.set_xlim(w / 4, 3 * w / 4)
                ax.set_ylim(h / 4, 3 * h / 4)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(rf"$({tex_mu(sm)},\, {tex_mu(sp)})$", fontsize=10)
            if c == 0:
                ax.set_ylabel(rf"$\mu_{{smooth}}={tex_mu(sm)}$")
            if r == n_rows - 1:
                ax.set_xlabel(rf"$\mu_{{sparse}}={tex_mu(sp)}$")

    label = r"$\widehat{\mathbf{x}}$" if args.quantity == "x" else r"$\mathbf{A}_0\,\widehat{\mathbf{x}}$"
    fig.suptitle(f"{title} – {label}")
    fig.tight_layout()
    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)
    outfile = figdir / f"{args.result}_grid_{args.quantity}_{args.channel}.pdf"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.show()


if __name__ == "__main__":
    main()
