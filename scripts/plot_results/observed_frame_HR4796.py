"""Plot one north-aligned observed frame y_{1,t} of HR 4796A (IRDIS, 2015-02-03).

Only the first spectral channel is used. The frame is rotated by its
parallactic angle as in disk_rec_realdata.py (north_align_disk).
Used as the background image of the N-folded symmetry illustration
(Illustration/EXOMILDpatches.svg, Figure `fig:nfolded_patches` of the paper).

Usage:
    python scripts/plot_results/observed_frame_HR4796.py [--frame 0] [--vmin -2] [--vmax 100]
"""
import argparse
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits

from utils.rotation import BatchRotationOperator


def north_align_disk(disk, rotation_deg):
    channels, height, width = disk.shape
    if height != width:
        raise ValueError(f"Expected square disk images, got {disk.shape}")
    operator = BatchRotationOperator(
        device=torch.device("cpu"),
        in_size=height,
        out_size=height,
        mode="bicubic",
        zero_init=False,
    )
    disk_tensor = torch.from_numpy(disk.astype(np.float32))[None]
    angle = torch.full((1, channels), float(rotation_deg))
    return operator.forward(x=disk_tensor, rot=angle)[0].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=ROOT / "results/realdata/HR_4796-2015-02-03/y.npz")
    parser.add_argument("--path_data", default=ROOT / "data/DISKS_IRDIS_CHARLES/HR_4796/2015-02-03/IRDIS/data")
    parser.add_argument("--frame", type=int, default=0, help="temporal frame index")
    parser.add_argument("--vmin", type=float, default=-2)
    parser.add_argument("--vmax", type=float, default=100)
    parser.add_argument("--cmap", default="hot")
    parser.add_argument("--out", default=ROOT / "figures/HR_4796_y")
    args = parser.parse_args()

    y = np.load(args.data)["y"]  # (1, n_wl, n_frames, H, W)
    frame = y[0, 0, args.frame]  # first spectral channel only

    parallactic_angles = np.asarray(
        fits.getdata(
            pathlib.Path(args.path_data)
            / "ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits"
        ),
        dtype=np.float32,
    ).reshape(-1)
    rotation_deg = -float(parallactic_angles[args.frame])
    frame = north_align_disk(frame[None], rotation_deg)[0]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
              origin="lower", interpolation="nearest")
    ax.axis("off")
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=240, pad_inches=0)
    print(f"saved {args.out}.png/.pdf  (frame {args.frame}, rotation {rotation_deg:.2f} deg)")


if __name__ == "__main__":
    main()
