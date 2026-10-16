"""Convert the HD 142527 (HIP 78092) IRDIS K12 observation of 2016-03-26 to the layout read by inference.load_folder
(as the other targets of data/real_data/DISKS_IRDIS_CHARLES/<target>/<date>/data/):
  *reduced_master_cube*.fits (C, T, 1024, 1024), *psf_master_cube*.fits (C, 64, 64), *rotation*.fits (T,)
The raw cube has T = 1408 frames of DIT 0.84 s (11.8 GB): consecutive frames are averaged by bins of --bin frames
(parallactic angles averaged too), which keeps the field rotation (65 deg) and reduces the number of frames for MD-REX
(1408 / 8 = 176). The cube is read with memmap, bin by bin.
Usage: python prepare_realdata_hd142527.py [--bin 8]"""
import argparse
import pathlib
import numpy as np
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--bin", type=int, default=8, help="number of consecutive frames averaged")
args = parser.parse_args()

src = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HD 142527/2016-03-26"
dst = ROOT / f"data/real_data/DISKS_IRDIS_CHARLES/HD_142527/2016-03-26-bin{args.bin}/data"
dst.mkdir(parents=True, exist_ok=True)
stem = "IRD_HD142527_SPHERE_2016-03-27T07:56:08.5118"

with fits.open(src / f"{stem}_half-center_im.fits", memmap=True) as hdul:
    cube, header = hdul[0].data, hdul[0].header.copy()
    C, T, H, W = cube.shape
    rot = fits.getdata(src / "rotnth.fits").reshape(-1).astype(np.float64)
    assert rot.size == T, (rot.size, T)
    nb = T // args.bin
    out = np.zeros((C, nb, H, W), dtype=np.float32)
    for b in range(nb):
        out[:, b] = np.asarray(cube[:, b * args.bin:(b + 1) * args.bin], dtype=np.float64).mean(axis=1)
        if b % 20 == 0:
            print(f"bin {b + 1}/{nb}", flush=True)
rot_b = rot[:nb * args.bin].reshape(nb, args.bin).mean(axis=1)
header["HISTORY"] = f"MD-REX: frames averaged by bins of {args.bin} ({T} -> {nb} frames)"
fits.writeto(dst / f"{stem}_bin{args.bin}-reduced_master_cube.fits", out, header, overwrite=True)
fits.writeto(dst / f"{stem}_bin{args.bin}-parallactic_rotation.fits", rot_b.astype(np.float32), overwrite=True)
fits.writeto(dst / f"{stem}-psf_master_cube.fits",
             fits.getdata(src / f"{stem}-modified_median_sat_2.fits").astype(np.float32), overwrite=True)
print(f"{dst}: cube {out.shape}, rotation {rot_b.min():.2f} .. {rot_b.max():.2f} deg, filter {header.get('HIERARCH ESO INS COMB IFLT')}")
