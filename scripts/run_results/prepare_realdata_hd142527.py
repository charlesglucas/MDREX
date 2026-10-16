"""Convert the HD 142527 (HIP 78092) IRDIS K12 observation of 2016-03-26 to the layout read by inference.load_folder
(as the other targets of data/real_data/DISKS_IRDIS_CHARLES/<target>/<date>/data/):
  *reduced_master_cube*.fits (C, T, 1024, 1024), *psf_master_cube*.fits (C, 64, 64), *rotation*.fits (T,)
All the T = 1408 frames (DIT 0.84 s, two sequences of 704 frames separated by a gap of 8.4 deg) are kept.
Centering: the cube is "half-center" (star between 4 pixels, at (511.5, 511.5)), whereas the IRDIS Data Center cubes
("center_im") and MD-REX (rotation centre at pixel 128 of the central 256 x 256 crop taken by load_folder, i.e. pixel
512 of the 1024 x 1024 frame) put the star at the centre of pixel (512, 512): every frame is shifted by --shift pixels
along both axes by Fourier interpolation (default +0.5).
The 11.8 GB cube is read with memmap and written frame by frame (StreamingHDU), without loading it in memory.
Usage: python prepare_realdata_hd142527.py [--shift 0.5]"""
import argparse
import pathlib
import numpy as np
from astropy.io import fits
from scipy.ndimage import fourier_shift

ROOT = pathlib.Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--shift", type=float, default=0.5, help="shift (pixels, both axes) moving the star to pixel (512, 512)")
args = parser.parse_args()

src = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HD 142527/2016-03-26"
dst = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HD_142527/2016-03-26/data"
dst.mkdir(parents=True, exist_ok=True)
stem = "IRD_HD142527_SPHERE_2016-03-27T07:56:08.5118"

with fits.open(src / f"{stem}_half-center_im.fits", memmap=True) as hdul:
    cube, src_header = hdul[0].data, hdul[0].header
    C, T, H, W = cube.shape
    header = fits.PrimaryHDU(data=np.zeros((1, 1, 1, 1), dtype=">f4")).header
    header["NAXIS1"], header["NAXIS2"], header["NAXIS3"], header["NAXIS4"] = W, H, T, C
    for k in ("HIERARCH ESO INS COMB IFLT", "HIERARCH ESO DET SEQ1 DIT", "HIERARCH ESO DET NDIT", "OBJECT", "DATE-OBS"):
        if k in src_header:
            header[k] = src_header[k]
    header["HISTORY"] = f"MD-REX: all frames, shifted by {args.shift} px (star at pixel 512)"
    out = fits.StreamingHDU(str(dst / f"{stem}-reduced_master_cube.fits"), header)
    for c in range(C):   # FITS order: channel-major, then frames
        for t in range(T):
            frame = np.asarray(cube[c, t], dtype=np.float64)
            shifted = np.fft.ifft2(fourier_shift(np.fft.fft2(frame), (args.shift, args.shift))).real
            out.write(shifted.astype(">f4"))
            if t % 200 == 0:
                print(f"channel {c}, frame {t + 1}/{T}", flush=True)
    out.close()
rot = fits.getdata(src / "rotnth.fits").reshape(-1).astype(np.float32)
assert rot.size == T, (rot.size, T)
fits.writeto(dst / f"{stem}-parallactic_rotation.fits", rot, overwrite=True)
fits.writeto(dst / f"{stem}-psf_master_cube.fits",
             fits.getdata(src / f"{stem}-modified_median_sat_2.fits").astype(np.float32), overwrite=True)
print(f"{dst}: cube ({C}, {T}, {H}, {W}), rotation {rot.min():.2f} .. {rot.max():.2f} deg, "
      f"filter {header.get('HIERARCH ESO INS COMB IFLT')}")
