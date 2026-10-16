"""Synthetic disks centred on the star, so that the initial parallactic angle only changes the orientation of the disk.

The files data/synthetic_disks/<shape>/hid_fake_disk_image_<shape>_<angle>degrees.fits (1024 x 1024) are the 0 deg
image rotated by -angle about the centre of the 1024 x 1024 image (511.5, 511.5), whereas the circle and the ellipse are
centred at (517, 517) (centre of 180 deg symmetry) and the spiral at about (511.5, 511.5): the rotated circles and
ellipses are displaced (by up to 11 px at 180 deg). Moreover, the star of the nuisance cubes is at pixel (512, 512) of
the 1024 x 1024 frames, i.e. pixel 128 of the 256 x 256 crop of load_folder (also the rotation centre of MD-REX), and
the disks are cropped as data[513-128:513+128, 513-128:513+128] by the scripts, so the star is at pixel (513, 513) of
the disk files.
New files: the 0 deg image is translated so that the centre of the disk is at (513, 513), then rotated by -angle about
(513, 513) (same direction as the original files), in a single cubic-spline interpolation; negative values set to 0
and total flux preserved.
Output: data/synthetic_disks_centered/<shape>/ with the same file names.
Usage: python scripts/run_results/make_centered_disks.py"""
import pathlib
import numpy as np
from astropy.io import fits
from scipy.ndimage import affine_transform

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAR = np.array([513.0, 513.0])                       # star in the disk files (pixel 128 of the crops)
CENTRES = {"medium_ellipse": (517.0, 517.0), "circle": (517.0, 517.0), "spiral": (511.5, 511.5)}
ANGLES = range(0, 325, 36)

for shape, c0 in CENTRES.items():
    src = ROOT / "data/synthetic_disks" / shape
    dst = ROOT / "data/synthetic_disks_centered" / shape
    dst.mkdir(parents=True, exist_ok=True)
    f0 = src / f"hid_fake_disk_image_{shape}_0degrees.fits"
    img0, header = fits.getdata(f0).astype(np.float64), fits.getheader(f0)
    for ang in ANGLES:
        t = np.deg2rad(-ang)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        # output pixel p <- input pixel c0 + R (p - STAR): rotation about STAR of the image translated by STAR - c0
        img = affine_transform(img0, R, offset=np.asarray(c0) - R @ STAR, order=3, mode="constant", cval=0.0)
        img = np.clip(img, 0, None)
        img *= img0.sum() / max(img.sum(), 1e-300)
        h = header.copy()
        h["HISTORY"] = f"centred on the star (513, 513) from the 0 deg image (centre {c0}), rotated by {-ang} deg"
        fits.writeto(dst / f"hid_fake_disk_image_{shape}_{ang}degrees.fits", img.astype(np.float32), h, overwrite=True)
    print(shape, "done", flush=True)
