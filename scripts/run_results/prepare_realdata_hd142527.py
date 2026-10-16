"""Expose the HD 142527 (HIP 78092) IRDIS K12 observation of 2016-03-26 under the layout read by inference.load_folder
(as the other targets of data/real_data/DISKS_IRDIS_CHARLES/<target>/<date>/data/), without modifying the data:
symbolic links to the original files, with names matched by load_folder
  *reduced_master_cube*.fits -> ..._half-center_im.fits (2, 1408, 1024, 1024), unchanged (1024 x 1024 frames, all the
                                1408 frames, original centring)
  *psf_master_cube*.fits     -> ...-modified_median_sat_2.fits (2, 64, 64)
  *rotation*.fits            -> rotnth.fits (1, 1408)
Usage: python prepare_realdata_hd142527.py"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
src = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HD 142527/2016-03-26"
dst = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HD_142527/2016-03-26/data"
dst.mkdir(parents=True, exist_ok=True)
stem = "IRD_HD142527_SPHERE_2016-03-27T07:56:08.5118"
links = {f"{stem}-reduced_master_cube.fits": f"{stem}_half-center_im.fits",
         f"{stem}-psf_master_cube.fits": f"{stem}-modified_median_sat_2.fits",
         f"{stem}-parallactic_rotation.fits": "rotnth.fits"}
for f in dst.iterdir():          # files of a previous preparation (copies, links, partial cube)
    f.unlink()
for name, target in links.items():
    (dst / name).symlink_to(src / target)
    print(dst / name, "->", src / target)
