#!/bin/bash
# Table 2: MD-REX at the couple (mu_smooth, mu_sparse) maximizing the PSNR (whole image, x_hat) at angle 0 on the
# half-decade grid (grids rhofixcpu, figures/grid_data_<shape>_<flux>_angle0_rhofixcpu.npz), reused for the 10 angles;
# fixed shrinkage coefficients, ftol 0 / xtol 1e-10:
#   results/mdrex_results/musmooth<..>_musparse<..>_<shape>_alpha<..>_ftol0_psnrmax/x_opt.fits
# Usage: bash table2_psnrmax.sh
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P="cluster IN ('vercors10','vercors11','vercors12','vercors13','vercors18')"
while read shape flux sm sp; do
  tag=$(echo $flux | sed 's/e-/em/')
  oarsub -l "gpu=1,walltime=24:00:00" -p "$P" -n "t2psnr_${shape}_${tag}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_shape.py --mu-smooth $sm --mu-sparse $sp \
     --shape $shape --flux $flux --ftol 0 --xtol 1e-10 --tag psnrmax" | grep -o "OAR_JOB_ID=[0-9]*" | sed "s/^/$shape $flux: /"
done <<LIST
medium_ellipse 1e-6 1e7 1e6
medium_ellipse 5e-6 1e6 1e6
medium_ellipse 1e-5 3e6 1e6
spiral 1e-6 3e7 3e5
spiral 5e-6 3e6 1e5
spiral 1e-5 3e6 1e5
circle 1e-6 3e7 3e5
circle 5e-6 3e6 1e5
circle 1e-5 3e6 3e4
LIST
