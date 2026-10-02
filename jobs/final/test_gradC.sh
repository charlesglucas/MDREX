#!/bin/bash

# Test of the gradient through the estimated means / covariances (commit e395fb4) on the synthetic data,
# with the default tolerances (ftol 1e-8) to isolate the effect of the gradient.
#   1. Table 2: the 3 shapes, 10 angles x 3 contrasts, (mu_smooth, mu_sparse) = (5e6, 1e5)
#      output: results/mdrex_results/musmooth5e6_musparse1e5_<shape>_gradC/x_opt.fits
#   2. MC-SURE: 4 x 4 sub-grid around the minima, ellipse, alpha = 5e-6
#      output: results/grids_111111111111/grid_sure_medium_ellipse_alpha5em6_gradC/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results

for shape in medium_ellipse spiral circle; do
  oarsub -l "gpu=1,walltime=10:00:00"\
    -p "cluster='vercors14'" \
    -n "gradC_shape_${shape}_sm5e6_sp1e5" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN/syntheticdata_shape.py --mu-smooth 5e6 --mu-sparse 1e5 --shape $shape --tag gradC"
done

for sm in 1e5 1e6 1e7 1e8; do
  for sp in 1e4 1e5 1e6 1e7; do
    oarsub -l "gpu=1,walltime=18:00:00"\
      -p "cluster='vercors14'" \
      -n "gradC_sure_medium_ellipse_alpha5e-6_sm${sm}_sp${sp}" \
      "CUDA_VISIBLE_DEVICES=0 $PY $RUN/syntheticdata_sure_gpu.py --mu-smooth $sm --mu-sparse $sp --flux 5e-6 \
        --shape medium_ellipse --tag gradC"
  done
done
