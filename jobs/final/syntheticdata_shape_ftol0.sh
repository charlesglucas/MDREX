#!/bin/bash

# Synthetic data (10 angles x 3 contrasts) with ftol = 0 (xtol = 1e-6) instead of ftol = 1e-8, couple (5e6, 1e5) of Table 2.
# On the convergence test (spiral, angle 0, alpha = 5e-6) ftol changed the PSNR by < 0.05 dB; checks the other contrasts.
# usage: bash syntheticdata_shape_ftol0.sh [shape ...]   (default: spiral)
# output: results/mdrex_results/musmooth5e6_musparse1e5_<shape>_ftol0/x_opt.fits
SHAPES=("${@:-spiral}")
for shape in "${SHAPES[@]}"; do
  oarsub -l "gpu=1,walltime=10:00:00"\
    -p "cluster='vercors9'" \
    -n "synthetic_shape_${shape}_sm5e6_sp1e5_ftol0" \
    "CUDA_VISIBLE_DEVICES=0 \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_shape.py \
      --mu-smooth 5e6 --mu-sparse 1e5 --shape $shape --ftol 0"
done
