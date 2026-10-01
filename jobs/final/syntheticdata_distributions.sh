#!/bin/bash

# Fig. 2 (multi-distribution, alpha = 5e-6, 10 angles): output results/distributions_5em6/<shape>_musmooth<mu>_musparse<mu>/x_opt.fits
# done: circle 5e6 1e5, medium_ellipse 1e6 1e6, medium_ellipse 1e7 1e5, circle 1e7 1e5
# (5e6, 1e5): couple of Table 2 (fixed for all the geometries); medium_ellipse launched on 2026-10-01
# usage: bash syntheticdata_distributions.sh [shape ...]   (default: spiral)
SHAPES=("${@:-spiral}")
for shape in "${SHAPES[@]}"; do
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='vercors18'" \
    -n "synthetic_distributions_${shape}_sm5e6_sp1e5" \
    "CUDA_VISIBLE_DEVICES=0 \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_distributions.py \
      --mu-smooth 5e6 --mu-sparse 1e5 --shape $shape"
done
