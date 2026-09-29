#!/bin/bash

# 10 angles x 3 contrasts for one shape; output: results/mdrex_results/musmooth<mu>_musparse<mu>_<shape>/x_opt.fits
launch() {  # launch <shape> <mu_smooth> <mu_sparse>
  oarsub -l "gpu=1,walltime=6:00:00"\
    -p "cluster='vercors9'" \
    -n "synthetic_shape_$1_sm$2_sp$3" \
    "CUDA_VISIBLE_DEVICES=0 \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_shape.py \
      --mu-smooth $2 --mu-sparse $3 --shape $1"
}

# best compromise over the 3 contrasts on the single-angle grids (results/grid_shape/*_angle0)
launch spiral 5e6 1e5
launch circle 5e6 1e5
