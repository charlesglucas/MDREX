#!/bin/bash

# 3 x 3 couples x 3 contrasts, one angle (~27 reconstructions); relaunching resumes where it stopped
oarsub -l "gpu=1,walltime=10:00:00"\
  -p "cluster='vercors9'" \
  -n "synthetic_shape_grid" \
  "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_shape_grid.py \
    --shape spiral --angle 0 --mu-smooth 1e6 5e6 1e7 --mu-sparse 5e4 1e5 5e5"
