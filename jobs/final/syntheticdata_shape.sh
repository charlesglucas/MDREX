#!/bin/bash


oarsub -l "gpu=1,walltime=6:00:00"\
  -p "cluster='vercors9'" \
  -n "synthetic_shape" \
  "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_shape.py \
    --mu-smooth 1e6 --mu-sparse 5e5 --shape spiral"
