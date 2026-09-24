#!/bin/bash


oarsub -l "gpu=1,walltime=15:00:00"\
  -p "cluster='vercors14'" \
  -n "synthetic_all" \
  "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_all.py \
    --mu-smooth 1e6 --mu-sparse 1e5"