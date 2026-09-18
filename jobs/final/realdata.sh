#!/bin/bash


oarsub -l "gpu=1,walltime=15:00:00"\
  -p "cluster='vercors14'" \
  -n "realdata" \
  "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid.py \
    --data 'HD_106906/2016-03-28/IRDIS/k1_k2/data/' \
    --band 'k1_k2' \
    --datares 'HD_106906-2016-03-28-k1_k2' "

