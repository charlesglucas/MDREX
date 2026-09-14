#!/bin/bash


oarsub -l "gpu=1,walltime=15:00:00"\
  -p "cluster='vercors18'" \
  -n "realdata" \
  "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid.py \
    --data 'RX_J161533255/2019-05-18/IRDIS/data/' \
    --band 'h2_h3' \
    --datares 'RX_J161533255' "

