#!/bin/bash

# RY Lup: effect of the batch size of the ExoMILD engine. Same run as results/realdata/RY_lup-2016-04-16-ftol0/
# musmooth1e7_musparse1e6.npz (band h2_h3, ftol 0, batch 128) with batch_size = None (no split, as in the old scripts).
# output: results/realdata/RY_lup-2016-04-16-ftol0-bsNone/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
oarsub -l "gpu=1,walltime=15:00:00"\
  -p "cluster='vercors14'" \
  -n "realdata_RY_lup_sm1e7_sp1e6_ftol0_bsNone" \
  "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
    --data 'RY_lup/2016-04-16/IRDIS/data/' --band 'h2_h3' --datares 'RY_lup-2016-04-16' \
    --mu-smooth 1e7 --mu-sparse 1e6 --ftol 0 --batch-size none"
