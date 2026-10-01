#!/bin/bash

# RY Lup with ftol = 0 (xtol = 1e-6), band h2_h3 (lambda = 1.593 / 1.667 um), other couples than (1e7, 1e5):
#   (1e6, 1e7): couple used at the very beginning (Figures/RY_lup_musmooth6_musparse7_RGB.pdf)
#   (1e7, 1e6): more sparsity than (1e7, 1e5)
# output: results/realdata/RY_lup-2016-04-16-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
for couple in 1e6:1e7 1e7:1e6; do
  sm=${couple%%:*}; sp=${couple##*:}
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='vercors14'" \
    -n "realdata_ftol0_RY_lup_sm${sm}_sp${sp}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data 'RY_lup/2016-04-16/IRDIS/data/' --band 'h2_h3' --datares 'RY_lup-2016-04-16' \
      --mu-smooth $sm --mu-sparse $sp --ftol 0"
done
