#!/bin/bash

# One job per (mu_smooth:mu_sparse) couple (add --overwrite to recompute existing result files).
# RY Lup: less sparsity than (1e7,1e6) / (1e6,1e7), which crush the disk
COUPLES=(1e6:1e5 1e6:5e5 1e7:1e5 1e7:5e5 5e7:1e5 5e7:5e5)
DATA='RY_lup/2016-04-16/IRDIS/data/'
BAND='h2_h3'   # RY Lup observed in DB_H23 (lambda = 1.593 / 1.667 um)
DATARES='RY_lup-2016-04-16'
for couple in "${COUPLES[@]}"; do
  sm=${couple%%:*}; sp=${couple##*:}
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='vercors14'" \
    -n "realdata_${DATARES}_sm${sm}_sp${sp}" \
    "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py \
    --data '$DATA' --band '$BAND' --datares '$DATARES' \
    --mu-smooth $sm --mu-sparse $sp"
done
