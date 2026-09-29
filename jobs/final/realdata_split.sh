#!/bin/bash

# One job per (mu_smooth:mu_sparse) couple. --overwrite recomputes existing result files.
COUPLES=(1e7:1e6 1e6:1e7)
DATA='RY_lup/2016-04-16/IRDIS/data/'
BAND='h2_h3'   # RY Lup observed in DB_H23 (lambda = 1.593 / 1.667 um)
DATARES='RY_lup-2016-04-16'
for couple in "${COUPLES[@]}"; do
  sm=${couple%%:*}; sp=${couple##*:}
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='vercors9'" \
    -n "realdata_${DATARES}_sm${sm}_sp${sp}" \
    "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py \
    --data '$DATA' --band '$BAND' --datares '$DATARES' \
    --mu-smooth $sm --mu-sparse $sp --overwrite"
done
