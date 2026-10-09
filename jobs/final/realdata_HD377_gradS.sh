#!/bin/bash

# HD 377, 2016-10-13, IRDIS H23 (lambda = 1.593 / 1.667 um -> coronagraph h2_h3), 80 frames, 35.8 deg of field
# rotation, gradS (gradient through m_hat / C_hat, shrinkage not differentiated), ftol 0. New target: small grid of
# couples, mu_smooth 1e6 / 1e7 x mu_sparse 1e5 / 1e6 / 1e7, one job per mu_smooth (couples in sequence, existing ones
# skipped). output: results/realdata/HD_377-2016-10-13-gradS-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
CLUSTERS=(vercors14 vercors9)
n=0
for sm in 1e6 1e7; do
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=20:00:00" -p "cluster='$CL'" -n "realdata_gradS_HD377_sm${sm}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data 'HD_377/2016-10-13/data/' --band 'h2_h3' --datares 'HD_377-2016-10-13-gradS' \
      --mu-smooth $sm --mu-sparse 1e5 1e6 1e7 --ftol 0"
done
