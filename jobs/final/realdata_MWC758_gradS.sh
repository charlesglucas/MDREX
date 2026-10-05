#!/bin/bash

# MWC 758, 2018-12-17, IRDIS H23 (lambda = 1.593 / 1.667 um -> coronagraph h2_h3), 64 frames, 29.1 deg of field
# rotation (largest rotation of the available epochs), gradS (gradient through m_hat / C_hat, shrinkage not
# differentiated), ftol 0. New target: small grid of couples, mu_smooth 1e6 / 1e7 x mu_sparse 1e5 / 1e6 / 1e7,
# one job per mu_smooth (couples in sequence, existing ones skipped).
# output: results/realdata/MWC_758-2018-12-17-gradS-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
CLUSTERS=(vercors14 vercors9)
n=0
for sm in 1e6 1e7; do
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=20:00:00" -p "cluster='$CL'" -n "realdata_gradS_MWC758_sm${sm}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data 'MWC_758/2018-12-17/IRDIS/data/' --band 'h2_h3' --datares 'MWC_758-2018-12-17-gradS' \
      --mu-smooth $sm --mu-sparse 1e5 1e6 1e7 --ftol 0"
done
