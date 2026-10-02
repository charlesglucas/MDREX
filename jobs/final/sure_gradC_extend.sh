#!/bin/bash

# Gradient through m_hat / C_hat: the MSE / MC-SURE minimum of the 4 x 4 test sub-grid (ellipse, alpha = 5e-6) is at
# (mu_smooth, mu_sparse) = (1e6, 1e7), on the edge of the sub-grid -> extension to mu_sparse = 1e8, 1e9
# (same mu_smooth 1e5 ... 1e8). output: results/grids_111111111111/grid_sure_medium_ellipse_alpha5em6_gradC/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results
CLUSTERS=(vercors14 vercors9)
n=0
for sm in 1e5 1e6 1e7 1e8; do
  for sp in 1e8 1e9; do
    CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
    oarsub -l "gpu=1,walltime=18:00:00"\
      -p "cluster='$CL'" \
      -n "gradC_sure_medium_ellipse_alpha5e-6_sm${sm}_sp${sp}" \
      "CUDA_VISIBLE_DEVICES=0 $PY $RUN/syntheticdata_sure_gpu.py --mu-smooth $sm --mu-sparse $sp --flux 5e-6 \
        --shape medium_ellipse --tag gradC"
  done
done
