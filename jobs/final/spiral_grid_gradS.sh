#!/bin/bash

# Spiral, gradS (gradient through m_hat / C_hat, shrinkage not differentiated): grid of (mu_smooth, mu_sparse) at the
# 3 contrasts, angle 0, to find the best couple(s) for Tables 1-2. One job per mu_smooth (3 mu_sparse x 3 contrasts),
# spread over vercors14 and vercors9; couples already computed are skipped.
# output: results/grid_shape/spiral_angle0_gradS/musmooth<mu>_musparse<mu>.npz (x_opt, PSNR whole / support)
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_shape_grid.py
CLUSTERS=(vercors14 vercors9)
n=0
for sm in 1e5 1e6 1e7 1e8; do
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=15:00:00" -p "cluster='$CL'" -n "gradS_grid_spiral_sm${sm}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN --shape spiral --angle 0 --mu-smooth $sm --mu-sparse 1e4 1e5 1e6 --tag gradS"
done
