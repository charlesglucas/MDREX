#!/bin/bash

# MC-SURE grid with the gradient through m_hat / C_hat, ellipse, alpha = 5e-6: completes the test sub-grid
# (mu_smooth 1e5..1e8 x mu_sparse 1e4..1e9, test_gradC.sh + sure_gradC_extend.sh) into a 6 x 6 grid
# mu_smooth 1e4..1e9 x mu_sparse 1e4..1e9 (rows mu_smooth = 1e4 and 1e9).
# output: results/grids_111111111111/grid_sure_medium_ellipse_alpha5em6_gradC/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results
CLUSTERS=(vercors14 vercors9)
n=0
for sm in 1e4 1e9; do
  for sp in 1e4 1e5 1e6 1e7 1e8 1e9; do
    CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
    oarsub -l "gpu=1,walltime=18:00:00"\
      -p "cluster='$CL'" \
      -n "gradC_sure_medium_ellipse_alpha5e-6_sm${sm}_sp${sp}" \
      "CUDA_VISIBLE_DEVICES=0 $PY $RUN/syntheticdata_sure_gpu.py --mu-smooth $sm --mu-sparse $sp --flux 5e-6 \
        --shape medium_ellipse --tag gradC"
  done
done
