#!/bin/bash

# Spiral, gradS, angle 0, 3 contrasts: intermediate couples around (1e7, 1e5), the most robust couple of the decade
# grid (spiral_grid_gradS.sh), to find a single couple good at all contrasts (1e-6 needs more smoothing).
# One job per mu_smooth, spread over vercors14 and vercors9; couples already computed are skipped.
# output: results/grid_shape/spiral_angle0_gradS/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_shape_grid.py
CLUSTERS=(vercors14 vercors9)
n=0
for job in "1e7:2e5 5e5" "2e7:1e5 2e5 5e5" "5e7:1e5 2e5 5e5"; do
  sm=${job%%:*}; sps=${job#*:}
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=15:00:00" -p "cluster='$CL'" -n "gradS_grid_spiral_refine_sm${sm}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN --shape spiral --angle 0 --mu-smooth $sm --mu-sparse $sps --tag gradS"
done
