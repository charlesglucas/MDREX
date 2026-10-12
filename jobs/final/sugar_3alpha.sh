#!/bin/bash

# Minimization of MC-SURE (C_y metric, gradS) over (mu_smooth, mu_sparse) with its MC-SUGAR gradient (implicit
# differentiation, reconstruction/sugar.py) and L-BFGS-B on log10(mu), angle 0, start (1e7, 1e6), up to 25 MC-SURE
# evaluations, each = 2 reconstructions from x = 0 + 2 CG solves (<= 20 Hessian-vector products each); test of the
# finite-difference Hessian-vector products at the first point.
# Usage: bash sugar_ellipse_1em6.sh [shape]   (default medium_ellipse) -> one job per contrast (1e-6, 5e-6, 1e-5),
# spread over vercors9 and vercors14. output: results/sugar/<shape>_alpha<1em6|5em6|1em5>_angle0_gradS/trajectory.npz
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
SHAPE=${1:-medium_ellipse}
CLUSTERS=(vercors9 vercors14)
n=0
for flux in 1e-6 5e-6 1e-5; do
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=48:00:00" -p "cluster='$CL'" -n "sugar_${SHAPE}_${flux}" \
    "CUDA_VISIBLE_DEVICES=0 $ROOT/.venv/bin/python $ROOT/scripts/run_results/syntheticdata_sugar.py \
      --flux $flux --shape $SHAPE --angle 0 --mu0 1e7 1e6 --maxfun 25 --hvp-step 1e-2"
done
