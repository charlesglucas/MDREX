#!/bin/bash

# Minimization of MC-SURE (C_y metric, gradS) over (mu_smooth, mu_sparse) with its MC-SUGAR gradient (implicit
# differentiation, reconstruction/sugar.py) and L-BFGS-B on log10(mu): ellipse, alpha = 1e-6, angle 0, start (1e7, 1e6)
# (center of the gradS grid; MSE minimum (1e8, 1e6), MC-SURE grid minimum (1e8, 1e6)). Up to 25 MC-SURE evaluations,
# each = 2 reconstructions (warm started) + 2 CG solves (<= 20 Hessian-vector products each).
# output: results/sugar/medium_ellipse_alpha1em6_angle0_gradS/trajectory.npz
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
oarsub -l "gpu=1,walltime=48:00:00" -p "cluster='vercors9'" -n "sugar_ellipse_1em6" \
  "CUDA_VISIBLE_DEVICES=0 $ROOT/.venv/bin/python $ROOT/scripts/run_results/syntheticdata_sugar.py \
    --flux 1e-6 --shape medium_ellipse --angle 0 --mu0 1e7 1e6 --maxfun 25"
