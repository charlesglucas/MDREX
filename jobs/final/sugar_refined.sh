#!/bin/bash

# Refined MC-SUGAR (gradS), angle 0: minimization of the refined MC-SURE (metric re-estimated on the residual of the
# reconstruction at the couple found by MC-SUGAR with C^y, results/sugar/<shape>_alpha<..>_angle0_gradS/trajectory.npz)
# with its MC-SUGAR gradient and L-BFGS-B on log10(mu), started at this couple, up to 25 MC-SURE evaluations.
# Bounds = domain of the angle-0 grids: mu_smooth in [1e6, 1e9], mu_sparse in [3e4, 3e7].
# Usage: bash sugar_refined.sh [shape]   (default medium_ellipse) -> one job per contrast on vercors18.
# output: results/sugar/<shape>_alpha<1em6|5em6|1em5>_angle0_gradS_refined/trajectory.npz
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
SHAPE=${1:-medium_ellipse}
for flux in 1e-6 5e-6 1e-5; do
  oarsub -l "gpu=1,walltime=48:00:00" -p "cluster='vercors18'" -n "sugar_refined_${SHAPE}_${flux}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $ROOT/.venv/bin/python $ROOT/scripts/run_results/syntheticdata_sugar.py \
      --flux $flux --shape $SHAPE --angle 0 --refined --bounds 6 9 4.4771 7.4771 --maxfun 25 --hvp-step 1e-2"
done
