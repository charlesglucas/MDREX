#!/bin/bash

# Hyperparameters at angle 0 (Table 1) by MC-SUGAR (gradS), for each shape and contrast, in two chained jobs:
#   1. MC-SUGAR with the metric C^y, started at (1e7, 1e6)
#      -> results/sugar/<shape>_alpha<1em6|5em6|1em5>_angle0_gradS/trajectory.npz (mu_opt)
#   2. refined MC-SUGAR (metric re-estimated on the residual of the reconstruction at mu_opt of step 1), started at
#      mu_opt, launched when step 1 has finished (oarsub -a)
#      -> results/sugar/<shape>_alpha<1em6|5em6|1em5>_angle0_gradS_refined/trajectory.npz
# Both minimizations: L-BFGS-B on log10(mu) in the domain of the angle-0 grids, mu_smooth in [1e6, 1e9] and
# mu_sparse in [3e4, 3e7], up to 25 MC-SURE evaluations (2 reconstructions from x = 0 + 2 CG solves each).
# A step already done (trajectory.npz with mu_opt) is skipped.
# Usage: bash sugar_angle0.sh [shape ...]   (default: spiral circle; the ellipse is in sugar_3alpha.sh / sugar_refined.sh)
#        NORM=1 bash sugar_angle0.sh medium_ellipse spiral circle   -> MC-SURE normalized by the norm of MC-SUGAR at the
#        first evaluation (--normalize), results in results/sugar/<shape>_alpha<..>_angle0_gradSnorm[_refined]
#        NORM=1 MAXFUN=60 bash sugar_angle0.sh ...   -> up to 60 evaluations, results in ..._angle0_gradSnorm60[_refined]
# Jobs spread alternately over vercors14 and vercors9 (MC-SUGAR and its refined step on the same cluster).
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/syntheticdata_sugar.py
MAXFUN=${MAXFUN:-25}   # maximum number of MC-SURE evaluations per minimization; MAXFUN != 25 -> tag suffixed with it
TAG=gradS; [ "${NORM:-0}" = 1 ] && TAG=gradSnorm; [ "$MAXFUN" != 25 ] && TAG=${TAG}${MAXFUN}
OPTS="--angle 0 --bounds 6 9 4.4771 7.4771 --maxfun $MAXFUN --hvp-step 1e-2 --tag $TAG"
[ "${NORM:-0}" = 1 ] && OPTS="$OPTS --normalize"
CLUSTERS=(vercors14 vercors9)
SHAPES=("${@:-spiral circle}")
done_step() {  # done_step <folder>: trajectory.npz with mu_opt
  [ -e "$1/trajectory.npz" ] && $PY -c "import numpy as np, sys; sys.exit(0 if 'mu_opt' in np.load('$1/trajectory.npz').files else 1)"
}
n=0
for shape in ${SHAPES[@]}; do
  for flux in 1e-6 5e-6 1e-5; do
    tag=$(echo $flux | sed 's/e-/em/')
    dir=$ROOT/results/sugar/${shape}_alpha${tag}_angle0_${TAG}
    CL=${CLUSTERS[$((n % ${#CLUSTERS[@]}))]}; n=$((n + 1))
    dep=""
    if ! done_step $dir; then
      out=$(oarsub -l "gpu=1,walltime=48:00:00" -p "cluster='$CL'" -n "sugar_${TAG}_${shape}_${tag}" \
        "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --flux $flux --shape $shape --mu0 1e7 1e6 $OPTS")
      echo "$out"
      id=$(echo "$out" | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2)
      [ -z "$id" ] && { echo "submission failed: $shape $flux"; continue; }
      dep="-a $id"
    fi
    if ! done_step ${dir}_refined; then
      oarsub $dep -l "gpu=1,walltime=48:00:00" -p "cluster='$CL'" -n "sugar_refined_${TAG}_${shape}_${tag}" \
        "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --flux $flux --shape $shape --refined $OPTS"
    fi
  done
done
