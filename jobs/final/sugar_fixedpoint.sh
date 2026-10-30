#!/bin/bash
# Refined MC-SUGAR iterated up to the fixed point of the covariance, from the second refined stage (rhofixfp_refined2):
# pilot = couple of rhofixfp_refined2, stages rhofixfp_refined3, _refined4, ... (at most 1 + --outer), stops when the couple
# moves by less than 0.1 decade. More accurate MC-SUGAR (CG up to 80 iterations), larger budget of evaluations and
# trust region of 0.5 decade around the best couple (no more line-search steps to the corners of the domain).
#   bash sugar_fixedpoint.sh <shape> <flux> [<shape> <flux> ...]   e.g. bash sugar_fixedpoint.sh spiral 1e-6 spiral 5e-6
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P="cluster IN ('vercors10','vercors11','vercors12','vercors13')"
OPTS="--angle 0 --bounds 6 9 4.4771 7.4771 --hvp-step 1e-2 --normalize --maxfun 30 --xtol 0.05 --ftol 0 --gtol 0 \
      --cg-maxiter 80 --max-step 0.5 --refined --refined-iter 3 --outer 4 --outer-tol 0.1 --rec-ftol 0 --rec-xtol 1e-10 \
      --tag rhofixfp"
while [ $# -ge 2 ]; do
  shape=$1; flux=$2; shift 2
  tag=$(echo $flux | sed 's/e-/em/')
  oarsub -l "gpu=1,walltime=72:00:00" -p "$P" -n "sugarfp_${shape}_${tag}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar.py --flux $flux --shape $shape $OPTS" \
    | grep -o "OAR_JOB_ID=[0-9]*" | sed "s/^/sugarfp_${shape}_${tag}: /"
done
