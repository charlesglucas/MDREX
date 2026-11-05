#!/bin/bash
# MC-SUGAR by iterative differentiation of the reconstructions (scripts/run_results/syntheticdata_sugar_iter.py,
# reconstruction/sugar_iterative.py; the implicit version syntheticdata_sugar.py / sugar.py is unchanged), angle 0, tag
# sugariter, reconstructions with the gradient of mu_sparse sum(x) (see the header of syntheticdata_sugar_iter.py).
#   MODE=check bash sugar_iter.sh <shape> <flux> <mu_smooth> <mu_sparse> [...]
#       validation at a couple: reconstructions and their convergence, MC-SUGAR (iterative) vs central finite differences
#       of MC-SURE (0.05 and 0.15 decade, same reconstructions), implicit MC-SUGAR (CG 80 / 320 / 640) at the same x_hat
#       -> results/sugar/<shape>_alpha<..>_angle0_sugariter_check/
#   MODE=opt bash sugar_iter.sh <shape> <flux> <mu_smooth_0> <mu_sparse_0> [...]
#       minimization of MC-SURE (C^y) from the given couple (e.g. the minimum of MC-SURE on the angle-0 grid), then
#       refined MC-SURE iterated to the fixed point of the covariance (--outer 2, 0.1 decade), in the same job
#       -> results/sugar/<shape>_alpha<..>_angle0_sugariter[_refined, _refined2]/trajectory.npz
# EXTRA (env): additional options, e.g. EXTRA="--abs-grad" (gradient of sum|x| as run_bfgs) with TAG=sugariterabs.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P="cluster IN ('vercors10','vercors11','vercors12','vercors13')"
TAG=${TAG:-sugariter}
BASE="--angle 0 --hvp-step 1e-2 --normalize --rec-ftol 0 --rec-xtol 1e-10 --tag $TAG $EXTRA"
while [ $# -ge 4 ]; do
  shape=$1; flux=$2; sm=$3; sp=$4; shift 4
  tag=$(echo $flux | sed 's/e-/em/')
  if [ "${MODE:-check}" = check ]; then
    OPTS="$BASE --check --check-mu $sm $sp --check-steps 0.05 0.15 --check-repeats 1 --check-cg 80 320 640 --cg-rtol 0"
    name=sugariterchk_${shape}_${tag}; wall=36
  else
    OPTS="$BASE --mu0 $sm $sp --bounds 6 9 4.4771 7.4771 --maxfun 20 --xtol 0.02 --ftol 0 --gtol 0 --gabs 100 \
          --restarts 3 --max-step 0.5 --outer 2 --outer-tol 0.1"
    name=sugariteropt_${shape}_${tag}; wall=168
  fi
  oarsub -l "gpu=1,walltime=$wall:00:00" -p "$P" -n "$name" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar_iter.py --flux $flux --shape $shape $OPTS" \
    | grep -o "OAR_JOB_ID=[0-9]*" | sed "s/^/$name: /"
done
