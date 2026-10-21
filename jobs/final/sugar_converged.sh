#!/bin/bash

# MC-SUGAR converged before the refinement, then refinement iterated (VMLMB, MC-SURE normalized):
#   stage 0: MC-SUGAR with C^y, warm start at the couple of the previous run (gradSnorm60tol), stopped ONLY by the test
#            on the projected gradient (5% of its initial norm; tests on MC-SURE and on the step disabled: they stopped
#            the previous runs with a gradient still at 30-86% of its initial norm), at most 60 evaluations
#            -> results/sugar/<shape>_alpha<..>_angle0_gradSconv/
#   stages 1, 2, 3: refined MC-SUGAR, covariance re-estimated on the residual of the reconstruction at the couple of the
#            previous stage, same stopping criterion -> ..._gradSconv_refined, _refined2, _refined3
# Each stage starts when the previous one has finished (oarsub -a); the 4 stages of a case on the same cluster.
# Usage: bash sugar_converged.sh [shape ...]   (default: medium_ellipse spiral circle)
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/syntheticdata_sugar.py
TAG=gradSconv
OPTS="--angle 0 --bounds 6 9 4.4771 7.4771 --maxfun 60 --hvp-step 1e-2 --normalize --ftol 0 --xtol 0 --gtol 0.05 --tag $TAG"
CLUSTERS=(vercors14 vercors9)
SHAPES=("${@:-medium_ellipse spiral circle}")
n=0
for shape in ${SHAPES[@]}; do
  for flux in 1e-6 5e-6 1e-5; do
    tag=$(echo $flux | sed 's/e-/em/')
    CL=${CLUSTERS[$((n % ${#CLUSTERS[@]}))]}; n=$((n + 1))
    mu0=$($PY -c "import numpy as np; m=np.load('$ROOT/results/sugar/${shape}_alpha${tag}_angle0_gradSnorm60tol/trajectory.npz')['mu_opt']; print(f'{m[0]:.4e} {m[1]:.4e}')")
    out=$(oarsub -l "gpu=1,walltime=48:00:00" -p "cluster='$CL'" -n "sugarconv_${shape}_${tag}_0" \
      "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --flux $flux --shape $shape --mu0 $mu0 $OPTS")
    id=$(echo "$out" | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2); echo "$shape $flux stage 0: $id"
    for k in 1 2 3; do
      [ -z "$id" ] && { echo "submission failed: $shape $flux stage $k"; break; }
      out=$(oarsub -a $id -l "gpu=1,walltime=48:00:00" -p "cluster='$CL'" -n "sugarconv_${shape}_${tag}_${k}" \
        "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --flux $flux --shape $shape --refined --refined-iter $k $OPTS")
      id=$(echo "$out" | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2); echo "$shape $flux stage $k: $id"
    done
  done
done
