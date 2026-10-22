#!/bin/bash

# MC-SUGAR with fixed-point iterations on the covariance, in a single job per case (VMLMB, MC-SURE normalized):
#   stage 0: MC-SUGAR with C^y, warm start at the couple of the previous run (gradSnorm60tol)
#   stages 1..4: covariance re-estimated on the residual of the reconstruction at the couple of the previous stage,
#            MC-SURE minimized again from this couple (--outer 4); stop when the couple moves by less than 0.1 decade,
#            or when mu_smooth reaches its lower bound (drift: previous stage kept)
# Each stage: stop when the step of VMLMB in log10(mu) is below 0.05 decade (12% on mu), at most 12 evaluations; the
# tests on MC-SURE and on the gradient are disabled (MC-SURE is piecewise smooth in mu_sparse: the gradient does not
# vanish near the optimum).
# output: results/sugar/<shape>_alpha<..>_angle0_gradSfp[_refined, _refined2, ...], <..>_gradSfp_fixedpoint.npz
# Usage: bash sugar_fixedpoint.sh [shape ...]   (default: medium_ellipse spiral circle); vercors14 / vercors9
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/syntheticdata_sugar.py
OPTS="--angle 0 --bounds 6 9 4.4771 7.4771 --hvp-step 1e-2 --normalize --maxfun 12 --xtol 0.05 --ftol 0 --gtol 0 \
      --outer 4 --outer-tol 0.1 --tag gradSfp"
CLUSTERS=(vercors14 vercors9)
SHAPES=("${@:-medium_ellipse spiral circle}")
n=0
for shape in ${SHAPES[@]}; do
  for flux in 1e-6 5e-6 1e-5; do
    tag=$(echo $flux | sed 's/e-/em/')
    CL=${CLUSTERS[$((n % ${#CLUSTERS[@]}))]}; n=$((n + 1))
    mu0=$($PY -c "import numpy as np; m=np.load('$ROOT/results/sugar/${shape}_alpha${tag}_angle0_gradSnorm60tol/trajectory.npz')['mu_opt']; print(f'{m[0]:.4e} {m[1]:.4e}')")
    oarsub -l "gpu=1,walltime=72:00:00" -p "cluster='$CL'" -n "sugarfp_${shape}_${tag}" \
      "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --flux $flux --shape $shape --mu0 $mu0 $OPTS" | grep OAR_JOB_ID
  done
done
