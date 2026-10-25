#!/bin/bash

# All the synthetic experiments with fixed shrinkage coefficients (Algorithm 1, default since 5eb1c72), float32,
# reconstructions with VMLMB ftol 0 / xtol 1e-10 (stop at the precision of the objective):
#   A. MC-SURE grids at angle 0 (grid_rhofixed.sh), on vercors10-13 / vercors18 (any free GPU)
#   B. per shape and contrast, on vercors9 / vercors14: MC-SUGAR then refined MC-SUGAR (fixed-point iterations on the
#      covariance, at most 2 refined stages, syntheticdata_sugar.py --outer 2), warm start at the couple of the previous
#      MC-SUGAR runs (gradSnorm60tol) -> results/sugar/<shape>_alpha<..>_angle0_rhofixfp[_refined, _refined2]
#   C. after B: Table 1 (10 angles) at the refined MC-SUGAR couple (run_from_sugar.sh table1)
#   D. after B at 5e-6: Fig. 3 patch distributions at the same couple (run_from_sugar.sh distrib)
# Usage: bash all_rhofix.sh
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
J=$ROOT/jobs/final
# A. grids
for shape in medium_ellipse spiral circle; do
  for flux in 1e-6 5e-6 1e-5; do
    CLUSTERS="vercors10 vercors11 vercors12 vercors13 vercors18" bash $J/grid_rhofixed.sh $shape $flux 4
  done
done
# B-D. MC-SUGAR, Table 1, Fig. 3
P9_14="cluster IN ('vercors9','vercors14')"
OPTS="--angle 0 --bounds 6 9 4.4771 7.4771 --hvp-step 1e-2 --normalize --maxfun 12 --xtol 0.05 --ftol 0 --gtol 0 \
      --outer 2 --outer-tol 0.1 --rec-ftol 0 --rec-xtol 1e-10 --tag rhofixfp"
for shape in medium_ellipse spiral circle; do
  for flux in 1e-6 5e-6 1e-5; do
    tag=$(echo $flux | sed 's/e-/em/')
    mu0=$($PY -c "import numpy as np; m=np.load('$ROOT/results/sugar/${shape}_alpha${tag}_angle0_gradSnorm60tol/trajectory.npz')['mu_opt']; print(f'{m[0]:.4e} {m[1]:.4e}')")
    id=$(oarsub -l "gpu=1,walltime=72:00:00" -p "$P9_14" -n "sugarrf_${shape}_${tag}" \
      "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar.py --flux $flux --shape $shape --mu0 $mu0 $OPTS" \
      | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2)
    [ -z "$id" ] && { echo "submission failed: $shape $flux"; continue; }
    echo "sugarrf_${shape}_${tag}: $id (mu0 = $mu0)"
    oarsub -a $id -l "gpu=1,walltime=24:00:00" -p "$P9_14" -n "t1rf_${shape}_${tag}" \
      "bash $J/run_from_sugar.sh table1 $shape $flux" | grep -o "OAR_JOB_ID=[0-9]*"
    [ $flux = 5e-6 ] && oarsub -a $id -l "gpu=1,walltime=60:00:00" -p "$P9_14" -n "distribrf_${shape}" \
      "bash $J/run_from_sugar.sh distrib $shape $flux" | grep -o "OAR_JOB_ID=[0-9]*"
  done
done
