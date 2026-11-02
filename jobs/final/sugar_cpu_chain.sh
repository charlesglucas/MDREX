#!/bin/bash
# Refined MC-SUGAR iterated up to the fixed point of the covariance, from scratch (MC-SUGAR with C^y, then refined stages),
# with the Monte Carlo probe drawn on the CPU (same vector on every GPU model), tag rhofixcpu. Same options as
# sugar_fixedpoint.sh. Two chained jobs of at most 72 h:
#   job A: MC-SUGAR with C^y (rhofixcpu) + refined stages 1, 2 (rhofixcpu_refined, _refined2)
#   job B (after A): refined stages 3 to 6 from the pilot of _refined2, skipped if A already reached the fixed point
#            (couple of _refined2 within 0.1 decade of that of _refined) or drifted to the lower bound of mu_smooth
#   bash sugar_cpu_chain.sh <shape> <flux> [<shape> <flux> ...]   e.g. bash sugar_cpu_chain.sh spiral 1e-6 spiral 5e-6
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P="cluster IN ('vercors10','vercors11','vercors12','vercors13')"
OPTS="--angle 0 --bounds 6 9 4.4771 7.4771 --hvp-step 1e-2 --normalize --maxfun 30 --xtol 0.05 --ftol 0 --gtol 0 \
      --cg-maxiter 80 --max-step 0.5 --outer-tol 0.1 --rec-ftol 0 --rec-xtol 1e-10 --tag rhofixcpu"
while [ $# -ge 2 ]; do
  shape=$1; flux=$2; shift 2
  tag=$(echo $flux | sed 's/e-/em/')
  base=$ROOT/results/sugar/${shape}_alpha${tag}_angle0_rhofixcpu
  ja=$(oarsub -l "gpu=1,walltime=72:00:00" -p "$P" -n "sugarcpuA_${shape}_${tag}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar.py --flux $flux --shape $shape $OPTS --outer 2" \
    | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2)
  check="import numpy as np, sys; a, b = (np.log10(np.load(f'${base}{s}/trajectory.npz')['mu_opt']) for s in ('_refined', '_refined2')); \
sys.exit(int(np.abs(b - a).max() < 0.1 or b[0] <= 6 + 1e-3))"
  jb=$(oarsub -a $ja -l "gpu=1,walltime=72:00:00" -p "$P" -n "sugarcpuB_${shape}_${tag}" \
    "cd $ROOT && [ -e ${base}_refined2/trajectory.npz ] && $PY -c \"$check\" && CUDA_VISIBLE_DEVICES=0 $PY \
     scripts/run_results/syntheticdata_sugar.py --flux $flux --shape $shape $OPTS --refined --refined-iter 3 --outer 3 \
     || echo 'fixed point already reached in job A: nothing to do'" \
    | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2)
  echo "${shape} ${flux}: A=$ja B=$jb"
done
