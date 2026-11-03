#!/bin/bash
# Refined MC-SUGAR iterated up to the fixed point of the covariance, from scratch (MC-SUGAR with C^y, then refined stages),
# with the Monte Carlo probe drawn on the CPU (same vector on every GPU model), tag rhofixcpu.
# One OAR job per stage (at most 40 MC-SURE evaluations, up to ~35 h), chained with -a:
#   stage 0: MC-SUGAR with C^y (rhofixcpu), stage k >= 1: refined MC-SUGAR, pilot = stage k-1 (rhofixcpu_refined[k])
# Stage k >= 2 is skipped if the fixed point is reached (couple of stage k-1 within 0.1 decade of that of stage k-2), if
# stage k-1 drifted to the lower bound of mu_smooth (stage k-2 is then kept), or if stage k-1 is missing.
# Each stage stops on the projected MC-SUGAR gradient (--gabs 100 per decade) and VMLMB is restarted from the best couple
# when it stops earlier on its variables-change test (narrow valley of MC-SURE, inexact MC-SUGAR), at most 6 times.
#   bash sugar_cpu_chain.sh <shape> <flux> [<shape> <flux> ...]   e.g. bash sugar_cpu_chain.sh spiral 1e-6 circle 5e-6
# NSTAGES (env, default 6): stages 0 to NSTAGES-1.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P="cluster IN ('vercors10','vercors11','vercors12','vercors13')"
NSTAGES=${NSTAGES:-6}
OPTS="--angle 0 --bounds 6 9 4.4771 7.4771 --hvp-step 1e-2 --normalize --maxfun 40 --xtol 0.02 --ftol 0 --gtol 0 \
      --gabs 100 --restarts 6 --cg-maxiter 80 --max-step 0.5 --rec-ftol 0 --rec-xtol 1e-10 --tag rhofixcpu"
dir() { [ $1 -eq 0 ] && echo "$2" || { [ $1 -eq 1 ] && echo "$2_refined" || echo "$2_refined$1"; }; }
while [ $# -ge 2 ]; do
  shape=$1; flux=$2; shift 2
  tag=$(echo $flux | sed 's/e-/em/')
  base=$ROOT/results/sugar/${shape}_alpha${tag}_angle0_rhofixcpu
  prev=""; ids=""
  for k in $(seq 0 $((NSTAGES - 1))); do
    if [ $k -eq 0 ]; then
      cmd="cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar.py --flux $flux --shape $shape $OPTS"
    else
      a=$(dir $((k - 1)) $base); b=$(dir $((k - 2)) $base)
      run="CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar.py --flux $flux --shape $shape $OPTS --refined --refined-iter $k"
      if [ $k -eq 1 ]; then
        cmd="cd $ROOT && [ -e $a/trajectory.npz ] && $run || echo 'stage 0 missing: nothing to do'"
      else
        check="import numpy as np, sys; a, b = (np.log10(np.load(f'{d}/trajectory.npz')['mu_opt']) for d in ('$a', '$b')); \
sys.exit(int(np.abs(a - b).max() < 0.1 or a[0] <= 6 + 1e-3))"
        cmd="cd $ROOT && [ -e $a/trajectory.npz ] && $PY -c \"$check\" && $run || echo 'fixed point reached, drift or missing stage: nothing to do'"
      fi
    fi
    j=$(oarsub ${prev:+-a $prev} -l "gpu=1,walltime=72:00:00" -p "$P" -n "sugarcpu${k}_${shape}_${tag}" "$cmd" \
        | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2)
    prev=$j; ids="$ids $j"
  done
  echo "${shape} ${flux}: stages 0-$((NSTAGES - 1)) =$ids"
done
