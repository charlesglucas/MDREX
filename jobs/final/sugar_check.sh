#!/bin/bash
# Diagnostic of MC-SURE / MC-SUGAR (tag rhofixcpu, metric C^y) at the couple where VMLMB stalls: MC-SURE evaluated again at
# the same couple (reproducibility of the reconstructions) and MC-SUGAR compared with central finite differences
# (0.05 and 0.15 decade on each axis). Results: results/sugar/<shape>_alpha<flux>_angle0_rhofixcpu_check/check.npz
#   bash sugar_check.sh <shape> <flux> <mu_smooth> <mu_sparse> [<shape> <flux> <mu_smooth> <mu_sparse> ...]
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P="cluster IN ('vercors10','vercors11','vercors12','vercors13')"
OPTS="--angle 0 --hvp-step 1e-2 --normalize --cg-maxiter 80 --rec-ftol 0 --rec-xtol 1e-10 --tag rhofixcpu"
while [ $# -ge 4 ]; do
  shape=$1; flux=$2; sm=$3; sp=$4; shift 4
  tag=$(echo $flux | sed 's/e-/em/')
  oarsub -l "gpu=1,walltime=24:00:00" -p "$P" -n "sugarcheck_${shape}_${tag}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar.py --flux $flux --shape $shape $OPTS --check --check-mu $sm $sp"
done
