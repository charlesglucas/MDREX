#!/bin/bash

# Fig. 3 (patch distributions, alpha = 5e-6, 10 parallactic angles) with the hyperparameters of Table 1, found by refined
# MC-SUGAR (gradSnorm60tol) at angle 0 for alpha = 5e-6, rounded to 3 significant digits.
# output: results/distributions_5em6/<shape>_musmooth<mu>_musparse<mu>_gradS/x_opt.fits
# Jobs spread alternately over vercors14 and vercors9.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
CLUSTERS=(vercors14 vercors9)
n=0
for job in medium_ellipse:3.89e6:1.4e6 spiral:1.02e7:5.39e5 circle:3.7e6:7.82e5; do
  IFS=: read shape sm sp <<< "$job"
  [ -e "$ROOT/results/distributions_5em6/${shape}_musmooth${sm}_musparse${sp}_gradS/x_opt.fits" ] && { echo "done: $job"; continue; }
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=24:00:00" -p "cluster='$CL'" -n "distrib_sugar_${shape}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_results/syntheticdata_distributions.py --mu-smooth $sm --mu-sparse $sp --shape $shape --tag gradS"
done
echo "$n jobs submitted"
