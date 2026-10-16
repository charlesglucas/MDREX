#!/bin/bash

# Synthetic results with the disks centred on the star (scripts/run_results/make_centered_disks.py), tag gradSc.
# Prerequisite (once, on the frontend): centred disks in data/synthetic_disks, originals kept in data/synthetic_disks_orig
#   python scripts/run_results/make_centered_disks.py
#   mv data/synthetic_disks data/synthetic_disks_orig && mv data/synthetic_disks_centered data/synthetic_disks
# This script submits (vercors14 / vercors9):
#   1. MC-SURE grids at angle 0, square 7 x 7 grid, 3 shapes x 3 contrasts (table1_sure_gradS.sh, 9 jobs)
#      -> results/grids_111111111111/grid_sure_<shape>_alpha<..>_angle0_gradSc/
#   2. patch distributions, alpha = 5e-6, 10 angles (Fig. 3, same couples as in the paper), 3 jobs
#      -> results/distributions_5em6/<shape>_musmooth<..>_musparse<..>_gradSc/
# Then (after the grids): selection by refined MC-SURE, Table 1 (10 angles) and MC-SUGAR.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
[ -d $ROOT/data/synthetic_disks_orig ] || { echo "centred disks not installed (see the prerequisite above)"; exit 1; }
TAG=gradSc bash $ROOT/jobs/final/table1_sure_gradS.sh
CLUSTERS=(vercors14 vercors9)
n=0
for job in medium_ellipse:1e6:1e6 spiral:5e6:1e5 circle:5e6:1e5; do
  IFS=: read shape sm sp <<< "$job"
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=20:00:00" -p "cluster='$CL'" -n "gradSc_distrib_${shape}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_results/syntheticdata_distributions.py --mu-smooth $sm --mu-sparse $sp --shape $shape --tag gradSc"
done
