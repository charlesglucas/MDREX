#!/bin/bash

# Table 1 with the hyperparameters selected by refined MC-SURE (gradS) on the angle-0 grids
# (results/grids_111111111111/grid_sure_<shape>_alpha<flux>_angle0_gradS): 10 parallactic angles, one contrast per job.
# Refined MC-SURE with one iteration (covariance re-estimated on the residual of the pilot reconstruction,
# pilot = argmin MC-SURE(C^y)), see disk_rec_sure_fixedW.py; hyperparameters selected at angle 0, reused for the
# 10 parallactic angles.
# output: results/mdrex_results/musmooth<mu>_musparse<mu>_<shape>_alpha<1em6|5em6|1em5>_gradS/x_opt.fits
# Jobs on vercors18.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/syntheticdata_shape.py
CLUSTERS=(vercors18)
n=0
#     shape:flux:mu_smooth:mu_sparse
for job in medium_ellipse:1e-6:3e7:1e6 medium_ellipse:5e-6:5e6:1e6 medium_ellipse:1e-5:3e6:1e6 \
           spiral:1e-6:1e8:1e6 spiral:5e-6:3e7:3e5 spiral:1e-5:1e7:3e5 \
           circle:1e-6:3e8:3e5 circle:5e-6:1e6:1e6 circle:1e-5:3e6:1e6; do
  IFS=: read shape flux sm sp <<< "$job"
  tag=$(echo $flux | sed 's/e-/em/')
  [ -e "$ROOT/results/mdrex_results/musmooth${sm}_musparse${sp}_${shape}_alpha${tag}_gradS/x_opt.fits" ] && { echo "done: $job"; continue; }
  CL=${CLUSTERS[$((n % ${#CLUSTERS[@]}))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=10:00:00" -p "cluster='$CL'" -n "t1ref_${shape}_a${tag}_sm${sm}_sp${sp}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --mu-smooth $sm --mu-sparse $sp --shape $shape --flux $flux --tag gradS"
done
echo "$n jobs submitted"
