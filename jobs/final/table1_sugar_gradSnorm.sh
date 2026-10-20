#!/bin/bash

# Table 1 with the hyperparameters found by refined MC-SUGAR (VMLMB, normalized, stopping criteria matched to the
# precision of MC-SURE, gradSnorm60tol) at angle 0 (results/sugar/<shape>_alpha<..>_angle0_gradSnorm60tol_refined/trajectory.npz, mu_opt rounded to 3 significant digits),
# reused for the 10 parallactic angles, one contrast per job.
# output: results/mdrex_results/musmooth<mu>_musparse<mu>_<shape>_alpha<1em6|5em6|1em5>_gradS/x_opt.fits
# Jobs spread alternately over vercors14 and vercors9.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/syntheticdata_shape.py
CLUSTERS=(vercors14 vercors9)
n=0
#     shape:flux:mu_smooth:mu_sparse
for job in medium_ellipse:1e-6:1.36e7:1.49e6 medium_ellipse:5e-6:3.89e6:1.4e6 medium_ellipse:1e-5:2.67e6:1.28e6 \
           spiral:1e-6:1.15e7:1.39e6 spiral:5e-6:1.02e7:5.39e5 spiral:1e-5:9.32e6:6.15e5 \
           circle:1e-6:1.08e7:1.6e6 circle:5e-6:3.7e6:7.82e5 circle:1e-5:3.46e6:8.97e5; do
  IFS=: read shape flux sm sp <<< "$job"
  tag=$(echo $flux | sed 's/e-/em/')
  [ -e "$ROOT/results/mdrex_results/musmooth${sm}_musparse${sp}_${shape}_alpha${tag}_gradS/x_opt.fits" ] && { echo "done: $job"; continue; }
  CL=${CLUSTERS[$((n % ${#CLUSTERS[@]}))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=10:00:00" -p "cluster='$CL'" -n "t1sugar_${shape}_a${tag}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --mu-smooth $sm --mu-sparse $sp --shape $shape --flux $flux --tag gradS"
done
echo "$n jobs submitted"
