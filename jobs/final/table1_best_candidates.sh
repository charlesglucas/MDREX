#!/bin/bash

# Table 1 with hyperparameters chosen per geometry and contrast to maximize the PSNR (oracle choice, ground truth used):
# 10-angle runs of candidate couples where no available run beats REXPACO, chosen from the PSNR of the angle-0 grids
# (results/grids_111111111111/grid_sure_<shape>_alpha<..>_angle0_gradS) around their PSNR maximum.
#   spiral 1e-6: (3e7, 3e5) (1e7, 3e5) (3e7, 1e5); spiral 5e-6 and 1e-5: (3e6, 1e5)
#   circle 1e-6: (3e7, 3e5) (1e7, 3e5) (1e7, 1e5); circle 5e-6: (1e7, 1e5) (3e6, 1e5) (1e7, 3e5)
# output: results/mdrex_results/musmooth<mu>_musparse<mu>_<shape>_alpha<1em6|5em6|1em5>_gradS/x_opt.fits
# Jobs spread alternately over vercors14 and vercors9.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/syntheticdata_shape.py
CLUSTERS=(vercors14 vercors9)
n=0
for job in spiral:1e-6:3e7:3e5 spiral:1e-6:1e7:3e5 spiral:1e-6:3e7:1e5 spiral:5e-6:3e6:1e5 spiral:1e-5:3e6:1e5 \
           circle:1e-6:3e7:3e5 circle:1e-6:1e7:3e5 circle:1e-6:1e7:1e5 circle:5e-6:1e7:1e5 circle:5e-6:3e6:1e5 circle:5e-6:1e7:3e5; do
  IFS=: read shape flux sm sp <<< "$job"
  tag=$(echo $flux | sed 's/e-/em/')
  [ -e "$ROOT/results/mdrex_results/musmooth${sm}_musparse${sp}_${shape}_alpha${tag}_gradS/x_opt.fits" ] && { echo "done: $job"; continue; }
  CL=${CLUSTERS[$((n % ${#CLUSTERS[@]}))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=10:00:00" -p "cluster='$CL'" -n "t1best_${shape}_a${tag}_sm${sm}_sp${sp}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --mu-smooth $sm --mu-sparse $sp --shape $shape --flux $flux --tag gradS"
done
echo "$n jobs submitted"
