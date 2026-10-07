#!/bin/bash

# Table 1 with hyperparameters selected by MC-SURE (C^y) for each geometry, contrast and parallactic angle (gradS).
# For each (shape, alpha, angle): 3 x 3 grid (mu_smooth x mu_sparse), each couple = x(y) + x(y + xi delta) (divergence).
#   ellipse: mu_smooth 1e6 / 1e7 / 1e8 x mu_sparse 3e5 / 1e6 / 3e6   (optimum 1e6 on the gradS SURE grids)
#   spiral : mu_smooth 1e6 / 1e7 / 1e8 x mu_sparse 3e4 / 1e5 / 3e5   (optimum 1e5 on the gradS spiral grid)
#   circle : same as the spiral (no gradS grid for the circle)
# One job per (shape, alpha, angle) = 90 jobs, the 9 couples in sequence; couples already computed are skipped
# (the script can be relaunched to complete). Jobs spread alternately over vercors14 and vercors9.
# Usage: bash table1_sure_gradS.sh [shape ...]   (default: medium_ellipse spiral circle)
# output: results/grids_111111111111/grid_sure_<shape>_alpha<1em6|5em6|1em5>_angle<angle>_gradS/musmooth<mu>_musparse<mu>.npz
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/syntheticdata_sure_gpu.py
CLUSTERS=(vercors14 vercors9)
SHAPES=("${@:-medium_ellipse spiral circle}")
n=0
for shape in ${SHAPES[@]}; do
  case $shape in
    medium_ellipse) SPS="3e5 1e6 3e6";;
    *)              SPS="3e4 1e5 3e5";;
  esac
  for flux in 1e-6 5e-6 1e-5; do
    tag=$(echo $flux | sed 's/e-/em/')
    for angle in 0 36 72 108 144 180 216 252 288 324; do
      dir=$ROOT/results/grids_111111111111/grid_sure_${shape}_alpha${tag}_angle${angle}_gradS
      cmd=""
      for sm in 1e6 1e7 1e8; do
        for sp in $SPS; do
          f=$dir/musmooth$(awk "BEGIN{printf \"%.1f\", $sm}")_musparse$(awk "BEGIN{printf \"%.1f\", $sp}").npz
          [ -e "$f" ] && continue
          cmd="$cmd CUDA_VISIBLE_DEVICES=0 $PY $RUN --mu-smooth $sm --mu-sparse $sp --flux $flux --shape $shape --angle $angle --tag gradS;"
        done
      done
      [ -z "$cmd" ] && { echo "complete: $dir"; continue; }
      CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
      oarsub -l "gpu=1,walltime=10:00:00" -p "cluster='$CL'" -n "t1sure_${shape}_a${tag}_ang${angle}" "$cmd"
    done
  done
done
echo "$n jobs submitted"
