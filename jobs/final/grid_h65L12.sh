#!/bin/bash
# Completion of the MD-REX PSNR grid on HIP 65426 (2017-02-06, IRDIS H2/H3), 12 learned projections (series h65L12,
# first launched by all_ckpt.sh with STAR=HIP_65426/2017-02-06: its single 24 h grid job was killed after 2 couples,
# ~10 h per couple). One job per missing couple (x_<log10 mu_smooth>_<log10 mu_sparse>.npz absent), so that every couple
# fits in the walltime; each job writes its own grid_<couple>.npz in results/sugar/<shape>_alpha<..>_angle0_h65L12_grid/.
# Usage: bash grid_h65L12.sh [shape flux]   (default: medium_ellipse 1e-5, the case of the DSM comparison)
#   GS / GP (env, log10): grid (default: that of the first job), WALL (env): walltime in hours (default 24)
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P=${P:-"cluster IN ('vercors9','vercors10','vercors11','vercors12','vercors13','vercors14','vercors16','vercors17','vercors18')"}
CKPT=${CKPT:-$ROOT/data/run/train/2026-11-12_08-52-53/exomild_H2_12/ckpt/ckpt_40000.pt}
TAG=${TAG:-h65L12}
STAR=${STAR:-HIP_65426/2017-02-06}
shape=${1:-medium_ellipse}; flux=${2:-1e-5}; tag=$(echo $flux | sed 's/e-/em/')
GS=${GS:-6 6.4771 7 7.4771 8}; GP=${GP:-5 5.4771 6 6.4771 7}; WALL=${WALL:-24}
dir=$ROOT/results/sugar/${shape}_alpha${tag}_angle0_${TAG}_grid
n=0
for ts in $GS; do
  for tp in $GP; do
    c=$(printf "%.3f_%.3f" $ts $tp)
    [ -e "$dir/x_$c.npz" ] && { echo "done: $c"; continue; }
    oarsub -l "gpu=1,walltime=$WALL:00:00" -p "$P" -n "grid${TAG}_${shape}_${tag}_$c" \
      "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar_iter.py --flux $flux --shape $shape \
       --angle 0 --hvp-step 1e-2 --normalize --rec-ftol 0 --rec-xtol 1e-10 --tag $TAG --ckpt $CKPT --star $STAR \
       --grid-smooth $ts --grid-sparse $tp --grid-name $c" | grep -o "OAR_JOB_ID=[0-9]*" | sed "s/^/$c: /"
    n=$((n + 1))
  done
done
echo "$n jobs submitted"
