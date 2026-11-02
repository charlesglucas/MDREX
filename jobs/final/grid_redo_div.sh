#!/bin/bash
# MC-SURE of an existing grid recomputed with a Monte Carlo vector drawn on the CPU (the CUDA generator gives different
# vectors on different GPU models for the same seed, so that the couples of a grid computed on different clusters did not
# share the same vector). The reconstructions x_hat are kept; only the perturbed reconstructions are redone.
#   bash grid_redo_div.sh <shape> <flux> [nj]   grid_sure_<shape>_alpha<..>_angle0_rhofix -> ..._angle0_rhofixcpu
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P="cluster IN ('vercors10','vercors11','vercors12','vercors13')"
shape=$1; flux=$2; nj=${3:-6}
tag=$(echo $flux | sed 's/e-/em/')
src=$ROOT/results/grids_111111111111/grid_sure_${shape}_alpha${tag}_angle0_rhofix
dst=${src}cpu
cmds=(); k=0
for f in $src/musmooth*_musparse*.npz; do
  b=$(basename $f .npz); sm=${b#musmooth}; sm=${sm%_musparse*}; sp=${b#*_musparse}
  [ -e "$dst/$b.npz" ] && continue
  cmds[$((k % nj))]="${cmds[$((k % nj))]} CUDA_VISIBLE_DEVICES=0 timeout 4h $PY $ROOT/scripts/run_results/syntheticdata_sure_gpu.py --mu-smooth $sm --mu-sparse $sp --flux $flux --shape $shape --angle 0 --tag rhofixcpu --ftol 0 --xtol 1e-10 --redo-div $f;"
  k=$((k + 1))
done
echo "$shape $flux: $k couples"
for i in $(seq 0 $((nj - 1))); do
  [ -n "${cmds[$i]}" ] && oarsub -l "gpu=1,walltime=48:00:00" -p "$P" -n "redodiv_${shape}_${tag}_$i" "cd $ROOT && ${cmds[$i]}" \
    | grep -o "OAR_JOB_ID=[0-9]*"
done
