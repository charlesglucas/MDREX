#!/bin/bash

# MC-SURE grid at angle 0 with fixed shrinkage coefficients (Algorithm 1: rho computed once on the data, constant in the
# objective and its gradient, default of reconstruction/mdrex.py since 5eb1c72), float32, VMLMB ftol 0 / xtol 1e-10
# (stop at the precision of the objective). Square 7 x 7 grid (mu_smooth 1e6..1e9, mu_sparse 3e4..3e7, half-decade step).
#   -> results/grids_111111111111/grid_sure_<shape>_alpha<..>_angle0_rhofix/
# Usage: bash grid_rhofixed.sh <shape> <flux> [number of jobs, default 6]   e.g. bash grid_rhofixed.sh spiral 1e-6
# Each couple is limited to 4 h (timeout: a hung start on a node does not block the whole job).
# Couples already computed are skipped. Jobs spread over vercors14 / vercors9, or, with CLUSTERS="vercors10 vercors11 ...",
# submitted to any free GPU of these clusters.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
if [ -n "$CLUSTERS" ]; then
  PROP="cluster IN ($(for c in $CLUSTERS; do printf "'%s'," $c; done | sed 's/,$//'))"
  PROPS=("$PROP" "$PROP")
else
  PROPS=("cluster='vercors14'" "cluster='vercors9'")
fi
shape=$1; flux=$2; nj=${3:-6}
tag=$(echo $flux | sed 's/e-/em/')
dir=$ROOT/results/grids_111111111111/grid_sure_${shape}_alpha${tag}_angle0_rhofix
cmds=(); k=0
# GRID_SMOOTH / GRID_SPARSE (env, space-separated): other grid (e.g. fine grid around the minima), same folder
for sm in ${GRID_SMOOTH:-1e6 3e6 1e7 3e7 1e8 3e8 1e9}; do
  for sp in ${GRID_SPARSE:-3e4 1e5 3e5 1e6 3e6 1e7 3e7}; do
    f=$dir/musmooth$(awk "BEGIN{printf \"%.1f\", $sm}")_musparse$(awk "BEGIN{printf \"%.1f\", $sp}").npz
    [ -e "$f" ] && continue
    cmds[$((k % nj))]="${cmds[$((k % nj))]} CUDA_VISIBLE_DEVICES=0 timeout 4h $PY $ROOT/scripts/run_results/syntheticdata_sure_gpu.py --mu-smooth $sm --mu-sparse $sp --flux $flux --shape $shape --angle 0 --tag rhofix --ftol 0 --xtol 1e-10;"
    k=$((k + 1))
  done
done
for i in $(seq 0 $((nj - 1))); do
  [ -n "${cmds[$i]}" ] && oarsub -l "gpu=1,walltime=48:00:00" -p "${PROPS[$((i % 2))]}" \
    -n "gridrf_${shape}_${tag}_$i" "cd $ROOT && ${cmds[$i]}" | grep -o "OAR_JOB_ID=[0-9]*"
done
