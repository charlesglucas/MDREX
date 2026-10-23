#!/bin/bash

# Fine grid between the MSE minimum (1e7, 3e5) and the PSNR maximum (3e6, 1e5) of the spiral at alpha = 5e-6, angle 0
# (step ~1/6 decade): mu_smooth 2e6 3e6 4.6e6 6.8e6 1e7 1.5e7 x mu_sparse 6.8e4 1e5 1.5e5 2.2e5 3e5 4.6e5.
# Same runs as the angle-0 grids (syntheticdata_sure_gpu.py, x(y) and x(y + xi delta)), same folder: couples already
# computed are skipped. 4 jobs (the couples in sequence), vercors14 / vercors9.
# output: results/grids_111111111111/grid_sure_spiral_alpha5em6_angle0_gradS/musmooth<mu>_musparse<mu>.npz
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/syntheticdata_sure_gpu.py
# CONV=1: converged reconstructions (ftol 0, xtol 1e-8, convergence test), folder ..._angle0_gradSconv, 6 jobs
SMS="2e6 3e6 4.6e6 6.8e6 1e7 1.5e7"; SPS="6.8e4 1e5 1.5e5 2.2e5 3e5 4.6e5"
TAG=gradS; EXTRA=""; NJ=4
[ "${CONV:-0}" = 1 ] && { TAG=gradSconv; EXTRA="--ftol 0 --xtol 1e-8"; NJ=6; }
DIR=$ROOT/results/grids_111111111111/grid_sure_spiral_alpha5em6_angle0_$TAG
CLUSTERS=(vercors14 vercors9)
cmds=("" "" "" "" "" ""); k=0
for sm in $SMS; do
  for sp in $SPS; do
    f=$DIR/musmooth$(awk "BEGIN{printf \"%.1f\", $sm}")_musparse$(awk "BEGIN{printf \"%.1f\", $sp}").npz
    [ -e "$f" ] && continue
    cmds[$((k % NJ))]="${cmds[$((k % NJ))]} CUDA_VISIBLE_DEVICES=0 $PY $RUN --mu-smooth $sm --mu-sparse $sp --flux 5e-6 --shape spiral --angle 0 --tag $TAG $EXTRA;"
    k=$((k + 1))
  done
done
for i in $(seq 0 $((NJ - 1))); do
  [ -z "${cmds[$i]}" ] && continue
  oarsub -l "gpu=1,walltime=30:00:00" -p "cluster='${CLUSTERS[$((i % 2))]}'" -n "finegrid_${TAG}_spiral5em6_$i" "cd $ROOT && ${cmds[$i]}" | grep OAR_JOB_ID
done
echo "$k couples submitted"
