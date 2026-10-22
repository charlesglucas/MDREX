#!/bin/bash

# Convergence curves of the reconstruction: objective, projected gradient and wavelength-averaged iterate recorded at
# each VMLMB iteration (convergence_test.py --trace), tolerances 1e-8:1e-6 (used for the grids and Table 1) and
# 0:1e-10, angle 0, for the MSE minimum and the PSNR maximum of the angle-0 grids (spiral 5e-6, circle 1e-5) and for
# the ellipse 5e-6 (3e6, 1e6).
# output: results/convergence/<case>/trace_ftol<v>_xtol<v>.npz
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
CLUSTERS=(vercors14 vercors9)
n=0
for job in spiral:5e-6:1e7:3e5 spiral:5e-6:3e6:1e5 circle:1e-5:1e6:3e5 circle:1e-5:1e7:1e5 medium_ellipse:5e-6:3e6:1e6; do
  IFS=: read shape flux sm sp <<< "$job"
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=24:00:00" -p "cluster='$CL'" -n "trace_${shape}_${flux}_sm${sm}_sp${sp}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_results/convergence_test.py --shape $shape --flux $flux \
      --angle 0 --mu-smooth $sm --mu-sparse $sp --tols 1e-8:1e-6 0:1e-10 --trace"
done
