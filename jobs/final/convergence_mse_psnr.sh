#!/bin/bash

# Is the gap between the MSE minimum and the PSNR maximum due to non-converged reconstructions?
# convergence_test.py: one configuration (angle 0) reconstructed from x = 0 with the stopping tolerances (ftol:xtol)
#   1e-8:1e-6 (default, used for the grids), 0:1e-8 and 0:1e-10 (gtol 1e-5 unchanged), for the MSE minimum and the
#   PSNR maximum of the angle-0 grids:
#   spiral 5e-6: MSE min (1e7, 3e5), PSNR max (3e6, 1e5); circle 1e-5: MSE min (1e6, 3e5), PSNR max (1e7, 1e5)
# output: results/convergence/<shape>_angle0_alpha<flux>_musmooth<mu>_musparse<mu>/ftol<v>_xtol<v>.npz
# Jobs spread alternately over vercors14 and vercors9.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
CLUSTERS=(vercors14 vercors9)
n=0
for job in spiral:5e-6:1e7:3e5 spiral:5e-6:3e6:1e5 circle:1e-5:1e6:3e5 circle:1e-5:1e7:1e5; do
  IFS=: read shape flux sm sp <<< "$job"
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=24:00:00" -p "cluster='$CL'" -n "conv_${shape}_${flux}_sm${sm}_sp${sp}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_results/convergence_test.py --shape $shape --flux $flux \
      --angle 0 --mu-smooth $sm --mu-sparse $sp --tols 1e-8:1e-6 0:1e-8 0:1e-10"
done
