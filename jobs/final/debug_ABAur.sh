#!/bin/bash

# Diagnostic run for AB Aurigae (non-finite statistics): one couple, same algorithm, MDREX_DEBUG=1 prints ||x||,
# max x, f and ||g|| at each evaluation of the objective. f at x = 0 was finite on vercors9 and NaN on vercors14:
# the same run on both clusters, with the GPU model printed. Results go to separate folders (-debug-<cluster>).
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
for CL in vercors14 vercors9; do
  oarsub -l "gpu=1,walltime=4:00:00" -p "cluster='$CL'" -n "debug_AB_Aur_$CL" \
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; hostname; \
     MDREX_DEBUG=1 CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data 'AB_AURIGAE/2020-01-18/IRDIS/data/' --band 'k1_k2' --datares 'AB_AURIGAE-2020-01-18-debug-$CL' \
      --mu-smooth 1e8 --mu-sparse 1e5 --ftol 0"
done
