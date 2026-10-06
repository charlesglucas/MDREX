#!/bin/bash

# Diagnostic run for AB Aurigae (non-finite statistics): one couple, same algorithm, MDREX_DEBUG=1 prints ||x||,
# max x, f and ||g|| at each evaluation of the objective. Results go to a separate folder (-debug).
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
oarsub -l "gpu=1,walltime=4:00:00" -p "cluster='vercors14'" -n "debug_AB_Aur" \
  "MDREX_DEBUG=1 CUDA_VISIBLE_DEVICES=0 $PY $RUN \
    --data 'AB_AURIGAE/2020-01-18/IRDIS/data/' --band 'k1_k2' --datares 'AB_AURIGAE-2020-01-18-debug' \
    --mu-smooth 1e8 --mu-sparse 1e5 --ftol 0"
