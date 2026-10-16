#!/bin/bash

# HD 142527 (HIP 78092), 2016-03-26, IRDIS K12 (lambda = 2.110 / 2.251 um -> coronagraph k1_k2), all the 1408 frames
# of DIT 0.84 s (two sequences of 704 frames, 65 deg of field rotation), gradS, ftol 0.
#   1. links to the original files (1024 x 1024 frames, all the frames, original centring, data unchanged) under the
#      names read by load_folder, created on the frontend (prepare_realdata_hd142527.py, instantaneous)
#      -> data/real_data/DISKS_IRDIS_CHARLES/HD_142527/2016-03-26/data/
#   2. grid of couples mu_smooth 1e6 / 1e7 / 1e8 x mu_sparse 1e5 / 1e6 / 1e7, ONE GPU job PER COUPLE (1408 frames:
#      about 16 times the cost of a target with ~90 frames), batch of 16 patches for the ExoMILD engine (memory)
#      -> results/realdata/HD_142527-2016-03-26-gradS-ftol0/
# GPU jobs spread over vercors14 and vercors9.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/realdata_grid_split.py
DATA=HD_142527/2016-03-26/data
CLUSTERS=(vercors14 vercors9)
$PY $ROOT/scripts/run_results/prepare_realdata_hd142527.py || exit 1
n=0
for sm in 1e6 1e7 1e8; do
  for sp in 1e5 1e6 1e7; do
    CL=${CLUSTERS[$((n % ${#CLUSTERS[@]}))]}; n=$((n + 1))
    oarsub -l "gpu=1,walltime=48:00:00" -p "cluster='$CL'" -n "realdata_gradS_HD142527_sm${sm}_sp${sp}" \
      "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN \
        --data '$DATA' --band 'k1_k2' --datares 'HD_142527-2016-03-26-gradS' \
        --mu-smooth $sm --mu-sparse $sp --ftol 0 --batch-size 16"
  done
done
