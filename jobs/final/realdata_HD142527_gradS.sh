#!/bin/bash

# HD 142527 (HIP 78092), 2016-03-26, IRDIS K12 (lambda = 2.110 / 2.251 um -> coronagraph k1_k2), 1408 frames of
# DIT 0.84 s, 65 deg of field rotation, gradS, ftol 0.
#   1. preparation (CPU): frames averaged by bins of 8 (1408 -> 176 frames), layout of the other targets
#      -> data/real_data/DISKS_IRDIS_CHARLES/HD_142527/2016-03-26-bin8/data/   (skipped if already done)
#   2. grid of couples mu_smooth 1e6 / 1e7 / 1e8 x mu_sparse 1e5 / 1e6 / 1e7, one GPU job per mu_smooth (couples in
#      sequence, existing ones skipped), started when the preparation has finished (oarsub -a)
#      -> results/realdata/HD_142527-2016-03-26-gradS-ftol0/
# GPU jobs spread over vercors14 and vercors9.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results/realdata_grid_split.py
DATA=HD_142527/2016-03-26-bin8/data
CLUSTERS=(vercors14 vercors9)
dep=""
if ! ls $ROOT/data/real_data/DISKS_IRDIS_CHARLES/$DATA/*reduced_master_cube*.fits >/dev/null 2>&1; then
  out=$(oarsub -l "nodes=1,walltime=3:00:00" -n "prep_HD142527" \
    "cd $ROOT && $PY $ROOT/scripts/run_results/prepare_realdata_hd142527.py --bin 8")
  echo "$out"
  id=$(echo "$out" | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2)
  [ -z "$id" ] && { echo "preparation job not submitted"; exit 1; }
  dep="-a $id"
fi
n=0
for sm in 1e6 1e7 1e8; do
  CL=${CLUSTERS[$((n % ${#CLUSTERS[@]}))]}; n=$((n + 1))
  oarsub $dep -l "gpu=1,walltime=30:00:00" -p "cluster='$CL'" -n "realdata_gradS_HD142527_sm${sm}" \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data '$DATA' --band 'k1_k2' --datares 'HD_142527-2016-03-26-gradS' \
      --mu-smooth $sm --mu-sparse 1e5 1e6 1e7 --ftol 0"
done
