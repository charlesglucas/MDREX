#!/bin/bash


SMS=(5e7)
SPS=(1e7)
DATA='HD_169142/2019-05-19/IRDIS/data/'
BAND='k1_k2'
DATARES='HD_169142-2019-05-19'
for sm in "${SMS[@]}"; do
  for sp in "${SPS[@]}"; do
    oarsub -l "gpu=1,walltime=15:00:00"\
      -p "cluster='vercors9'" \
      -n "realdata_${DATARES}_sm${sm}_sp${sp}" \
      "CUDA_VISIBLE_DEVICES=0 \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py \
      --data '$DATA' --band '$BAND' --datares '$DATARES' \
      --mu-smooth $sm --mu-sparse $sp"
  done
done
