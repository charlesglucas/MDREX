#!/bin/bash


SMS=(1e6)
SPS=(1e6 5e6 1e7)
DATA='HR_4796/2015-02-03/IRDIS/data/'
BAND='h2_h3'
DATARES='HR_4796A-2015-02-03'
for sm in "${SMS[@]}"; do
  for sp in "${SPS[@]}"; do
    oarsub -l "gpu=1,walltime=15:00:00"\
      -p "cluster='vercors14'" \
      -n "realdata_${DATARES}_sm${sm}_sp${sp}" \
      "CUDA_VISIBLE_DEVICES=0 \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
      /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py \
      --data '$DATA' --band '$BAND' --datares '$DATARES' \
      --mu-smooth $sm --mu-sparse $sp"
  done
done
