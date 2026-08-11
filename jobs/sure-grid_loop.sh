#!/bin/bash

cd /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO

export GPUS=(0 1 2 3 4)
i=0
for sm in 1e3 1e4 1e5 1e6 1e7; do
  for sp in 1e3 1e4 1e5 1e6 1e7; do
    gpu=${GPUS[i % ${#GPUS[@]}]}
    CUDA_VISIBLE_DEVICES=$gpu ./.venv/bin/python scripts/disk_rec_exomildloss_reg_vmlmb_hierarchic_sure_gpu.py \
      --mu-smooth $sm --mu-sparse $sp \
      --out results/result_musmooth${sm}_musparse${sp}.npz &
    i=$((i+1))
  done
done
wait

