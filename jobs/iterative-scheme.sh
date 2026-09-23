#!/bin/bash

SMS=(1e3 1e4 1e5 1e6 1e7)
SPS=(1e3 1e4 1e5 1e6 1e7)

for sm in "${SMS[@]}"; do
  for sp in "${SPS[@]}"; do

    oarsub -l "gpu=1,walltime=8:00:00" \
      -p "cluster='vercors18'" \
      -n "iter_sm${sm}_sp${sp}" \
      "CUDA_VISIBLE_DEVICES=0 \
       /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
       /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/disk_rec_exomildloss_reg_vmlmb_iterative_gpu.py \
       --mu-smooth $sm --mu-sparse $sp "

  done
done

