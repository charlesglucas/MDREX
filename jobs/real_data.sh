#!/bin/bash


oarsub -l "gpu=1,walltime=8:00:00"\
  -p "cluster='vercors14'" \
  -n "realdata" \
  "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/disk_rec_exomildloss_reg_realdata.py "
