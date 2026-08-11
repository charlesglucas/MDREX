#!/bin/bash


oarsub -l "gpu=1,walltime=20:00:00"\
  -p "cluster='vercors14'" \
  -n "exomild" \
  "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/main.py name="exomild" log_wandb=true data.train.num_workers=16 "


