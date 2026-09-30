#!/bin/bash

# MC-SURE on real data (scripts/run_results/realdata_sure.py): x(y) (reused from results/realdata when available)
# and x(y + xi delta) for each couple; one job per mu_smooth. The best couple is then selected by
# scripts/plot_results/disk_rec_realdata_sure.py (MC-SURE with the pilot covariance).
# first run (ftol 1e-8, cold start, 1 probe): results/realdata_sure/RY_lup-2016-04-16
# now: tighter tolerance + warm start of x(y + xi delta) from x(y), less optimizer noise in the divergence (1 probe)
SMS=(1e5 1e6 1e7 5e7)
SPS="5e4 1e5 5e5"
SEEDS=(42)
FTOL=1e-12
DATA='RY_lup/2016-04-16/IRDIS/data/'
BAND='h2_h3'   # RY Lup observed in DB_H23 (lambda = 1.593 / 1.667 um)
DATARES='RY_lup-2016-04-16'
for seed in "${SEEDS[@]}"; do
for sm in "${SMS[@]}"; do
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='vercors14'" \
    -n "realdata_sure_${DATARES}_sm${sm}_seed${seed}" \
    "CUDA_VISIBLE_DEVICES=0 \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
    /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_sure.py \
    --data '$DATA' --band '$BAND' --datares '$DATARES' \
    --mu-smooth $sm --mu-sparse $SPS --ftol $FTOL --warm-start --seed $seed"
done
done
