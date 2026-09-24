#!/bin/bash


SMS=(1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9 1e10)
SPS=(1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9 1e10)
FLUXS=(1e-5)
shape='medium_ellipse'
for flux in "${FLUXS[@]}"; do
  for sm in "${SMS[@]}"; do
    for sp in "${SPS[@]}"; do
      oarsub -l "gpu=1,walltime=18:00:00"\
        -p "cluster='vercors9'" \
        -n "sure_${shape}_alpha${flux}_sm${sm}_sp${sp}" \
        "CUDA_VISIBLE_DEVICES=0 \
        /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python \
        /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/syntheticdata_sure_gpu.py \
        --mu-smooth $sm --mu-sparse $sp --flux $flux --shape $shape"
    done
  done
done
