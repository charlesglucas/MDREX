#!/bin/bash

# HD 106906, IRDIS K12, gradS, ftol 0: so far only mu_smooth = 1e6 (mu_sparse 1e5 ... 1e7); the background shows
# speckle-like blobs of a few pixels -> stronger smoothing, mu_smooth 5e6 / 1e7 / 5e7 x mu_sparse 1e6 / 2e6 / 5e6
# (around the couple of Fig. 6, (1e6, 5e6)). One job per mu_smooth, spread over vercors14 and vercors9.
# output: results/realdata/HD_106906-2016-03-28-k1_k2-gradS-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
CLUSTERS=(vercors14 vercors9)
n=0
for sm in 5e6 1e7 5e7; do
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=15:00:00" -p "cluster='$CL'" -n "realdata_gradS_HD106906K_sm${sm}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data 'HD_106906/2016-03-28/IRDIS/k1_k2/data/' --band 'k1_k2' --datares 'HD_106906-2016-03-28-k1_k2-gradS' \
      --mu-smooth $sm --mu-sparse 1e6 2e6 5e6 --ftol 0"
done
