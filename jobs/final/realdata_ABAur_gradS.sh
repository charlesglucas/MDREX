#!/bin/bash

# AB Aurigae, 2020-01-18, IRDIS K12 (lambda = 2.110 / 2.251 um -> coronagraph k1_k2), gradS (gradient through
# m_hat / C_hat, shrinkage not differentiated), ftol 0. New target: small grid of couples,
# mu_smooth 1e6 / 1e7 x mu_sparse 1e5 / 1e6 / 1e7, one job per mu_smooth (couples in sequence, existing ones skipped).
# dxabs = 1e-3: very bright target (data ~8x and PSF ~3x the other targets), the default first step of VMLMB
# (norm 1) made the statistics non-finite. Non-finite trial steps are rejected by the line search (f = +inf).
# output: results/realdata/AB_AURIGAE-2020-01-18-gradS-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
CLUSTERS=(vercors14 vercors9)
n=0
for sm in 1e6 1e7; do
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=20:00:00" -p "cluster='$CL'" -n "realdata_gradS_AB_Aur_sm${sm}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data 'AB_AURIGAE/2020-01-18/IRDIS/data/' --band 'k1_k2' --datares 'AB_AURIGAE-2020-01-18-gradS' \
      --mu-smooth $sm --mu-sparse 1e5 1e6 1e7 --ftol 0 --dxabs 1e-3"
done
