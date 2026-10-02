#!/bin/bash

# More sparsity for RX J1615 and HD 106906 (K12), gradS, ftol 0: intermediate mu_sparse values between the couple of
# the paper and the next decade, where the reconstruction vanishes (x = 0 at mu_sparse = 5e7 for RX J1615 and 1e7 for
# HD 106906 K12). One job per target, couples in sequence. Output: results/realdata/<datares>-gradS-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
CLUSTERS=(vercors14 vercors9)
n=0
#  datares                     data                                     band   mu_smooth mu_sparse
while read datares data band sm sps; do
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='$CL'" \
    -n "realdata_gradS_sparser_${datares}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data '$data' --band '$band' --datares '${datares}-gradS' \
      --mu-smooth $sm --mu-sparse ${sps//,/ } --ftol 0"
done <<'LIST'
RX_J161533255-2019-05-18     RX_J161533255/2019-05-18/IRDIS/data/        h2_h3  5e7 1e7,2e7
HD_106906-2016-03-28-k1_k2   HD_106906/2016-03-28/IRDIS/k1_k2/data/      k1_k2  1e6 2e6,5e6
LIST
