#!/bin/bash

# The 8 real-data observations of Fig. 6 with the gradient through m_hat / C_hat, shrinkage coefficient not
# differentiated, ftol 0.
# For each target: couple of the paper (mu_smooth, mu_sparse) and mu_sparse /10 and x10. One job per target, couples in sequence;
# couples already computed are skipped. output: results/realdata/<datares>-gradS-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
CLUSTERS=(vercors14 vercors9)
n=0
#  datares                     data                                     band   mu_smooth mu_sparse (/10 x1 x10)
while read datares data band sm sps; do
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='$CL'" \
    -n "realdata_gradS_${datares}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data '$data' --band '$band' --datares '${datares}-gradS' \
      --mu-smooth $sm --mu-sparse ${sps//,/ } --ftol 0"
done <<'LIST'
HR_4796A-2015-02-03          HR_4796/2015-02-03/IRDIS/data/              h2_h3  1e6 5e5,5e6,5e7
PDS_70-2018-02-24            PDS_70/2018-02-24/IRDIS/data/               k1_k2  1e7 1e5,1e6,1e7
RY_lup-2016-04-16            RY_lup/2016-04-16/IRDIS/data/               h2_h3  1e7 1e5,1e6,1e7
SAO_206462-2015-05-15        SAO_206462/2015-05-15/IRDIS/data/           k1_k2  1e6 1e5,1e6,1e7
HD_169142-2019-05-19         HD_169142/2019-05-19/IRDIS/data/            k1_k2  5e7 5e4,5e5,5e6
RX_J161533255-2019-05-18     RX_J161533255/2019-05-18/IRDIS/data/        h2_h3  5e7 5e5,5e6,5e7
HD_106906-2016-03-28-h2_h3   HD_106906/2016-03-28/IRDIS/h2_h3/data/      h2_h3  1e6 1e5,1e6,1e7
HD_106906-2016-03-28-k1_k2   HD_106906/2016-03-28/IRDIS/k1_k2/data/      k1_k2  1e6 1e5,1e6,1e7
LIST
