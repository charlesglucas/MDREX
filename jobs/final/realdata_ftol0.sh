#!/bin/bash

# Fig. 6 (real data) recomputed with ftol = 0 (xtol = 1e-6): on RY Lup, ftol = 1e-8 stops far from convergence
# (x at 47% of the converged solution, max underestimated by ~40%, see results/convergence/).
# One job per target, with the couple of the paper; output: results/realdata/<datares>-ftol0/
# Band (coronagraph transmission) from the wavelengths of each observation (*-lam.fits):
#   h2_h3: lambda = 1.593 / 1.667 um, k1_k2: lambda = 2.110 / 2.251 um
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
#  datares                     data                                     band   mu_smooth mu_sparse
while read datares data band sm sp; do
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='vercors14'" \
    -n "realdata_ftol0_${datares}_sm${sm}_sp${sp}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data '$data' --band '$band' --datares '$datares' \
      --mu-smooth $sm --mu-sparse $sp --ftol 0"
done <<'LIST'
HR_4796A-2015-02-03          HR_4796/2015-02-03/IRDIS/data/              h2_h3  1e6 5e6
PDS_70-2018-02-24            PDS_70/2018-02-24/IRDIS/data/               k1_k2  1e7 1e6
RY_lup-2016-04-16            RY_lup/2016-04-16/IRDIS/data/               h2_h3  1e7 1e5
SAO_206462-2015-05-15        SAO_206462/2015-05-15/IRDIS/data/           k1_k2  1e6 1e6
HD_169142-2019-05-19         HD_169142/2019-05-19/IRDIS/data/            k1_k2  5e7 5e5
RX_J161533255-2019-05-18     RX_J161533255/2019-05-18/IRDIS/data/        h2_h3  5e7 5e6
HD_106906-2016-03-28-h2_h3   HD_106906/2016-03-28/IRDIS/h2_h3/data/      h2_h3  1e6 1e6
HD_106906-2016-03-28-k1_k2   HD_106906/2016-03-28/IRDIS/k1_k2/data/      k1_k2  1e6 1e6
LIST
