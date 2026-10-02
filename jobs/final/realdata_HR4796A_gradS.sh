#!/bin/bash

# HR 4796A, couple (mu_smooth, mu_sparse) = (1e6, 1e6), then (1e6, 1e7), gradient through m_hat / C_hat with the shrinkage
# coefficient not differentiated (gradS), band h2_h3 (lambda = 1.593 / 1.667 um), ftol 0.
# output: results/realdata/HR_4796A-2015-02-03-gradS-ftol0/ (couples already computed are skipped)
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
oarsub -l "gpu=1,walltime=15:00:00"\
  -p "cluster='vercors14'" \
  -n "realdata_gradS_HR_4796A_sm1e6_sp1e7" \
  "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
    --data 'HR_4796/2015-02-03/IRDIS/data/' --band 'h2_h3' --datares 'HR_4796A-2015-02-03-gradS' \
    --mu-smooth 1e6 --mu-sparse 1e7 --ftol 0"
