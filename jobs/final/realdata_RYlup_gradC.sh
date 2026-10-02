#!/bin/bash

# RY Lup with the gradient through the estimated means / covariances restored (no torch.no_grad on fit_params, no
# detach of C_inv in get_log_likelihood, exact analytic gradient w.r.t. C_inv), band h2_h3, ftol 0.
# Compare with results/realdata/RY_lup-2016-04-16-ftol0/ (same runs without the gradient through m_hat, C_hat).
# output: results/realdata/RY_lup-2016-04-16-gradC-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
for couple in 1e7:1e6 1e6:1e7; do
  sm=${couple%%:*}; sp=${couple##*:}
  oarsub -l "gpu=1,walltime=15:00:00"\
    -p "cluster='vercors14'" \
    -n "realdata_RY_lup_gradC_sm${sm}_sp${sp}" \
    "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
      --data 'RY_lup/2016-04-16/IRDIS/data/' --band 'h2_h3' --datares 'RY_lup-2016-04-16-gradC' \
      --mu-smooth $sm --mu-sparse $sp --ftol 0"
done
