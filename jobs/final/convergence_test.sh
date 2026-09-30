#!/bin/bash

# Convergence test of VMLMB (ftol:xtol = 1e-8:1e-6 [current], 0:1e-6, 0:1e-8, 0:1e-10), from x = 0:
#   - synthetic: spiral, angle 0, alpha = 5e-6, (mu_smooth, mu_sparse) = (5e6, 1e5)   (couple of Table 1)
#   - real data: RY Lup, (mu_smooth, mu_sparse) = (1e7, 1e5)                        (couple of the paper)
# Relaunching resumes (tolerances already computed are skipped). Analysis: scripts/plot_results/convergence_test.py
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results

oarsub -l "gpu=1,walltime=20:00:00"\
  -p "cluster='vercors14'" \
  -n "convergence_test_spiral" \
  "CUDA_VISIBLE_DEVICES=0 $PY $RUN/convergence_test.py \
    --shape spiral --angle 0 --flux 5e-6 --mu-smooth 5e6 --mu-sparse 1e5"

oarsub -l "gpu=1,walltime=20:00:00"\
  -p "cluster='vercors14'" \
  -n "convergence_test_RY_lup" \
  "CUDA_VISIBLE_DEVICES=0 $PY $RUN/convergence_test_realdata.py \
    --data 'RY_lup/2016-04-16/IRDIS/data/' --band 'h2_h3' --datares 'RY_lup-2016-04-16' \
    --mu-smooth 1e7 --mu-sparse 1e5"
