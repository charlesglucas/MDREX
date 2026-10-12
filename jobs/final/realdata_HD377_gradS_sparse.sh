#!/bin/bash

# HD 377, gradS, ftol 0: the disk appears around (mu_smooth, mu_sparse) = (1e6, 1e5) -> less sparsity,
# mu_smooth 1e6 x mu_sparse 1e4 / 2e4 / 5e4. output: results/realdata/HD_377-2016-10-13-gradS-ftol0/
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results/realdata_grid_split.py
oarsub -l "gpu=1,walltime=20:00:00" -p "cluster='vercors14'" -n "realdata_gradS_HD377_sparse" \
  "CUDA_VISIBLE_DEVICES=0 $PY $RUN \
    --data 'HD_377/2016-10-13/data/' --band 'h2_h3' --datares 'HD_377-2016-10-13-gradS' \
    --mu-smooth 1e6 --mu-sparse 1e4 2e4 5e4 --ftol 0"
