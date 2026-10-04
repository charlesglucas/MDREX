#!/bin/bash

# Circle, gradS, (mu_smooth, mu_sparse) = (5e6, 1e5): Tables 1-2 / Figs. 3-4 (10 angles x 3 contrasts) and Fig. 2
# (distribution sets, alpha = 5e-6). Same runs as in synthetic_gradS.sh, relaunched for the circle only.
# output: results/mdrex_results/musmooth5e6_musparse1e5_circle_gradS/x_opt.fits
#         results/distributions_5em6/circle_musmooth5e6_musparse1e5_gradS/x_opt.fits
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results
oarsub -l "gpu=1,walltime=15:00:00" -p "cluster='vercors14'" -n "gradS_shape_circle_sm5e6_sp1e5" \
  "CUDA_VISIBLE_DEVICES=0 $PY $RUN/syntheticdata_shape.py --mu-smooth 5e6 --mu-sparse 1e5 --shape circle --tag gradS"
oarsub -l "gpu=1,walltime=20:00:00" -p "cluster='vercors9'" -n "gradS_distrib_circle_sm5e6_sp1e5" \
  "CUDA_VISIBLE_DEVICES=0 $PY $RUN/syntheticdata_distributions.py --mu-smooth 5e6 --mu-sparse 1e5 --shape circle --tag gradS"
