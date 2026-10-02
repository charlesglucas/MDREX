#!/bin/bash

# Synthetic results of the paper with the gradient through m_hat / C_hat, shrinkage coefficient not differentiated
# (gradS), default tolerances (as the gradS MC-SURE grids), same couples as in the paper.
#   Tables 1-2, Figs. 3-4: 10 angles x 3 contrasts per shape
#     ellipse (1e6, 1e6) [Table 1] and (5e6, 1e5) [Table 2]; spiral and circle (5e6, 1e5) [Tables 1 and 2]
#     output: results/mdrex_results/musmooth<mu>_musparse<mu>_<shape>_gradS/x_opt.fits
#   Fig. 2: distribution sets, alpha = 5e-6, 10 angles
#     ellipse, spiral, circle (5e6, 1e5) and ellipse (1e6, 1e6)
#     output: results/distributions_5em6/<shape>_musmooth<mu>_musparse<mu>_gradS/x_opt.fits
# Jobs spread alternately over vercors14 and vercors9.
PY=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/.venv/bin/python
RUN=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/scripts/run_results
CLUSTERS=(vercors14 vercors9)
n=0
submit() {  # submit <name> <walltime> <command>
  CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
  oarsub -l "gpu=1,walltime=$2" -p "cluster='$CL'" -n "$1" "CUDA_VISIBLE_DEVICES=0 $3"
}
for job in medium_ellipse:1e6:1e6 medium_ellipse:5e6:1e5 spiral:5e6:1e5 circle:5e6:1e5; do
  IFS=: read shape sm sp <<< "$job"
  submit "gradS_shape_${shape}_sm${sm}_sp${sp}" 15:00:00 \
    "$PY $RUN/syntheticdata_shape.py --mu-smooth $sm --mu-sparse $sp --shape $shape --tag gradS"
done
for job in medium_ellipse:5e6:1e5 spiral:5e6:1e5 circle:5e6:1e5 medium_ellipse:1e6:1e6; do
  IFS=: read shape sm sp <<< "$job"
  submit "gradS_distrib_${shape}_sm${sm}_sp${sp}" 20:00:00 \
    "$PY $RUN/syntheticdata_distributions.py --mu-smooth $sm --mu-sparse $sp --shape $shape --tag gradS"
done
echo "$n jobs submitted"
