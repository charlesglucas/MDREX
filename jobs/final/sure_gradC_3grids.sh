#!/bin/bash

# MC-SURE grids with the gradient through m_hat / C_hat (Fig. 5), ellipse, the 3 contrasts,
# 6 x 6 grid mu_smooth 1e4..1e9 x mu_sparse 1e4..1e9. Couples already computed (file present) are skipped, so this
# script also completes alpha = 5e-6 (test_gradC.sh, sure_gradC_extend.sh, sure_gradC_grid.sh).
# Jobs spread alternately over vercors14 and vercors9.
# output: results/grids_111111111111/grid_sure_medium_ellipse_alpha<1em6|5em6|1em5>_gradC/
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN=$ROOT/scripts/run_results
CLUSTERS=(vercors14 vercors9)
n=0
for flux in 1e-6 5e-6 1e-5; do
  tag=$(echo $flux | sed 's/e-/em/')
  dir=$ROOT/results/grids_111111111111/grid_sure_medium_ellipse_alpha${tag}_gradC
  for sm in 1e4 1e5 1e6 1e7 1e8 1e9; do
    for sp in 1e4 1e5 1e6 1e7 1e8 1e9; do
      f=$dir/musmooth$(awk "BEGIN{printf \"%.1f\", $sm}")_musparse$(awk "BEGIN{printf \"%.1f\", $sp}").npz
      if [ -e "$f" ]; then echo "skip $f"; continue; fi
      CL=${CLUSTERS[$((n % 2))]}; n=$((n + 1))
      oarsub -l "gpu=1,walltime=18:00:00"\
        -p "cluster='$CL'" \
        -n "gradC_sure_medium_ellipse_alpha${flux}_sm${sm}_sp${sp}" \
        "CUDA_VISIBLE_DEVICES=0 $PY $RUN/syntheticdata_sure_gpu.py --mu-smooth $sm --mu-sparse $sp --flux $flux \
          --shape medium_ellipse --tag gradC"
    done
  done
done
echo "$n jobs submitted"
