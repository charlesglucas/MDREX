#!/bin/bash

# All the synthetic experiments with converged reconstructions: ftol 0 (xtol 1e-6, gtol 1e-5), as for the real data.
#   A. MC-SURE grids at angle 0, square 7 x 7 grid (mu_smooth 1e6..1e9, mu_sparse 3e4..3e7, half-decade step), 3 shapes
#      x 3 contrasts, + fine grid of the spiral at 5e-6 (between the MSE minimum and the PSNR maximum), 3 jobs per case
#      -> results/grids_111111111111/grid_sure_<shape>_alpha<..>_angle0_gradSf0/
#   B. Table 1, 10 angles, couples maximizing the PSNR (one contrast per job)
#      -> results/mdrex_results/musmooth<mu>_musparse<mu>_<shape>_alpha<..>_ftol0_gradS/
#   C. Fig. 2, patch distributions at 5e-6 with the couples of Table 1 -> results/distributions_5em6/<..>_ftol0_gradS/
#   D. MC-SUGAR with fixed-point iterations on the covariance (sugar_fixedpoint.sh, REC_FTOL=0)
# Couples already computed are skipped. Jobs spread over vercors14 / vercors9.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
CLUSTERS=(vercors14 vercors9)
n=0
sub() {  # sub <name> <walltime> <command>
  oarsub -l "gpu=1,walltime=$2" -p "cluster='${CLUSTERS[$((n % 2))]}'" -n "$1" "cd $ROOT && $3" | grep -o "OAR_JOB_ID=[0-9]*"
  n=$((n + 1))
}
# A. grids
SMS="1e6 3e6 1e7 3e7 1e8 3e8 1e9"; SPS="3e4 1e5 3e5 1e6 3e6 1e7 3e7"
for shape in medium_ellipse spiral circle; do
  for flux in 1e-6 5e-6 1e-5; do
    tag=$(echo $flux | sed 's/e-/em/')
    dir=$ROOT/results/grids_111111111111/grid_sure_${shape}_alpha${tag}_angle0_gradSf0
    couples=""
    for sm in $SMS; do for sp in $SPS; do couples="$couples $sm:$sp"; done; done
    if [ $shape = spiral ] && [ $flux = 5e-6 ]; then
      for sm in 2e6 3e6 4.6e6 6.8e6 1e7 1.5e7; do for sp in 6.8e4 1e5 1.5e5 2.2e5 3e5 4.6e5; do couples="$couples $sm:$sp"; done; done
    fi
    nj=3; [ $shape = spiral ] && [ $flux = 5e-6 ] && nj=5
    cmds=(); k=0
    for c in $(echo $couples | tr ' ' '\n' | sort -u); do
      IFS=: read sm sp <<< "$c"
      f=$dir/musmooth$(awk "BEGIN{printf \"%.1f\", $sm}")_musparse$(awk "BEGIN{printf \"%.1f\", $sp}").npz
      [ -e "$f" ] && continue
      cmds[$((k % nj))]="${cmds[$((k % nj))]} CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_results/syntheticdata_sure_gpu.py --mu-smooth $sm --mu-sparse $sp --flux $flux --shape $shape --angle 0 --tag gradSf0 --ftol 0;"
      k=$((k + 1))
    done
    for i in $(seq 0 $((nj - 1))); do [ -n "${cmds[$i]}" ] && sub "gridf0_${shape}_${tag}_$i" 60:00:00 "${cmds[$i]}"; done
  done
done
# B. Table 1
for job in medium_ellipse:1e-6:3e7:1e6 medium_ellipse:5e-6:3e6:1e6 medium_ellipse:1e-5:2.67e6:1.28e6 \
           spiral:1e-6:3e7:3e5 spiral:5e-6:5e6:1e5 spiral:1e-5:3e6:1e5 \
           circle:1e-6:3e7:3e5 circle:5e-6:3e6:3e5 circle:1e-5:5e6:1e5; do
  IFS=: read shape flux sm sp <<< "$job"; tag=$(echo $flux | sed 's/e-/em/')
  [ -e "$ROOT/results/mdrex_results/musmooth${sm}_musparse${sp}_${shape}_alpha${tag}_ftol0_gradS/x_opt.fits" ] && continue
  sub "t1f0_${shape}_${tag}" 24:00:00 "CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_results/syntheticdata_shape.py --mu-smooth $sm --mu-sparse $sp --shape $shape --flux $flux --ftol 0 --tag gradS"
done
# C. Fig. 2
for job in medium_ellipse:3e6:1e6 spiral:5e6:1e5 circle:3e6:3e5; do
  IFS=: read shape sm sp <<< "$job"
  [ -e "$ROOT/results/distributions_5em6/${shape}_musmooth${sm}_musparse${sp}_ftol0_gradS/x_opt.fits" ] && continue
  sub "distribf0_${shape}" 60:00:00 "CUDA_VISIBLE_DEVICES=0 $PY $ROOT/scripts/run_results/syntheticdata_distributions.py --mu-smooth $sm --mu-sparse $sp --shape $shape --ftol 0 --tag gradS"
done
# D. MC-SUGAR
REC_FTOL=0 bash $ROOT/jobs/final/sugar_fixedpoint.sh
