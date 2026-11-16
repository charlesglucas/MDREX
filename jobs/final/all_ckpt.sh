#!/bin/bash
# Full synthetic series with an ExoMILD checkpoint (e.g. the 12-term one, jobs/exomild_H2_12.sh):
#   CKPT=<path to ckpt_*.pt> TAG=<series tag> bash all_ckpt.sh [shape flux ...]   (default: the 9 cases)
# for each case (angle 0 for the hyperparameters, 10 angles for the tables):
#   1. MC-SUGAR (refined, iterated to the fixed point of the covariance, at most OUTER=6 re-estimations) from the
#      sugarimp couple (12 terms, random projections) -> results/sugar/<case>_<TAG>[_refined<k>], _fixedpoint.npz
#      then Table 1 at the fixed point (job after it)  -> results/mdrex_results/..._ftol0_<TAG>fixedpoint
#   2. grid 5 x 5 (log10 mu_smooth 6..8, log10 mu_sparse 4.48..6.48, half-decade), PSNR / oracle MSE at angle 0
#      -> results/sugar/<case>_<TAG>_grid/grid_0.npz, then Table 2 at the argmax of the PSNR -> ..._ftol0_<TAG>psnrmax
J=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/jobs/final
P="cluster IN ('vercors10','vercors11','vercors12','vercors13')"
: ${CKPT:?CKPT (checkpoint path) required}; : ${TAG:?TAG (series tag) required}
# STAR (env): observation <star>/<date> under data/nuisances (default HIP 72192); then MU0ALL="<mu_smooth> <mu_sparse>"
# (starting couple of every case), GS / GP (grid, log10), GWALL (walltime of the grids, hours) may be set
export CKPT TAG
[ -n "$STAR" ] && export STAR EXTRA="--star $STAR $EXTRA"
GS=${GS:-6 6.4771 7 7.4771 8}; GP=${GP:-4.4771 5 5.4771 6 6.4771}; GWALL=${GWALL:-24}
declare -A MU0=(   # starting couples: sugarimp fixed points (2026-11-11)
  [medium_ellipse_1e-6]="1.43e7 1.01e6" [medium_ellipse_5e-6]="1.45e6 1.13e6" [medium_ellipse_1e-5]="1.37e6 1.16e6"
  [spiral_1e-6]="7.49e7 6.11e5" [spiral_5e-6]="1.03e7 3.06e5" [spiral_1e-5]="4.50e6 2.44e5"
  [circle_1e-6]="9.62e6 8.75e5" [circle_5e-6]="6.66e5 4.90e5" [circle_1e-5]="6.27e5 4.56e5")
CASES=("$@"); [ ${#CASES[@]} -eq 0 ] && CASES=(medium_ellipse 1e-6 medium_ellipse 5e-6 medium_ellipse 1e-5 spiral 1e-6 \
  spiral 5e-6 spiral 1e-5 circle 1e-6 circle 5e-6 circle 1e-5)
set -- "${CASES[@]}"
while [ $# -ge 2 ]; do
  shape=$1; flux=$2; shift 2; tag=$(echo $flux | sed 's/e-/em/')
  read sm sp <<< "${MU0[${shape}_$flux]}"
  [ -n "$MU0ALL" ] && read sm sp <<< "$MU0ALL"
  id=$(MODE=opt OUTER=6 bash $J/sugar_iter.sh $shape $flux $sm $sp | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2)
  echo "$shape $flux: MC-SUGAR $id"
  oarsub -a $id -l "gpu=1,walltime=24:00:00" -p "$P" -n "t1${TAG}_${shape}_${tag}" \
    "CKPT=$CKPT STAR=$STAR SUGAR_TAG=$TAG bash $J/run_from_sugar.sh table1 $shape $flux fixedpoint" | grep -o "OAR_JOB_ID=[0-9]*" | sed "s/^/  table 1 /"
  id=$(MODE=grid GS="$GS" GP="$GP" NAME=0 WALL=$GWALL bash $J/sugar_iter.sh $shape $flux 0 0 \
       | grep -o "OAR_JOB_ID=[0-9]*" | cut -d= -f2)
  echo "  grid $id"
  oarsub -a $id -l "gpu=1,walltime=24:00:00" -p "$P" -n "t2${TAG}_${shape}_${tag}" \
    "CKPT=$CKPT STAR=$STAR SUGAR_TAG=$TAG bash $J/run_from_sugar.sh table1 $shape $flux psnrmax" | grep -o "OAR_JOB_ID=[0-9]*" | sed "s/^/  table 2 /"
done
