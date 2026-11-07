#!/bin/bash
# All the synthetic experiments redone with the sparse term mu_sparse sum(x) of MDREX (since 2026-11-07, gradient
# mu_sparse at x = 0 instead of 0 with sum|x|: converged reconstructions, see reconstruction/mdrex.py, sparse_abs),
# fixed shrinkage coefficients, float32, VMLMB ftol 0 / xtol 1e-10, Monte Carlo probe drawn on the CPU:
#   A. MC-SURE grids at angle 0, half-decade 7 x 7 (grid_rhofixed.sh, TAG=sumx)
#      -> results/grids_111111111111/grid_sure_<shape>_alpha<..>_angle0_sumx/
#   B. MC-SUGAR (implicit, CG to 1e-4, syntheticdata_sugar_iter.py via sugar_iter.sh, tag sugarimp) then refined
#      MC-SUGAR to the fixed point of the covariance (at most OUTER=3 refined stages), from the couple of the previous
#      series (rhofixfp, last refined stage) -> results/sugar/<shape>_alpha<..>_angle0_sugarimp[_refined<k>]
#      (spiral 1e-6 and circle 5e-6: already computed on 2026-11-06/07, skipped)
#   C/D (after A and B): Table 1 (10 angles, run_from_sugar.sh with SUGAR_TAG=sugarimp), Table 2 (PSNR maximum of the
#      sumx grids), Fig. 3 distributions, figures.
# Usage: bash all_sumx.sh [A] [B]   (default: both)
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
J=$ROOT/jobs/final
parts=${*:-A B}
if [[ $parts == *A* ]]; then
  for shape in medium_ellipse spiral circle; do
    for flux in 1e-6 5e-6 1e-5; do
      echo "grid $shape $flux"
      TAG=sumx CLUSTERS="vercors10 vercors11 vercors12 vercors13 vercors18" bash $J/grid_rhofixed.sh $shape $flux 4
    done
  done
fi
if [[ $parts == *B* ]]; then
  cd $J
  for shape in medium_ellipse spiral circle; do
    for flux in 1e-6 5e-6 1e-5; do
      [ "$shape $flux" = "spiral 1e-6" ] || [ "$shape $flux" = "circle 5e-6" ] && continue
      tag=$(echo $flux | sed 's/e-/em/')
      prev=$(ls -d $ROOT/results/sugar/${shape}_alpha${tag}_angle0_rhofixfp* | grep -v npz | sort -V | tail -1)
      mu0=$($PY -c "import numpy as np; m=np.load('$prev/trajectory.npz')['mu_opt']; print(f'{m[0]:.4e} {m[1]:.4e}')")
      echo "sugar $shape $flux: mu0 = $mu0 ($(basename $prev))"
      MODE=opt OUTER=3 BOUNDS="5 9 4.4771 7.4771" bash sugar_iter.sh $shape $flux $mu0
    done
  done
fi
