#!/bin/bash

# Reconstructions at the couple found by refined MC-SUGAR (fixed shrinkage coefficients, tag rhofixfp, first refined stage
# = refined MC-SUGAR of the paper; if absent, the MC-SUGAR stage), ftol 0 / xtol 1e-10:
#   bash run_from_sugar.sh table1 <shape> <flux>   -> Table 1, 10 angles: results/mdrex_results/musmooth<..>_musparse<..>_<shape>_alpha<..>_ftol0_rhofix/
#   bash run_from_sugar.sh distrib <shape> 5e-6    -> Fig. 3, patch distributions: results/distributions_5em6/<shape>_musmooth<..>_musparse<..>_ftol0_rhofix/
# Launched by all_rhofix.sh as jobs depending on the MC-SUGAR job of the case.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
kind=$1; shape=$2; flux=$3
tag=$(echo $flux | sed 's/e-/em/')
base=$ROOT/results/sugar/${shape}_alpha${tag}_angle0_rhofixfp
traj=${base}_refined/trajectory.npz
[ -e "$traj" ] || traj=$base/trajectory.npz
read sm sp <<< $($PY -c "import numpy as np; m=np.load('$traj')['mu_opt']; print(f'{m[0]:.3g} {m[1]:.3g}')")
echo "$kind $shape $flux: mu = ($sm, $sp) from $traj"
cd $ROOT
if [ $kind = table1 ]; then
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_shape.py --mu-smooth $sm --mu-sparse $sp --shape $shape \
    --flux $flux --ftol 0 --xtol 1e-10 --tag rhofix
else
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_distributions.py --mu-smooth $sm --mu-sparse $sp \
    --shape $shape --ftol 0 --xtol 1e-10 --tag rhofix
fi
