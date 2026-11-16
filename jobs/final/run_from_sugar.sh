#!/bin/bash

# Reconstructions at the couple found by refined MC-SUGAR (fixed shrinkage coefficients, tag rhofixfp, first refined stage
# = refined MC-SUGAR of the paper; if absent, the MC-SUGAR stage), ftol 0 / xtol 1e-10:
#   bash run_from_sugar.sh table1 <shape> <flux>   -> Table 1, 10 angles: results/mdrex_results/musmooth<..>_musparse<..>_<shape>_alpha<..>_ftol0_rhofix/
#   bash run_from_sugar.sh distrib <shape> 5e-6    -> Fig. 3, patch distributions: results/distributions_5em6/<shape>_musmooth<..>_musparse<..>_ftol0_rhofix/
# Launched by all_rhofix.sh as jobs depending on the MC-SUGAR job of the case.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
#   4th argument (optional): refined stage, refined (default) or refined<k> (k-th fixed-point iteration), tag rhofix<k>
#   CKPT (env): ExoMILD checkpoint (default exomild_H2); STAR (env): <star>/<date> under data/nuisances (default HIP 72192)
#   SUGAR_TAG (env, default rhofixfp): MC-SUGAR series; rhofixcpu (CPU probe, sugar_cpu_chain.sh) -> tag rhofixcpu<k>
kind=$1; shape=$2; flux=$3; stage=${4:-refined}
SUGAR_TAG=${SUGAR_TAG:-rhofixfp}
tag=$(echo $flux | sed 's/e-/em/')
base=$ROOT/results/sugar/${shape}_alpha${tag}_angle0_${SUGAR_TAG}
traj=${base}_${stage}/trajectory.npz
# stage fixedpoint: couple of <tag>_fixedpoint.npz (end of the fixed-point iterations of MODE=opt);
# stage psnrmax: argmax of the PSNR (angle 0) on the grid <tag>_grid/grid_0.npz (oracle, Table 2)
if [ $stage = fixedpoint ]; then
  traj=${base}_fixedpoint.npz
  # fixed point not reached in the walltime: last completed refined stage
  [ -e "$traj" ] || traj=$(ls -d ${base}_refined* 2>/dev/null | sort -V | while read d; do [ -e $d/trajectory.npz ] && $PY -c "import numpy as np, sys; sys.exit('mu_opt' not in np.load('$d/trajectory.npz').files)" && echo $d/trajectory.npz; done | tail -1)
fi
[ $SUGAR_TAG = rhofixfp ] && otag=rhofix${stage#refined} || otag=${SUGAR_TAG}${stage#refined}
[ -e "$traj" ] || [ $stage = psnrmax ] || traj=$base/trajectory.npz
if [ $stage = psnrmax ]; then
  traj=${base}_grid/grid_0.npz
  read sm sp <<< $($PY -c "import numpy as np; d=np.load('$traj'); c=list(d['columns']); r=d['rows']; i=r[:, c.index('psnr')].argmax(); print(f'{10**r[i,0]:.3g} {10**r[i,1]:.3g}')")
else
  read sm sp <<< $($PY -c "import numpy as np; m=np.load('$traj')['mu_opt']; print(f'{m[0]:.3g} {m[1]:.3g}')")
fi
echo "$kind $shape $flux: mu = ($sm, $sp) from $traj"
cd $ROOT
if [ $kind = table1 ]; then
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_shape.py --mu-smooth $sm --mu-sparse $sp --shape $shape \
    --flux $flux --ftol 0 --xtol 1e-10 --tag $otag ${CKPT:+--ckpt $CKPT} ${STAR:+--star $STAR}
else
  CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_distributions.py --mu-smooth $sm --mu-sparse $sp \
    --shape $shape --ftol 0 --xtol 1e-10 --tag $otag
fi
