#!/bin/bash
# MC-SUGAR by iterative differentiation of the reconstructions (scripts/run_results/syntheticdata_sugar_iter.py,
# reconstruction/sugar_iterative.py; the implicit version syntheticdata_sugar.py / sugar.py is unchanged), angle 0, tag
# sugariter, reconstructions with the gradient of mu_sparse sum(x) (see the header of syntheticdata_sugar_iter.py).
#   MODE=check bash sugar_iter.sh <shape> <flux> <mu_smooth> <mu_sparse> [...]
#       validation at a couple: reconstructions and their convergence, MC-SUGAR (iterative) vs central finite differences
#       of MC-SURE (0.05 and 0.15 decade, same reconstructions), implicit MC-SUGAR (CG 80 / 320 / 640) at the same x_hat
#       -> results/sugar/<shape>_alpha<..>_angle0_sugariter_check/
#   MODE=opt bash sugar_iter.sh <shape> <flux> <mu_smooth_0> <mu_sparse_0> [...]
#       minimization of MC-SURE (C^y) from the given couple (e.g. the minimum of MC-SURE on the angle-0 grid), then
#       refined MC-SURE iterated to the fixed point of the covariance (--outer 2, 0.1 decade), in the same job
#       -> results/sugar/<shape>_alpha<..>_angle0_sugariter[_refined, _refined2]/trajectory.npz
# MC-SUGAR of MODE=opt: implicit (CG, default of syntheticdata_sugar_iter.py; validated by the check jobs of 2026-11-06:
# within 3-5% of the finite differences) with TAG=sugarimp; EXTRA="--sugar-mode iterative" for the tangents (inaccurate).
#   MODE=refine ITER=<k> bash sugar_iter.sh <shape> <flux> 0 0 [...]
#       continuation of the fixed point: refined MC-SURE iteration k from the couple of <tag>_refined<k-1> (--refined
#       --refined-iter k), then --outer 2 more -> <tag>_refined<k>, ... (mu arguments unused); BOUNDS (env) overrides --bounds
#   MODE=grid GS="<log10 mu_smooth ...>" GP="<log10 mu_sparse ...>" NAME=<n> bash sugar_iter.sh <shape> <flux> 0 0
#       MSE / PSNR of the reconstructions of y (gradient of sum(x)) on the grid GS x GP
#       -> results/sugar/<shape>_alpha<..>_angle0_<tag>_grid/grid_<n>.npz
# CKPT (env): ExoMILD checkpoint (default exomild_H2). ANTERIOR (env): job id this one waits for.
# EXTRA (env): additional options, e.g. EXTRA="--abs-grad" (gradient of sum|x| as run_bfgs) with TAG=sugariterabs.
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
P=${P:-"cluster IN ('vercors10','vercors11','vercors12','vercors13')"}   # P (env): other GPU nodes
TAG=${TAG:-sugarimp}
BASE="--angle 0 --hvp-step 1e-2 --normalize --rec-ftol 0 --rec-xtol 1e-10 --tag $TAG ${CKPT:+--ckpt $CKPT} $EXTRA"
while [ $# -ge 4 ]; do
  shape=$1; flux=$2; sm=$3; sp=$4; shift 4
  tag=$(echo $flux | sed 's/e-/em/')
  if [ "${MODE:-check}" = check ]; then
    OPTS="$BASE --check --check-mu $sm $sp --check-steps 0.05 0.15 --check-repeats 1 --check-cg 80 320 640 --cg-rtol 0"
    name=sugariterchk_${shape}_${tag}; wall=36
  elif [ "$MODE" = objtest ]; then   # <mu_smooth> <mu_sparse> given in log10: see --objtest
    OPTS="$BASE --objtest $sm $sp"
    name=objtest_${shape}_${tag}; wall=12
  elif [ "$MODE" = gsure ]; then   # GSTEP=perturb (x_hat(y + eps delta) on GS x GP) or eval (GSURE vs oracle W-MSE)
    OPTS="$BASE --gsure ${GSTEP:-perturb} --grid-smooth $GS --grid-sparse $GP"
    name=gsure${GSTEP:-perturb}_${shape}_${tag}_${NAME:-0}; wall=${WALL:-10}
  elif [ "$MODE" = grid ]; then
    OPTS="$BASE --grid-smooth $GS --grid-sparse $GP --grid-name ${NAME:-0}"
    name=sugargrid_${shape}_${tag}_${NAME:-0}; wall=${WALL:-10}
  else
    OPTS="$BASE --mu0 $sm $sp --bounds ${BOUNDS:-6 9 4.4771 7.4771} --maxfun 20 --xtol 0.02 --ftol 0 --gtol 0 --gabs 100 \
          --srel 1e-5 --sstall 3 --restarts 3 --max-step 0.5 --outer ${OUTER:-2} --outer-tol 0.1 --cg-maxiter 200 --cg-rtol 1e-4"
    name=sugariteropt_${shape}_${tag}; wall=168
    if [ "$MODE" = refine ]; then OPTS="$OPTS --refined --refined-iter $ITER"; name=sugarref${ITER}_${shape}_${tag}; fi
  fi
  oarsub -l "gpu=1,walltime=$wall:00:00" -p "$P" -n "$name" ${ANTERIOR:+-a $ANTERIOR} \
    "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/run_results/syntheticdata_sugar_iter.py --flux $flux --shape $shape $OPTS" \
    | grep -o "OAR_JOB_ID=[0-9]*" | sed "s/^/$name: /"
done
