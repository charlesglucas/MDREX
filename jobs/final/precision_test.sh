#!/bin/bash

# Why does VMLMB stop after ~100 iterations (zero steps, gradient still at 0.5-4% of its initial norm)?
#   1. warm restarts: after the reconstruction (ftol 0, xtol 1e-10), VMLMB restarted 5 times from its result (L-BFGS
#      memory reset): do the objective and the flux keep decreasing?
#   2. float64: objective and gradient computed in float64 (ExoMILD model, forward model, data): does VMLMB go further?
# Case: spiral 5e-6, (3e6, 1e5), angle 0 (largest distance to the converged iterate in the convergence tests).
# output: results/convergence/spiral_angle0_alpha5e-6_musmooth3e6_musparse1e5/trace_[double_][restart5_]ftol0_xtol1e-10.npz
ROOT=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO
PY=$ROOT/.venv/bin/python
RUN="$ROOT/scripts/run_results/convergence_test.py --shape spiral --flux 5e-6 --angle 0 --mu-smooth 3e6 --mu-sparse 1e5 --tols 0:1e-10 --trace"
oarsub -l "gpu=1,walltime=24:00:00" -p "cluster='vercors14'" -n "prec_restarts" "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --restarts 5" | grep OAR_JOB_ID
oarsub -l "gpu=1,walltime=48:00:00" -p "cluster='vercors9'" -n "prec_double" "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY $RUN --double" | grep OAR_JOB_ID
