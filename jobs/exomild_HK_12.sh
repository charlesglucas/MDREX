#!/bin/bash
# ExoMILD with the 12 terms (p8/p16/p32/p64 x s1/s2/s4) trained with the code of this repository (engine used by MD-REX);
# GPU >= 45 GB (24 GB: out of memory); same settings as exomild_H2_12 but trained on the H23 and K12 observations together (validation / detection in H23, as exomild_H2_12); outputs in data/run/train/<date>/exomild_HK_12
C=/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes
R=$C/EXMILDPACO
DB=$R/data/db_new
OUT=$R/data/run/train/$(date +%Y-%m-%d_%H-%M-%S)/exomild_HK_12
cd $R
oarsub -l "gpu=1,walltime=40:00:00" -p "gpu_mem >= 45000" -n "exomild_HK_12" \
  "CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $R/.venv/bin/python $R/main.py log_wandb=true hydra.run.dir=$OUT \
    name=exomild_HK_12 model=exomild train=exomild data.train.dataset_params_train.channel_idx=null \
    data.train.dataset_params_val.channel_idx=null data.detection.dataset_params.channel_idx=null \
    data.detection.n_cubes=5 sampler=exomild data.train.dataset_params_train.multibands=true \
    data.detection.dataset_params.multibands=true model.n_channels=2 data.train.num_workers=0 \
    data.detection.num_workers=0 data.train.dataset_params_val.multibands=true data.train.dataset_params_train.n_frames=32 \
    data.train.dataset_params_val.n_frames=32 data.detection.dataset_params.n_frames=32 \
    model.repeats=[1,1,1,1,1,1,1,1,1,1,1,1] \
    data.train.dataset_params_train.filter_obs_params.dual_filter=[DB_H23,DB_K12] \
    data.train.dataset_params_train.path_db=$DB data.train.dataset_params_val.path_db=$DB data.detection.dataset_params.path_db=$DB"
