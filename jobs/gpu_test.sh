#!/bin/bash

#OAR -l walltime=8:00:00
#OAR -l gpu=1
#OAR -n sure-grid

# aller dans le dossier du projet
cd /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO

# lancer le script avec Python du venv directement
.venv/bin/python scripts/disk_rec_exomildloss_reg_vmlmb_hierarchic_sure_gpu.py  --mu-smooth 1e3 --mu-sparse 1e3