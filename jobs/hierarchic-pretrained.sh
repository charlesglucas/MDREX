#!/bin/bash

#OAR -l walltime=08:00:00
#OAR -l gpu=1
#OAR -n iterative-scheme

# aller dans le dossier du projet
cd /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO

# lancer le script avec Python du venv directement
./.venv/bin/python -m scripts.disk_rec_exomildloss_reg_vmlmb_hierarchic_pretrained