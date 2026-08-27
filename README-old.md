# Deep Learning for High Contrast Imaging

## Setup

The database must be located in `./data/db_new` (for the Inria cluster, replace it with a symlink to `/scratch/vasher/tbodrito/exo/data/db_new`).

## MODEL&CO

### Training

To launch a training:

```
python -m main name="modelco_H2" model=modelco data.train.dataset_params_train.channel_idx=0 data.detection.n_cubes=5
```

#### Main arguments

* `name`: name of the run
* `model`: deep learning architecture
* `data.train.dataset_params_train.channel_idx`: index of the channel to use (e.g. in H2/H3, `0`->H2, `1`->H3, `null`-> H2 and H3)
* `data.detection.n_cubes`: number of cubes to use for computing detection metrics. This parameter is usually set to 100, but it is practical to reduce its value for debugging.


## ExoMILD

### Training

To launch a training:

Monospectral:
```
python -m main name="exomild_H2" model=exomild train=exomild data.train.dataset_params_train.channel_idx=0 data.detection.n_cubes=5 sampler=exomild data.train.dataset_params_train.multibands=false data.detection.dataset_params.multibands=false model.n_channels=1 model.use_dataparallel=true data.detection.n_cubes=5
```

To debug, with low memory constraints:
```
python -m main name="exomild_H2" model=exomild train=exomild data.train.dataset_params_train.channel_idx=0 data.detection.n_cubes=5 sampler=exomild data.train.dataset_params_train.multibands=false data.detection.dataset_params.multibands=false model.n_channels=1 model.use_dataparallel=true data.detection.n_cubes=5 data.train.dataset_params_train.n_frames=16 data.train.dataset_params_val.n_frames=16 data.detection.dataset_params.n_frames=16
```

Multispectral:
```
python -m main name="exomild_H2" model=exomild train=exomild data.train.dataset_params_train.channel_idx=null data.train.dataset_params_val.channel_idx=null data.detection.dataset_params.channel_idx=null data.detection.n_cubes=5 sampler=exomild data.train.dataset_params_train.multibands=true data.detection.dataset_params.multibands=true model.n_channels=2 data.train.num_workers=0 data.detection.num_workers=0 data.train.dataset_params_val.multibands=true data.train.dataset_params_train.n_frames=32 data.train.dataset_params_val.n_frames=32 data.detection.dataset_params.n_frames=32
```

#### Main arguments

* `name`: name of the run
* `model`: deep learning architecture
* `model.n_channels`: number of input channels
* `model.use_dataparallel`: if set to `true`, the run will use all GPUs available on the machine for the forward pass, otherwise it will use only 1 GPU
* `model.repeats`: determines the number of distributions used in each category: 
* * [
* *  patches 8x8   + no symmetry, patches 8x8   + 180deg symmetry, patches   8x8 + 180&90deg symmetry,
* *  patches 16x16 + no symmetry, patches 16x16 + 180deg symmetry, patches 16x16 + 180&90deg symmetry,
* *  patches 32x32 + no symmetry, patches 32x32 + 180deg symmetry, patches 32x32 + 180&90deg symmetry,
* *  patches 64x64 + no symmetry, patches 32x32 + 180deg symmetry, patches 32x32 + 180&90deg symmetry,
* * ]
* `model.batch_size`: the spatial locations of speckles patches are processed in batches during the forward pass of the neural network (to compute the quantities a and b, cf Exomild/PACO paper). This parameter determines the size of the batch. The parameter can be reduced when the memory available is limited (or on CPU). Not to be confused with the training batch size, i.e., the number of cubes processed simultaneously during training, which is always set to 1 because of large memory usage.
* `train`: selection of the training configuration. Must be equal to `model`
* `sampler`: selection of the sampling configuration. Must be equal to `model`
* `data.train.dataset_params_train.channel_idx`: index of the channel to use (e.g. in H2/H3, `0`->H2, `1`->H3, `null`->H2 and H3)
* `data.detection.n_cubes`: number of cubes to use for computing detection metrics. This parameter is usually set to 100, but it is practical to reduce its value for debugging.
* `data.train.dataset_params_[train|val].multibands`: determines if the [training|validation] dataset is multiband (boolean)
* `data.detection.dataset_params.multibands`: determines if the detection dataset is multiband (boolean)
* `data.[train|detection].num_workers`: number of child processes dedicated to generating the [training|detection] synthetic cubes
* `data.train.dataset_params_[train|val].n_frames`: number of frames in [training|validation] cubes
* `data.detection.dataset_params.n_frames`: number of frames in detection cubes

### Inference

Checkpoints and calibration files are available [here](https://sdrive.cnrs.fr/s/LknDyxtHPAoX8RM#). To perform inference:
```
python -m main mode=inference name="exomild_inference_asdi_north_aligned" sampler=exomild model=exomild data.inference.path_root="data/real_data/data_am" sampler.path_ckpts="data/checkpoints/1_asdi" model.repeats='[1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]' model.n_channels=2  model.use_dataparallel=true  model.rot_zero_init=false model.batch_size=16 data.inference.recursive=true sampler.path_calib="data/calib/calib_exomild_asdi.pt"
```

#### Main arguments

* `name`: name of the run
* `model`: deep learning architecture
* `model.n_channels`: number of input channels
* `model.rot_zero_init`: if set to `false`, the output detection maps will be north aligned
* `model.use_dataparallel`: if set to `true`, the run will use all GPUs available on the machine for the forward pass, otherwise it will use only 1 GPU
* `model.batch_size`: the spatial locations of speckles patches are processed in batches during the forward pass of the neural network (to compute the quantities a and b, cf Exomild/PACO paper). This parameter determines the size of the batch. The parameter can be reduced when the memory available is limited (or on CPU). Not to be confused with the training batch size, i.e., the number of cubes processed simultaneously during training, which is always set to 1 because of large memory usage.
* `data.inference.path_root`: path to the directory where inference cubes are located. The structure of the folder should look like `path_root/cube_1/*.fits`, `path_root/cube_2/*.fits`, ... if `data.inference.recursive=true`. Otherwise, the structure is simply `path_root/*.fits`.
* `sampler`: selection of the sampling configuration. Must be equal to `model`
* `model.repeats`: determines the number of distributions used in each category: 
* * [
* *  patches 8x8   + no symmetry, patches 8x8   + 180deg symmetry, patches   8x8 + 180&90deg symmetry,
* *  patches 16x16 + no symmetry, patches 16x16 + 180deg symmetry, patches 16x16 + 180&90deg symmetry,
* *  patches 32x32 + no symmetry, patches 32x32 + 180deg symmetry, patches 32x32 + 180&90deg symmetry,
* *  patches 64x64 + no symmetry, patches 32x32 + 180deg symmetry, patches 32x32 + 180&90deg symmetry,
* * ]
* `sampler.path_calib`: path to the calibration folder

## Using Jean Zay

The previous commands can be embedded into a Slurm script.  Different GPU partitions are available on Jean Zay, see [here](http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm.html) for more details.

Here is an example of a training script, using 2 GPUs on the same machine:

```
#!/usr/bin/bash

#SBATCH --job-name=myjobname
#SBATCH --error=path/to/error
#SBATCH --output=path/to/output
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=19:59:59
#SBATCH --account=myproject@jeanzaypartition(e.g., v100)

source ~/.bashrc

date + "%T"
echo `date +"%Y-%m-%d %T"`
echo "Host is `hostname`"
echo "Host devices : \n `nvidia-smi -L`"
echo "Visible devices : $CUDA_VISIBLE_DEVICES"

cd path/to/dl4hci

conda activate myenv


python -m main name="exomild_H2" model=exomild train=exomild data.train.dataset_params_train.channel_idx=0 data.detection.n_cubes=5 sampler=exomild data.train.dataset_params_train.multibands=false data.detection.dataset_params.multibands=false model.n_channels=1 model.use_dataparallel=true
```


## Disk modeling

Some scripts show toy examples of how to reconstruct a disk using a differentiable model of a disk. The reconstruction loss can either be the L2 loss:
```
python -m scripts.disk_l2
```
Or the negative log-likelihood of ExoMILD statistical model:
```
python -m scripts.disk_exomild model=exomild
```

### Training
To launch a training for disk reconstruction:

From visuthoth
```
cd /scratch2/clear/chalucas/codes/DiscRec
source ~/astro/bin/activate
.venv/bin/python -m main name="diskNN" log_wandb=true data.train.num_workers=16
```


### Regularization approach

cd /scratch2/clear/chalucas/codes/DiscRecCopy/scripts
source ~/astro/bin/activate
.venv/bin/python -m disk_rec_exomildloss_reg_vmlmb_hierarchic_sure

### Transfer results

scp -3 -r grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/results/grid_sure_pretrained/\* visuthoth:/scratch2/clear/chalucas/codes/REXMILDPACO/results/grid_sure_pretrained_1em-7/


rsync -r /scratch/vasher/tbodrito/exo/data/real_data/HR_4796/2015-02-03 grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/data/real_data/HR_4796/

rsync -r grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/results/results_realdata.npz /scratch2/clear/chalucas/codes/REXMILDPACO/results/


rsync -r /scratch/vasher/tbodrito/exo/data/real_data/HR_4796 /scratch2/clear/chalucas/codes/REXMILDPACO/data/real_data/HR_4796

rsync -r /Users/charleslucas/Documents/Science/astronomy/Rexpaco_results visuthoth:/scratch2/clear/chalucas/codes/REXMILDPACO/results/

rsync /scratch2/clear/chalucas/codes/REXMILDPACO/checkpoints_calib_exomild/  grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/

rsync -rv grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/results/mdrex_results /scratch2/clear/chalucas/codes/REXMILDPACO/results/

rsync -v visuthoth:/scratch2/clear/chalucas/codes/REXMILDPACO/results/HR4796.fits /Users/charleslucas/Documents/Science/astronomy/example-astro/

rsync -rv /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/dl4hci/data/run/train/2026-06-10_10-32-06/exomild_H2 /srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/checkpoints_calib_exomild/checkpoints/

rsync -rv grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/results/grids_111111111111/grid_sure_circle_alpha1em6 /scratch2/clear/chalucas/codes/REXMILDPACO/results/grids_111111111111/

rsync -rv grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/results/grid_sure_medium_ellipse_alpha1em6 /scratch2/clear/chalucas/codes/REXMILDPACO/results/grids_111111111111/

rsync -rv grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/results/mdrex_results /scratch2/clear/chalucas/codes/REXMILDPACO/results/

rsync -rv grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO /Users/charleslucas/Documents/codes-astro/MDREX

rsync -av --partial --progress grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO /Users/charleslucas/Documents/codes-astro/MDREX


rsync -rv grenoble.g5k:/srv/storage/thoth1@storage4.grenoble.grid5000.fr/chalucas/codes/EXMILDPACO/figures/ /Users/charleslucas/Documents/Science/astronomy/reconstruction-paper/mdrex/Figures

## noeud grid5000

oarsub -l nodes=1,walltime=04:00 "~/ssh_nodes/start_ssh.sh"

ssh localhost -p 2026 -i ~/ssh_nodes/client_key

# MDREX
