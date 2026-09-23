###########
# Imports #
###########
# Standard
import os
import subprocess
import shutil

# Config
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig
from omegaconf.errors import MissingMandatoryValue
from hydra.utils import get_original_cwd
import re

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def filter_fn_name(x):
    return not (
        x.startswith("launcher")
        or x.startswith("cluster")
        or x.startswith("load_ckpt")
        or x.startswith("+")
    )


def filter_fn_cmd(x):
    return not (x.startswith("launcher") or x.startswith("cluster"))


def enclose_args(x):
    enclosed = []
    for arg in x:
        key, value = arg.split("=")
        enclosed.append(f'{key}="{value}"')
    return enclosed


def handle_thoth_gpu(cfg: DictConfig, hydra_cfg: DictConfig) -> str:
    """
    Output example:
    -----------------
    #!/usr/bin/zsh

    #OAR -l walltime=12:00:00
    #OAR -n mybigbeautifuljob
    #OAR -t besteffort
    #OAR -t idempotent
    #OAR -p gpumem>'20000'
    #OAR -p gpumodel='p100' #NOTE: last property takes priority
    #OAR -d /path/to/dir/
    #OAR -E /path/to/file.stderr
    #OAR -O /path/to/file.stdout

    source gpu_setVisibleDevices.sh

    conda activate $my_env

    python train.py $overrides
    """
    # Setup
    cmd = ""
    cluster = cfg.cluster
    launcher = cfg.launcher
    # create OAR log folder
    os.makedirs(f"{cluster.engine}", exist_ok=True)
    # copy config file when job is created
    shutil.copy2(".hydra/config.yaml", "config.yaml")

    #####################
    # Construct command #
    #####################

    # Shebang
    cmd += f"#!{cluster.shell.bin_path}\n"
    # Space
    cmd += f"\n"
    # Walltime
    cmd += f"{cluster.directive} -l walltime={launcher.walltime}\n"
    # Remove overrides from launcher/cluster
    overrides = hydra_cfg.overrides.task
    # Job name
    filtered_args_name = list(filter(filter_fn_name, overrides))
    filtered_args_name = [
        e if (not e.startswith("comment")) else e.replace("comment", "c")
        for e in filtered_args_name
    ]
    job_name = ",".join([a.split(".")[-1] for a in filtered_args_name][:3])
    cmd += f"{cluster.directive} -n {job_name}\n"
    # write model_id to file
    # with open("model_id", "w") as f:
    # f.write(cfg.model.id)
    # Best effort
    if launcher.besteffort:
        cmd += f"{cluster.directive} -t besteffort\n"
    # Idempotent (i.e. automatic restart)
    if launcher.idempotent:
        cmd += f"{cluster.directive} -t idempotent\n"
    # GPU memory property
    if launcher.gpumem is not None:
        cmd += f"{cluster.directive} -p gpumem>{launcher.gpumem!r}\n"
    # GPU model property
    # NOTE: `gpumodel` takes priority over `gpumem` if both are defined
    if cluster.gpumodel is not None:
        if type(cluster.gpumodel) == ListConfig:
            cmd += f"{cluster.directive} -p "
            cmd += " or ".join([f"gpumodel={m!r}" for m in cluster.gpumodel])
            cmd += "\n"
        else:
            cmd += f"{cluster.directive} -p gpumodel={launcher.gpumodel!r}\n"
    if cluster.gpuhost is not None:
        if type(cluster.gpuhost) == ListConfig:
            cmd += f"{cluster.directive} -p "
            cmd += " or ".join([f"host={m!r}" for m in cluster.gpuhost])
            cmd += "\n"
        else:
            cmd += f"{cluster.directive} -p host={cluster.gpuhost!r}\n"
    # path to dir
    cmd += f"{cluster.directive} -d {hydra_cfg.runtime.cwd}\n"
    # Job stderr
    cwd = os.getcwd()
    err_path = os.path.join(cwd, f"{cluster.engine}/%jobid%.stderr")
    cmd += f"{cluster.directive} -E {err_path}\n"
    # Job stdout
    out_path = os.path.join(cwd, f"{cluster.engine}/%jobid%.stdout")
    cmd += f"{cluster.directive} -O {out_path}\n"
    # Space
    cmd += f"\n"
    # Shell instance
    cmd += f"source {cluster.shell.config_path}\n"
    # Space
    cmd += f"\n"
    # source gpu_setVisibleDevices.sh
    cmd += f"{cluster.cleanup}\n"
    # Space
    cmd += f"\n"
    # Print hostname & devices
    cmd += f'echo "Host is `hostname`"\n'
    cmd += f'echo "Host devices : \\n `nvidia-smi -L`"\n'
    cmd += f'echo "Visible devices : $CUDA_VISIBLE_DEVICES"\n'
    # conda environment
    cmd += f"conda activate {launcher.conda_env}\n"
    cmd += f"export PYTHONBREAKPOINT=0\n"
    # Space
    cmd += f"\n"
    # Python command
    filtered_args_cmd = list(filter(filter_fn_cmd, overrides))
    args = " ".join(enclose_args(filtered_args_cmd))
    cmd += f"python {launcher.cmd} {args} hydra.run.dir={cwd}"
    return cmd


def handle_thoth_cpu(cfg: DictConfig, hydra_cfg: DictConfig) -> str:
    """
    Output example:
    -----------------
    #!/usr/bin/zsh

    #OAR -l walltime=12:00:00
    #OAR -n mybigbeautifuljob
    #OAR -t besteffort
    #OAR -t idempotent
    #OAR -p gpumem>'20000'
    #OAR -p cluster='thoth'
    #OAR -p gpumodel='p100' #NOTE: last property takes priority
    #OAR -d /path/to/dir/
    #OAR -E /path/to/file.stderr
    #OAR -O /path/to/file.stdout

    conda activate $my_env

    python train.py $overrides
    """
    # Setup
    cmd = ""
    cluster = cfg.cluster
    launcher = cfg.launcher
    # create OAR log folder
    os.makedirs(f"{cluster.engine}")
    # copy config file when job is created
    shutil.copy2(".hydra/config.yaml", "config.yaml")

    #####################
    # Construct command #
    #####################

    # Shebang
    cmd += f"#!{cluster.shell.bin_path}\n"
    # Space
    cmd += f"\n"
    # Walltime
    # Remove overrides from launcher/cluster
    overrides = hydra_cfg.overrides.task
    # Job name
    filtered_args_name = list(filter(filter_fn_name, overrides))
    job_name = ",".join([a.split(".")[-1] for a in filtered_args_name][:3])
    cmd += f"{cluster.directive} -n {job_name}\n"
    cmd += f"{cluster.directive} -p cluster='thoth'"
    if cluster.host is not None:
        if type(cluster.host) == ListConfig:
            cmd += "and ("
            cmd += " or ".join([f"host={m!r}" for m in cluster.host])
            cmd += ")"
        else:
            cmd += f"and host={cluster.host!r}"
    cmd += "\n"
    cmd += f"{cluster.directive} -l nodes=1/core={cluster.cores},"
    cmd += f"walltime={cluster.walltime}\n"
    # cmd += f"{cluster.directive} -l walltime={launcher.walltime}\n"
    # cmd += f"{cluster.directive} -l walltime=4:00:00\n"
    # write model_id to file
    # with open("model_id", "w") as f:
    # f.write(cfg.model.id)
    # Best effort
    if launcher.besteffort:
        cmd += f"{cluster.directive} -t besteffort\n"
    # Idempotent (i.e. automatic restart)
    if launcher.idempotent:
        cmd += f"{cluster.directive} -t idempotent\n"
    # path to dir
    # cmd += f"{cluster.directive} -d {hydra_cfg.runtime.cwd}\n"
    # Job stderr
    cwd = os.getcwd()
    err_path = os.path.join(cwd, f"{cluster.engine}/%jobid%.stderr")
    cmd += f"{cluster.directive} -E {err_path}\n"
    # Job stdout
    out_path = os.path.join(cwd, f"{cluster.engine}/%jobid%.stdout")
    cmd += f"{cluster.directive} -O {out_path}\n"
    # Space
    cmd += f"\n"
    # Shell instance
    cmd += f'echo "$OAR_JOB_ID" > job_id\n'
    cmd += f"source {cluster.shell.config_path}\n"
    # Space
    cmd += f"\n"
    # source gpu_setVisibleDevices.sh
    cmd += f"{cluster.cleanup}\n"
    # Space
    cmd += f"\n"
    # Print hostname & devices
    cmd += f'echo "Host is `hostname`"\n'
    # cmd += f'echo "Host devices : \\n `nvidia-smi -L`"\n'
    # cmd += f'echo "Visible devices : $CUDA_VISIBLE_DEVICES"\n'
    # conda environment
    cmd += f"export PYTHONBREAKPOINT=0\n"
    cmd += f"conda activate {launcher.conda_env}\n"
    cmd += f"cd /home/tbodrito/exo/dl4hci\n"
    cmd += f'echo "CWD $PWD"\n'
    # Space
    cmd += f"\n"
    # Python command
    filtered_args_cmd = list(filter(filter_fn_cmd, overrides))
    args = " ".join(enclose_args(filtered_args_cmd))
    cmd += f"python {launcher.cmd} {args} hydra.run.dir={cwd}"
    return cmd


def handle_jean_zay(cfg: DictConfig, hydra_cfg: DictConfig) -> str:
    """
    Output example:
    -----------------

    conda activate $my_env

    python train.py $overrides
    """
    # Setup
    cmd = ""
    cluster = cfg.cluster
    launcher = cfg.launcher
    # create OAR log folder
    os.makedirs(f"{cluster.engine}")
    # copy config file when job is created
    shutil.copy2(".hydra/config.yaml", "config.yaml")

    #####################
    # Construct command #
    #####################

    # Shebang
    cmd += f"#!{cluster.shell.bin_path}\n"
    # Space
    cmd += f"\n"
    # Walltime
    # cmd += f"{cluster.directive} --time={launcher.walltime}\n"
    # Remove overrides from launcher/cluster
    overrides = hydra_cfg.overrides.task
    # Job name
    filtered_args_name = list(filter(filter_fn_name, overrides))

    job_name = ",".join([a.split(".")[-1] for a in filtered_args_name])
    for arg in filtered_args_name:
        if "comment" in arg:
            job_name = arg.split("=")[-1]
    # cmd += f"{cluster.directive} --job-name={job_name}\n"
    cmd += f"{cluster.directive} --job-name={cfg.name}\n"
    # cmd += f"{cluster.directive} -p cluster='thoth'\n"
    # write model_id to file
    # with open("model_id", "w") as f:
    # f.write(cfg.model.id)
    # Best effort
    if launcher.besteffort:
        cmd += f"{cluster.directive} -t besteffort\n"
    # Idempotent (i.e. automatic restart)
    if launcher.idempotent:
        cmd += f"{cluster.directive} -t idempotent\n"
    # GPU memory property
    # if launcher.gpumem is not None:
    # cmd += f"{cluster.directive} -p gpumem>{launcher.gpumem!r}\n"
    # GPU model property
    # NOTE: `gpumodel` takes priority over `gpumem` if both are defined
    # if launcher.gpumodel is not None:
    # if type(launcher.gpumodel) == ListConfig:
    # cmd += f"{cluster.directive} -p "
    # cmd += " or ".join([f"gpumodel={m!r}" for m in launcher.gpumodel])
    # cmd += "\n"
    # else:
    # cmd += f"{cluster.directive} -p gpumodel={launcher.gpumodel!r}\n"
    # path to dir
    # cmd += f"{cluster.directive} -d {hydra_cfg.runtime.cwd}\n"
    # Job stderr
    cwd = os.getcwd()
    err_path = os.path.join(cwd, f"{cluster.engine}/%j.stderr")
    cmd += f"{cluster.directive} --error={err_path}\n"
    # Job stdout
    out_path = os.path.join(cwd, f"{cluster.engine}/%j.stdout")
    cmd += f"{cluster.directive} --output={out_path}\n"

    # 1 GPU set up
    cmd += f"{cluster.directive} -C {cluster.partition}\n"
    cmd += f"{cluster.directive} --nodes={cluster.nodes}\n"
    cmd += f"{cluster.directive} --ntasks-per-node={cluster.tasks_node}\n"
    cmd += f"{cluster.directive} --gres=gpu:{cluster.gpus}\n"
    cmd += f"{cluster.directive} --cpus-per-task={cluster.cpus_task}\n"
    cmd += f"{cluster.directive} --hint=nomultithread\n"
    cmd += f"{cluster.directive} --qos={cluster.qos}\n"
    cmd += f"{cluster.directive} --time={cluster.walltime}\n"
    cmd += f"{cluster.directive} --account={cluster.account}@v100\n"
    if cluster.array_arg is not None:
        assert cluster.array_max > 0
        print(f"{cluster.array_max=}")
        cmd += f"{cluster.directive} --array=0-{cluster.array_max}\n"

    # Space
    cmd += f"\n"
    # Shell instance
    cmd += f"source {cluster.shell.config_path}\n"
    # Space
    cmd += f"\n"
    # source gpu_setVisibleDevices.sh
    cmd += f"{cluster.cleanup}\n"
    # Space
    cmd += f"\n"
    # Print hostname & devices

    cmd += f'date + "%T"\n'
    cmd += 'echo `date +"%Y-%m-%d %T"`\n'
    cmd += f'echo "Host is `hostname`"\n'
    cmd += f'echo "Host devices : \\n `nvidia-smi -L`"\n'
    cmd += f'echo "Visible devices : $CUDA_VISIBLE_DEVICES"\n'
    # cmd += f'echo "Current dir : $PWD"\n'
    cmd += f"export PYTHONBREAKPOINT=0\n"
    # cmd += f"export WANDB_CACHE_DIR='./wandb/wandb_cache_dir'\n"
    # cmd += f"export WANDB_CACHE_DIR='./wandb/wandb_cache_dir'\n"
    # cmd += f"export WANDB_DIR='./wandb'\n"
    # cmd += f"export WANDB_DATA_DIR='./wandb/wandb_data_dir'\n"
    # cmd += f"cd $WORK/exo/new_project/code\n"
    cmd += f"cd $WORK/exo/dl4hci\n"
    cmd += f'echo "Current dir : $PWD"\n'
    # conda environment
    cmd += f"conda activate {launcher.conda_env}\n"
    cmd += f"export WANDB_MODE=offline\n"
    # cmd += f"export CUDA_LAUNCH_BLOCKING=1\n"
    # Space
    cmd += f"\n"
    # Python command
    filtered_args_cmd = list(filter(filter_fn_cmd, overrides))
    args = " ".join(enclose_args(filtered_args_cmd))

    if cluster.array_arg is not None:
        assert cluster.array_max > 0
        print(f"{cluster.array_max=}")
        # for i in range(0, cluster.array_max + 1):
            # path_dir = os.path.join(cwd, f"{cluster.array_arg}_{i}")
            # os.makedirs(path_dir)
        # cmd += f"{cluster.directive} --array=0-{cluster.array_max}\n"
        # cmd += (f"python {launcher.cmd} log_wandb=true disable_tqdm=true "
        # f"{args} {cluster.array_arg}=$\{SLURM_ARRAY_TASK_ID\} "
        # f"hydra.run.dir={cwd}/$\{SLURM_ARRAY_TASK_ID\}")
        # cmd += ("python {} log_wandb=true disable_tqdm=true "
        # "{} {}=$\{SLURM_ARRAY_TASK_ID\} "
        # "hydra.run.dir={}/$\{SLURM_ARRAY_TASK_ID\}").format(launcher.cmd, args, cluster.array_arg, cwd)
        cmd += (
            f"python {launcher.cmd} log_wandb=true disable_tqdm=true "
            + f"{args} {cluster.array_arg}="
            + "${SLURM_ARRAY_TASK_ID} "
            + f"hydra.run.dir={cwd}/{cluster.array_arg}_"
            + "${SLURM_ARRAY_TASK_ID}"
        )
    else:
        cmd += f"python {launcher.cmd} log_wandb=true disable_tqdm=true {args} hydra.run.dir={cwd}"

    cmd += '\necho `date +"%Y-%m-%d %T"`'
    return cmd


def create(cfg: DictConfig) -> None:
    with open("pending", "a"):
        pass
    log.debug(cfg)
    cluster = cfg.cluster
    # Some assertions on possible combinations
    launcher = cfg.launcher
    assert (
        launcher.name in cluster.launchers
    ), f"{launcher.name} not in {cluster.launchers}"
    # Get Hydra config
    hydra_cfg = HydraConfig.get()
    # Construct command
    use_ssh = False
    if cluster.name == "thoth_gpu":
        cmd = handle_thoth_gpu(cfg, hydra_cfg)
        use_ssh = True
    elif cluster.name == "thoth_cpu":
        cmd = handle_thoth_cpu(cfg, hydra_cfg)
        use_ssh = True
    elif cluster.name == "jeanzay":
        cmd = handle_jean_zay(cfg, hydra_cfg)
    else:
        raise ValueError(f"No valid cluster selected: {cfg.cluster.name!r}")

    # make sure that the runs are logged
    # cmd += " log_wandb=true"

    # disable progress bar
    # cmd += " disable_tqdm=true"

    log.info(
        f"Selected cluster: {cluster.name}" f" (running on {cluster.engine})"
    )
    log.info(
        f"Using {cluster.shell.bin_path!r} for shebang,"
        f" {cluster.directive!r} as directive"
    )
    log.debug(cmd)
    print(cmd)

    # Get path to script
    sh_path = os.path.join(os.getcwd(), launcher.filename)

    # fix access2-cp
    # sh_folder = os.path.join(get_original_cwd(), "launch_scripts")
    # os.makedirs(sh_folder, exist_ok=True)
    # workdir = os.getcwd()
    # workdir = re.sub('^.*trainings/', '', workdir)
    # workdir = re.sub('^.*tests/', '', workdir)
    # workdir = workdir.replace('/', '_')
    # sh_path = os.path.join(sh_folder, workdir)

    # Write down .sh file
    with open(sh_path, "w") as f:
        f.write(cmd)

    # Make file executable
    chmod_cmd = f"chmod +x {sh_path!r}"
    subprocess.check_call(chmod_cmd, shell=True)
    # Connect to frontal node and invoke launch command
    cluster_cmd = f"{cluster.cmd} {sh_path!r}"
    if use_ssh:
        ssh_cmd = f'ssh {cluster.node} "{cluster_cmd}"'
        log.debug(ssh_cmd)
        # Launch job over SSH
        subprocess.check_call(ssh_cmd, shell=True)
    else:
        subprocess.check_call(cluster_cmd, shell=True)

    try:
        job_id = hydra_cfg.job.id
    except MissingMandatoryValue:
        job_id = f"MAIN"

    logging.info(f"Job {job_id} launched!")


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    log.info(f"Current working directory: {os.getcwd()}")
    try:
        create(cfg)
    except Exception as e:
        log.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
