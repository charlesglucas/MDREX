import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.viz import cube_3d_viewer
import json
import random
from omegaconf import ListConfig

from models import get_model
from diffusion import get_diffusion
from sampler import get_sampler
from normalizer import create_normalizer
from inverse_pb.operators import MixingOperator
from datasets.dataloaders import get_dataloaders
from utils.logger import WandBLogger

# from samplers.sampler_wrapper import SamplerWrapper
from detection.detector import SamplingDetector, DeterministicDetector
from detection.evaluator import DetectionEvaluator
from datasets.transforms.symmetry import get_symmetry
from detection.utils import (
    MultiModelDetectionEvaluator,
    ModelDetectionEvaluator,
)
import copy

from detection.evaluation import eval_detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

k_filter = ["auc_pr", "auc_fdr_tpr", "tpr_max"]

aliases_all = {
    # H
    "hd21": "HD_216803__2015-06-27__09-36-00",
    "hd20": "HD_206860__2015-09-30__01-47-41",
    "hd18": "HD_188228__2015-06-01__07-40-44",
    "hd10": "HD_102647__2015-05-31__23-39-09",
    "hd15": "HD_159911__2016-04-16__09-34-10",
    "hd38": "HD_38678__2017-02-11__00-52-32",
    "hd88": "HD_88955__2018-02-27__04-08-45",
    "hip11": "HIP_112312__2015-10-01__02-04-36",
    "hip37": "HIP_37288__2017-02-13__03-24-12",
    # K
    "hd11": "HD_116434__2017-02-09__08-53-33",
    "hd11b": "HD_116434__2018-05-13__03-08-58",
    "hd189": "HD_189245__2015-05-14__08-55-56",
    "hd95": "HD_95086__2015-05-05__01-27-44",
}

# aliases_cat0 = {
    # # "hd18": "HD_188228__2015-06-01__07-40-44", # 320, LWE
    # # "hd21": "HD_216803__2015-06-27__09-36-00", # 512, LWE
    # # "hd85": "HD_8558__2015-10-28__03-08-29",
    # # "hip11":   "HIP_112312__2015-10-01__02-04-36", # 128
    # "tyc93": "TYC_9340-437-1__2016-06-27__09-30-54", # 115
    # "hip11": "HIP_112312__2015-10-01__02-04-36",
# }

# aliases_cat1 = {
    # # "hd10": "HD_102647__2015-05-31__23-39-09", # 1024
    # # "hd15": "HD_159911__2016-04-16__09-34-10", # 80
    # # "hd17": "HD_176367__2015-06-07__07-25-32", # 80
    # "hip37": "HIP_37288__2017-02-13__03-24-12",
    # "hd15": "HD_159911__2016-04-16__09-34-10", # 80
    # "hd17": "HD_176367__2015-06-07__07-25-32", # 80
# }

# aliases_cat2 = {
    # "hd14": "HD_14228__2015-09-26__07-10-54",
    # "hd20": "HD_206860__2015-09-30__01-47-41",
    # "hd38": "HD_38678__2017-02-11__00-52-32",
    # "hd88": "HD_88955__2018-02-27__04-08-45",
# }


aliases_cat0 = {
    # "hd18": "HD_188228__2015-06-01__07-40-44", # 320, LWE
    # "hd21": "HD_216803__2015-06-27__09-36-00", # 512, LWE
    # "hd85": "HD_8558__2015-10-28__03-08-29",
    # "hip11":   "HIP_112312__2015-10-01__02-04-36", # 128
    # "hip11": "HIP_112312__2015-10-01__02-04-36",
    # "hd17": "HD_176367__2015-06-07__07-25-32", # 80
    # "hd10": "HD_102647__2015-05-31__23-39-09", # s=1.02, t=1.28
    # "hd56": "HD_56022__2015-12-29__06-30-09", # s=2.02, t=1,21
    "hd30": "HD_30447__2015-12-29__01-40-09",
}

aliases_cat1 = {
    # "hd10": "HD_102647__2015-05-31__23-39-09", # 1024
    # "hd15": "HD_159911__2016-04-16__09-34-10", # 80
    # "hd17": "HD_176367__2015-06-07__07-25-32", # 80
    # "TYC_9340-437-1__2016-06-27__09-30-54"
    "tyc93": "TYC_9340-437-1__2016-06-27__09-30-54", # 115, s=0.91, t=2.95
    "hip11": "HIP_112312__2015-10-01__02-04-36", # s=0.65, t=2.32
}

aliases_cat2 = {
    # "tyc87": "TYC_8760-1468-1__2015-06-27__05-23-59",
    # "hd84": "HD_84075__2017-02-11__04-54-44",
    "hd88": "HD_88955__2018-02-27__04-08-45", # s=0.71, t=6.75
    "hip37": "HIP_37288__2017-02-13__03-24-12", # s=0.75, t=10.5
    # "hd14": "HD_14228__2015-09-26__07-10-54",
    # "hd20": "HD_206860__2015-09-30__01-47-41",
    # "hd38": "HD_38678__2017-02-11__00-52-32",

    # "hd15": "HD_159911__2016-04-16__09-34-10", # 80
}



def eval_grid_detection(cfg_init):
    cfg = copy.deepcopy(cfg_init)

    cfg.log_wandb = False
    cfg.eval.save_output = False
    cfg.data.detection.dataset_params.filter_obs_params.split = None

    cfg_grid = cfg.data.grid_detection
    force_rot_pre = cfg_grid.force_rot_pre
    force_rot_post = cfg_grid.force_rot_post
    all_n_frames = cfg_grid.n_frames
    # all_obs_id = cfg_grid.obs_id
    all_rma = cfg_grid.range_max_adaptive
    all_filenames = cfg_grid.filenames

    category = cfg_grid.category
    print(f"Category: {category}")
    if category is None:
        aliases = aliases_all
    elif category == 0:
        aliases = aliases_cat0
    elif category == 1:
        aliases = aliases_cat1
    elif category == 2:
        aliases = aliases_cat2
    else:
        raise ValueError()

    if all_filenames == "all":
        all_filenames = list(aliases.keys())
        all_rma = [all_rma[0]] * len(all_filenames)
        force_rot_pre = [force_rot_pre[0]] * len(all_filenames)
    elif isinstance(all_filenames, ListConfig):
        pass
    elif isinstance(all_filenames, int):
        idx = all_filenames
        print(f"FILENAME: {idx=}")
        filename = list(aliases.keys())[idx]
        print(f"FILENAME: {idx=}, {filename=}")
        all_filenames = [filename]
    else:
        assert isinstance(all_filenames, str)
        all_filenames = [all_filenames, ]
    print(f"{all_filenames=}")

    all_obs_id = [aliases[fn] for fn in all_filenames]

    save_metrics = cfg_grid.save_metrics
    assert len(all_filenames) == len(all_obs_id)
    assert len(all_filenames) == len(all_rma)
    assert len(all_filenames) == len(force_rot_pre)

    assert cfg.data.detection.name == "single"
    N_rot = len(force_rot_post)
    N_n_frames = len(all_n_frames)
    N_obs_id = len(all_obs_id)

    all_df = {k: np.zeros((N_obs_id, N_rot, N_n_frames)) for k in k_filter}
    all_df["force_rot_post"] = force_rot_post
    all_df["n_frames"] = all_n_frames
    all_df["obs_id"] = all_obs_id
    all_df["range_max_adaptive"] = all_rma

    all_metrics = {}
    all_snr = {}
    for i in range(N_obs_id):
        for j in range(N_rot):
            for k in range(N_n_frames):
                _obs_id = all_obs_id[i]
                _fn = all_filenames[i]
                _rma = all_rma[i]
                # _rot = all_rot[j]
                _n_frames = all_n_frames[k]
                _force_rot_pre = force_rot_pre[i]
                _force_rot_post = force_rot_post[j]
                txt = (
                    f"\n#### GRID ITER: "
                    f"filename={_fn}, "
                    f"obs_id={_obs_id}, "
                    f"rma={_rma}, "
                    f"force_rot_post={_force_rot_post}, "
                    f"n_frames={_n_frames}\n"
                )
                print(txt)

                cfg.data.detection.obs_id = _obs_id
                cfg.data.detection.range_max_adaptive = _rma
                cfg.data.detection.force_rot_pre = _force_rot_pre
                cfg.data.detection.force_rot_post = _force_rot_post
                cfg.data.detection.force_n_frames_post = _n_frames
                metrics, snr = eval_detection(cfg)
                for key, val in metrics.items():
                    if key not in k_filter:
                        continue
                    all_df[key][i, j, k] = val
                filename = f"{_fn}_nframes{_n_frames}_rot{_force_rot_post}"
                all_metrics[filename] = metrics
                all_snr[filename] = snr
                # if save_metrics:
                    # filename = f"{_fn}_{_n_frames}_{_force_rot_post}.pt"
                    # all_metrics[filename] = metrics
                    # path_metrics = os.path.join(os.getcwd(), filename)
                    # torch.save(metrics, path_metrics)
                    # print(f"Metrics saved to {path_metrics!r}")
                    # # breakpoint()

    if save_metrics:
        path_metrics = os.path.join(os.getcwd(), "all_metrics.pt")
        torch.save(all_metrics, path_metrics)
        print(f"Metrics saved to {path_metrics!r}")
    if cfg_grid.save_snr:
        path_snr = os.path.join(os.getcwd(), "all_snr.pt")
        torch.save(all_snr, path_snr)
        print(f"SNR saved to {path_snr!r}")

    cwd = os.getcwd()
    path_grid = os.path.join(cwd, "grids.pt")

    torch.save(all_df, path_grid)
    print(f"Grids saved to {path_grid}")

    name = cfg.name
    txt = f"{name!r}: {path_grid!r},"
    print(txt)
    # breakpoint()
