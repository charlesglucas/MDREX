import wandb
import os
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from copy import deepcopy
import numpy as np

from collections.abc import MutableMapping


def dict_subset(d, keys):
    return {k: list(d[k]) for k in keys}


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, ListConfig):
            n = len(v)
            for i in range(n):
                key_i = f"{new_key}{sep}{i}"
                v_i = v[i]
                if isinstance(v_i, MutableMapping):
                    items.extend(flatten_dict(v_i, key_i, sep=sep).items())
                else:
                    items.append((key_i, v_i))
        elif isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class WandBLogger:
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, name=None, cfg=None):
        # wandb.init(project="diffusion", config=self.load_config())

        if cfg is not None:
            cfg = flatten_dict(cfg)
        os.environ["WANDB_CACHE_DIR"] = "./wandb/wandb_cache_dir"
        os.environ["WANDB_DATA_DIR"] = "./wandb/data_cache_dir"
        wandb.init(project="exodiff", name=name, config=cfg)
        print("WANDB logger initialized")

    # def load_config(self):
    # cfg = OmegaConf.load("config.yaml")
    # domain = cfg.domain
    # cfg_flat = OmegaConf.create(
    # {
    # "domain": domain,
    # **cfg.task[domain].params_train,
    # **cfg.model[domain].params,
    # **cfg.diffusion[domain].params,
    # }
    # )
    # return cfg_flat

    def write_kvs(self, kvs, step, snr=None):
        if snr is not None:
            assert isinstance(snr, dict)
            for k, v in snr.items():
                # snr = np.stack(snr)
                # s_max = np.max(snr)
                # s_min = np.min(snr)
                snr_list = [
                    256 * (_snr - np.min(_snr)) / (np.max(_snr) - np.min(_snr))
                    for _snr in v[:5]
                ]
                # snr = 256 * (snr - s_min) / (s_max - s_min)
                images = [wandb.Image(_snr) for _snr in snr_list]
                kvs = deepcopy(kvs)
                kvs[f"snr_{k}"] = images

        wandb.log(kvs, step)

    # def write_kvs(self, kvs):
    # wandb.log(kvs)

    def log_plot(self, name, dictionary):
        keys = list(dictionary.keys())
        assert len(keys) == 2
        data = [
            [x, y] for (x, y) in zip(dictionary[keys[0]], dictionary[keys[1]])
        ]
        table = wandb.Table(data=data, columns=[keys[0], keys[1]])
        wandb.log({name: wandb.plot.line(table, keys[0], keys[1], title=name)})
        print(f"Logged {name} to wandb")

    def log_eval_metrics(self, metrics):
        metrics["snr"] = metrics["thresholds"]

        self.write_kvs({"auc_fdr_tpr": metrics["auc_fdr_tpr"]}, step=None)
        self.write_kvs({"auc_pr": metrics["auc_pr"]}, step=None)

        self.log_plot(
            dictionary=dict_subset(metrics, ["fdr", "tpr"]),
            name="fdr_tpr_curve",
        )
        self.log_plot(
            dictionary=dict_subset(metrics, ["recall", "precision"]),
            name="pr_curve",
        )
        self.log_plot(
            dictionary=dict_subset(metrics, ["snr", "fdr"]),
            name="calibration_curve",
        )

        bins = metrics["contrast"]["bins"] * 0.01225
        contrast = 5 * metrics["contrast"]["mean"]
        d_contrast = {"angular_sep": bins, "contrast": contrast}
        self.log_plot(
            dictionary=d_contrast,
            name="contrast_curve",
        )

    def log_images(self, images, step):
        assert images[0].ndim == 3
        wandb.log({"snr": images}, step=step)

    def close(self):
        wandb.finish()
