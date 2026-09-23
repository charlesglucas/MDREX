import torch
import os
from copy import deepcopy

from datasets.dataloaders import get_dataloaders
from detection.detector import DeterministicDetector
from detection.evaluator import DetectionEvaluator
from sampler import get_sampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config_detection = {
    "1.8deg": {
        "obs_id": "HD_102647__2015-05-31__23-39-09",
        "range_max_adaptive": 6.0 * 8,
    },
    "5.9deg": {
        "obs_id": "HD_188228__2015-06-01__07-40-44",
        "range_max_adaptive": 1.0 * 8,
    },
    "10deg": {
        "obs_id": "HD_206860__2015-09-30__01-47-41",
        "range_max_adaptive": 1.0 * 8,
    },
    "23deg": {
        "obs_id": "HD_216803__2015-06-27__09-36-00",
        "range_max_adaptive": 0.5 * 8,
    },
}


class MultiModelDetectionEvaluator:
    def __init__(self, cfg, model, save_metrics=True):
        print("INITIALIZING MULTIMODEL DETECTION EVALUATOR")
        self.evaluators = {}
        cfg_copy = deepcopy(cfg)

        print(f"{config_detection=}")

        for k, v in config_detection.items():
            obs_id = v["obs_id"]
            range_max_adaptive = v["range_max_adaptive"]
            cfg_copy.data.detection.obs_id = obs_id
            cfg_copy.data.detection.range_max_adaptive = range_max_adaptive
            self.evaluators[k] = ModelDetectionEvaluator(
                cfg=cfg_copy,
                model=model,
                suffix=f"_{k}",
                save_metrics=save_metrics,
            )

    def check(self):
        k0 = list(config_detection.keys())[0]
        self.evaluators[k0].check()

    def __call__(self, model):
        model.eval()
        out = {"metrics": {}, "snr": {}}
        for k in list(config_detection.keys()):
            _out = self.evaluators[k](model)
            metrics = _out["metrics"]
            for m, v in metrics.items():
                out["metrics"][f"{m}_{k}"] = v
            snr = _out["snr"]["default"]
            out["snr"][k] = snr
        return out


class ModelDetectionEvaluator:
    def __init__(self, cfg, model, suffix="", save_metrics=True):
        print("INITIALIZING MODEL DETECTION EVALUATOR")
        self.suffix = suffix
        self.save_metrics = save_metrics

        cwd = os.getcwd()
        self.path_metrics = os.path.join(cwd, f"metrics{suffix}.pt")

        sampler = get_sampler(
            model=model,
            mode=cfg.mode,
            **cfg.sampler,
        )

        dataloaders = get_dataloaders(**cfg.data.detection)
        dataloader_val = dataloaders["val"]

        self.detector = DeterministicDetector(sampler=sampler)

        coords_shift = cfg.data.detection.crop // 2
        self.detection_evaluator = DetectionEvaluator(
            dataloader=dataloader_val,
            coords_shift=coords_shift,
            suffix=suffix,
            **cfg.eval,
        )

    def check(self):
        print("CHECKING DETECTION")
        max_iter = self.detection_evaluator.max_iter
        self.detection_evaluator.max_iter = 1
        self.detection_evaluator.run(detector=self.detector)
        self.detection_evaluator.max_iter = max_iter
        print("DETECTION CHECKED")

    def __call__(self, model):
        model.eval()
        self.detector.sampler.update_model(model)
        out = self.detection_evaluator.run(detector=self.detector)
        snr = out.pop("snr")
        out["snr"] = {"default": snr}
        if self.save_metrics:
            metrics = out["metrics"]
            torch.save(metrics, self.path_metrics)
            print(f"metrics saved to {self.path_metrics}")
        return out
