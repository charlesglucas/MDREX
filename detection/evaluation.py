import torch
import os

from models import get_model
from sampler import get_sampler
from datasets.dataloaders import get_dataloaders
from utils.logger import WandBLogger
from utils.misc import ContextPlaceHolder
from detection.detector import DeterministicDetector
from detection.evaluator import DetectionEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_detection(cfg):
    torch.manual_seed(cfg.seed)

    if cfg.log_wandb:
        logger = WandBLogger(cfg=cfg, name=cfg.name)

    model = get_model(
        **cfg.model,
    )

    sampler = get_sampler(
        model=model,
        mode=cfg.mode,
        **cfg.sampler,
    )

    cfg.data.batch_size = 1
    assert cfg.data.repeat_idx is None
    dataloaders = get_dataloaders(**cfg.data.detection)
    dataloader_val = dataloaders["val"]
    print(f"length dataloader: {len(dataloader_val)}")

    detector = DeterministicDetector(sampler=sampler)

    coords_shift = cfg.data.crop // 2
    detection_evaluator = DetectionEvaluator(
        dataloader=dataloader_val,
        coords_shift=coords_shift,
        **cfg.eval,
    )

    if cfg.eval.use_amp:
        context = torch.cuda.amp.autocast()
    else:
        context = ContextPlaceHolder()
    with context:
        results = detection_evaluator.run(detector=detector)
    metrics = results["metrics"]
    snr = results["snr"]
    cwd = os.getcwd()
    path_metrics = os.path.join(cwd, "metrics.pt")
    torch.save(metrics, path_metrics)
    print(f"metrics saved to {path_metrics}")

    if cfg.log_wandb:
        logger.log_eval_metrics(metrics)
        logger.close()

    return metrics, snr
