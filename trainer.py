import torch
import numpy as np
import gc
import copy
from torch.optim import AdamW, lr_scheduler
from collections import defaultdict
import os

from datasets.dataloaders import get_dataloaders
from models import get_model
from detection.utils import MultiModelDetectionEvaluator
from utils.misc import get_timestamp, ContextPlaceHolder, set_seed
from utils.logger import WandBLogger
from losses import NLLLoss, MODELCOLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


class Trainer:
    def __init__(
        self,
        model,
        n_iter,
        lr,
        lr_step,
        lr_gamma,
        weight_decay,
        log_interval,
        val_interval,
        detection_interval,
        save_interval,
        save_steps,
        detection_evaluator,
        logger,
        use_amp,
        loss_type,
        **kwargs,
    ):
        self.model = model
        self.n_iter = n_iter
        self.lr = float(lr)
        print(f"[Trainer] {self.lr=}")
        self.weight_decay = weight_decay

        print(f"[Trainer] {self.weight_decay=}")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.use_amp = use_amp
        print(f"[Trainer] {self.use_amp=}")

        self.lr_sched = lr_scheduler.StepLR(
            self.optimizer, step_size=lr_step, gamma=lr_gamma
        )
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.detection_interval = detection_interval
        self.save_steps = save_steps
        self.logger = logger
        self.detection_evaluator = detection_evaluator

        if loss_type == "modelco":
            self.loss_module = MODELCOLoss()
        elif loss_type == "nll":
            self.loss_module = NLLLoss()
        else:
            raise ValueError(f"{loss_type=} not recognized")

    def run(self, train_dataloader, val_dataloader=None):
        self.model.train()

        metrics_train = defaultdict(list)
        for i in range(self.n_iter):

            if (
                self.val_interval is not None
                and (i % self.val_interval == 0)
                and val_dataloader
            ):
                self.run_val(val_dataloader, step=i)
            if (
                self.detection_interval is not None
                and i % self.detection_interval == 0
            ):
                self.run_detection(step=i)

            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

            batch = next(train_dataloader)

            batch = to_device(batch)

            y = batch["y"]
            s_0 = batch["s_0"]
            # (bs, C, T, H, W)
            lbda = batch["lbda"]
            # (b, C)

            rot = batch["rot"]
            psf = batch["psf"]
            mask_temporal = batch["mask_temporal"]
            mask = batch["mask"].squeeze(2)
            # (bs, T, H, W)

            psf_max = torch.max(psf)
            y = y / psf_max
            s_0 = s_0 / psf_max
            psf = psf / psf_max

            coords = batch["coords"]
            n_sources = batch["n_sources"].item()
            coords = coords[:, :n_sources, :]

            with torch.no_grad():
                targets = self.model.get_target(
                    y=y,
                    s_0=s_0,
                    rot=rot,
                    mask=mask,
                    mask_temporal=mask_temporal,
                )
            # (b, T, H, W)

            if self.use_amp:
                context = torch.cuda.amp.autocast()
            else:
                context = ContextPlaceHolder()
            with context:
                pred = self.model(
                    x=y,
                    rot=rot,
                    psf=psf,
                    s_0=s_0,
                    lbda=lbda,
                    mask_temporal=mask_temporal,
                    mask=mask,
                )
                # (bs, H, W)

                loss = self.loss_module(
                    **pred,
                    **targets,
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.lr_sched.step()

            metrics_train["train_loss"].append(loss.item())

            if i % self.log_interval == 0:
                to_log = {}
                for k, v in metrics_train.items():
                    to_log[k] = np.mean(v)
                to_log["iterations"] = i
                self.log(to_log, step=i)
                metrics_train = defaultdict(list)

            if i in self.save_steps:
                self.save_ckpt(txt=f"{i}")
            if self.save_interval:
                if ((i + 1) % self.save_interval) == 0:
                    self.save_ckpt(txt=f"{i}")

        if val_dataloader:
            self.run_val(val_dataloader, step=i)
        self.run_detection(step=i)
        self.run_charac(step=i)
        self.save_ckpt(txt=f"{i}")
        self.save_error_register()

    def log(self, kvs, snr=None, step=None):
        if self.logger:
            self.logger.write_kvs(kvs, snr=snr, step=step)
        print(f"## {step=}")
        print(f"[{get_timestamp(format='time')}] {step=}")
        for k, v in kvs.items():
            print(f"{k}={v:.5f}")

    def save_ckpt(self, txt):
        print("Saving ckpt..")
        path_folder = os.path.join(os.getcwd(), "ckpt")
        os.makedirs(path_folder, exist_ok=True)
        name_ckpt = f"ckpt_{txt}.pt"
        path_ckpt = os.path.join(path_folder, name_ckpt)
        path_latest = os.path.join(path_folder, "ckpt_latest.pt")

        torch.save(
            {
                "net": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "sched": self.lr_sched.state_dict(),
            },
            path_ckpt,
        )
        print(f"Saved ckpt to {path_ckpt}")
        if os.path.exists(path_latest):
            os.unlink(path_latest)
        os.symlink(src=name_ckpt, dst=path_latest)

    def run_val(self, val_dataloader, step):
        print(f"Running validation ({step=})..")
        print(f"Len val_dataloader: {len(val_dataloader)}")
        self.model.eval()
        val_mse = []
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                gc.collect()
                batch = to_device(batch)

                y = batch["y"]
                s_0 = batch["s_0"]
                # (b, C, T, H, W)

                rot = batch["rot"]
                # (b, T)

                psf = batch["psf"]
                # (b, C, h, w)

                lbda = batch["lbda"]
                # (b, C)

                mask = batch["mask"].squeeze(2)
                mask_temporal = batch["mask_temporal"]
                coords = batch["coords"]
                n_sources = batch["n_sources"].item()
                coords = coords[:, :n_sources, :]

                psf_max = torch.max(psf)
                y = y / psf_max
                s_0 = s_0 / psf_max
                psf = psf / psf_max

                targets = self.model.get_target(
                    y=y,
                    s_0=s_0,
                    rot=rot,
                    mask=mask,
                    mask_temporal=mask_temporal,
                )

                if self.use_amp:
                    context = torch.cuda.amp.autocast()
                else:
                    context = ContextPlaceHolder()
                with context:
                    pred = self.model(
                        x=y,
                        rot=rot,
                        psf=psf,
                        s_0=s_0,
                        lbda=lbda,
                        mask_temporal=mask_temporal,
                    )
                    # (b, c, h, w)

                    loss = self.loss_module(
                        **pred,
                        **targets,
                    )
                    # breakpoint()
                val_mse.append(loss.item())

        val_metrics = {"val_mse": np.mean(val_mse)}

        self.log(val_metrics, step=step)
        self.model.train()
        print("Running validation done")

    def run_detection(self, step):
        if self.detection_evaluator is None:
            return
        print("DETECTION EVALUATION..")
        self.model.eval()
        out = self.detection_evaluator(model=self.model)
        metrics = out["metrics"]
        snr = out["snr"]
        self.model.train()
        print("DETECTION EVALUATION DONE")
        det_metrics = {
            k: v
            for k, v in metrics.items()
            if (k.startswith("auc") or k.startswith("tpr_max"))
        }
        self.log(det_metrics, snr=snr, step=step)


def train(cfg_init):
    cfg = copy.deepcopy(cfg_init)
    set_seed(cfg.seed)

    logger = None
    if cfg.log_wandb:
        print("Initializing WandBLogger..")
        logger = WandBLogger(cfg=cfg, name=cfg.name)
    else:
        print("WandBLogger is disabled")
    print(f"CWD: {os.getcwd()}")

    model = get_model(**cfg.model, image_channels=1)
    dataloaders = get_dataloaders(**cfg.data.train)

    if cfg.train.eval_detection:
        detection_evaluator = MultiModelDetectionEvaluator(
            cfg=cfg, model=model
        )
        detection_evaluator.check()
    else:
        detection_evaluator = None

    trainer = Trainer(
        model=model,
        logger=logger,
        detection_evaluator=detection_evaluator,
        **cfg.train.params,
    )

    trainer.run(
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
    )

    print("Training done")
