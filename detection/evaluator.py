import torch
from utils.misc import set_seed
import numpy as np
import sys
from tqdm import tqdm
from detection.metrics import MetricsSuite
from utils.misc import get_timestamp, ContextPlaceHolder
from utils.prefix_log import TimeLogMixin
from time import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetectionEvaluator(TimeLogMixin):
    def __init__(
        self,
        dataloader,
        # mixing_operator,
        dilation_t,
        offset_t,
        use_amp,
        obj_scale=1.0,
        seed=0,
        max_iter=None,
        **kwargs,
    ):
        self.dataloader = dataloader
        # self.mixing_operator = mixing_operator

        # self.normalizer = mixing_operator.normalizer
        # self.rotator = mixing_operator.rotator
        self.obj_scale = obj_scale
        self.seed = seed
        self.max_iter = max_iter
        self.dilation_t = dilation_t
        self.offset_t = offset_t
        self.log(f"dilation_t eval: {self.dilation_t}")
        self.log(f"offset_t eval: {self.offset_t}")
        self.metrics_suite = MetricsSuite(**kwargs)
        self.use_amp = use_amp
        self.log(f"{self.use_amp=}")

    def run(self, detector):
        set_seed(self.seed)

        # metrics = {}
        # for i, batch in enumerate(self.dataloader):
        n_clips = len(self.dataloader)
        self.log(f"Len dataloader: {n_clips}")
        for i, batch in enumerate(self.dataloader):
            if self.max_iter and (i >= self.max_iter):
                break
            # if i > 30:
                # break

            self.log(f"Batch {i}")
            # x_0 = batch["x_0"].to(device)
            y = batch["y"].to(device)
            # (bs, C, H, W)
            # breakpoint()

            # (1, 1, h, w)
            s_0 = batch["s_0"].to(device)
            # (1, T, h, w)

            lbda = batch["lbda"].to(device)
            rot = batch["rot"].to(device)
            mask_temporal = batch["mask_temporal"].to(device)
            # rot = - rot
            # (1, T)
            # mask = batch["mask"].to(device)
            # # (1, T, h, w)
            # breakpoint()
            obs_id = batch.get("obs_id")
            cube_name = batch.get("cube_name")

            rot_range = (rot[0, -1] - rot[0, 0]).item()
            # print(f"{rot_range=}")
            print(f"{obs_id=}, {rot_range=:.2f}")

            # mask_0 = mask[:, 0]
            # # (1, h, w)
            n_sources_real = batch["n_sources_real"].to(device)
            radius_real = batch["radius_real"].to(device)
            coords_real = batch["coords_real"].to(device)
            lbda = batch["lbda"].float().to(device)

            psf = batch["psf"].to(device)

            psf_max = torch.max(psf)
            y = y / psf_max
            s_0 = s_0 / psf_max
            psf = psf / psf_max

            # amplitude = batch["amplitude"].to(device)

            coords = batch["coords"].to(device)
            alpha_coords = batch["alpha"].float().to(device)
            n_sources = batch["n_sources"].to(device)
            idx = batch["idx"].to(device)

            # rot *= 10
            # rot *= 5
            rot *= 1
            # breakpoint()

            if (self.dilation_t > 1) or (self.offset_t > 0):
                y = y[:, self.offset_t :: self.dilation_t, :, :].clone()
                s_0 = s_0[:, self.offset_t :: self.dilation_t, :, :].clone()
                rot = rot[:, self.offset_t :: self.dilation_t].clone()

            # x_0_rot = self.rotator.forward(obj=x_0, rot=rot, debug=False)

            # y = s_0 + x_0_rot * self.obj_scale
            # y_params = self.normalizer.get_params(x=y)
            # y_norm = self.normalizer.forward(x=y, **y_params)

            # plt.imshow(y_norm.cpu()[0, 0])
            # plt.show()

            # breakpoint()
            # cube_3d_viewer(y_norm.cpu().squeeze(0).numpy())
            # breakpoint()

            if self.use_amp:
                context = torch.cuda.amp.autocast()
            else:
                context = ContextPlaceHolder()

            tic = time()
            with context:
                alpha, sigma_alpha = detector.forward(
                    rot=rot,
                    y=y,
                    psf=psf,
                    mask_temporal=mask_temporal,
                    # amplitude=amplitude,
                    s_0=s_0,
                    lbda=lbda,
                    idx=idx,
                    use_tqdm=False,
                    prefix=f"{i:03d}",
                    cube_name=cube_name,
                    obs_id=obs_id,
                )
            tac = time()
            elapsed = tac - tic
            self.log(f"detection done")
            self.metrics_suite.add(
                alpha=alpha.float(),
                sigma_alpha=sigma_alpha.float(),
                n_sources=n_sources,
                coords_gt=coords,
                alpha_coords=alpha_coords,
                coords_real=coords_real,
                radius_real=radius_real,
                n_sources_real=n_sources_real,
                inference_time=elapsed,
            )
            self.log(f"metrics computed")

        metrics = self.metrics_suite.get_metrics()
        return metrics
