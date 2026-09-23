import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy
from hydra.utils import to_absolute_path

from detection.calibrator import Calibrator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLAT_SPECTRUM = [1.0, 1.0]


class ExoMILDSampler:
    TYPE = "deterministic"

    def __init__(
        self,
        model,
        skip_filter,
        weights_spectrum=FLAT_SPECTRUM,
        **kwargs,
    ):
        self.update_model(model)
        self.skip_filter = skip_filter
        if weights_spectrum is not None:
            self.weights_spectrum = torch.tensor(
                weights_spectrum, device=device
            )
        else:
            self.weights_spectrum = None
        print(f"[BangerSampler] {self.skip_filter=}")

    def update_model(self, model):
        self.model = model
        self.model.eval()

    def run(self, y, psf, rot, s_0, lbda=None, mask=None, **kwargs):
        """
        Args:
        -----
        * y: torch.tensor(b, C, H, W)
        * psf: torch.tensor(b, h, w)
        * rot: torch.tensor(b, C)
        * idx: torch.tensor(b,)
        * amplitude: torch.tensor(b,)

        Returns:
        --------
        * alpha: torch.tensor(b, H', W')
        * sigma_alpha: torch.tensor(b, H', W')
        """
        # assert psf.ndim == 3
        assert psf.ndim == 4

        bs, C, T, H, W = y.shape
        with torch.no_grad():
            out = self.model(
                x=y,
                # s_0=y,
                s_0=s_0,
                psf=psf,
                t=None,
                rot=rot,
                skip_filter=self.skip_filter,
                mask=mask,
                lbda=lbda,
                output="sample",
            )
        alpha = out["alpha"]
        alpha_raw = out["alpha_raw"]
        sigma = out["sigma"]
        bs, C, H, W = alpha.shape

        a = 1 / sigma**2
        b = alpha * a
        b_raw = alpha_raw * a
        a_new = torch.mean(a, dim=1)
        b_new = torch.mean(b, dim=1)
        b_raw_new = torch.mean(b_raw, dim=1)

        alpha = b_new / a_new
        alpha_raw = b_raw_new / a_new
        sigma = 1 / torch.sqrt(a_new)

        snr = alpha / sigma
        if snr.isnan().any():
            breakpoint()

        return {
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "sigma_alpha": sigma,
            # "a": a_new,
            # "b": b_new,
        }

    def run_inference(self, y, rot, mask_temporal, psf, lbdas, **kwargs):
        print(f"[Sampler] {y.shape=}")
        bs, C, T, H, W = y.shape
        assert bs == 1
        # channels_str = "".join([str(i) for i in range(C)])
        y = y[:, :, mask_temporal.flatten(), :, :]
        lbdas = lbdas.view(1, -1)
        out = self.run(
            y=y,
            s_0=y,
            rot=rot,
            # mask_temporal=mask_temporal,
            psf=psf,
            lbda=lbdas,
            # idx=0,
            # mode="sb",
        )
        snr = out["alpha"] / out["sigma_alpha"]
        snr_raw = out["alpha_raw"] / out["sigma_alpha"]
        out["snr"] = snr
        out["snr_raw"] = snr_raw
        results = {
            "snr": snr.cpu().numpy(),
            "snr_raw": snr_raw.cpu().numpy(),
            "alpha": out["alpha"].cpu().numpy(),
            "alpha_raw": out["alpha_raw"].cpu().numpy(),
            "sigma_alpha": out["sigma_alpha"].cpu().numpy(),
        }

        return results


def find_latest_ckpt(path_src):
    iter_max = -1
    path_latest = None
    for path in Path(path_src).rglob("*ckpt*.pt"):
        path = str(path)
        if "latest" in path.split("/")[-1]:
            continue
        n_iter = int(path.split("ckpt_")[-1].split(".pt")[0])
        if n_iter > iter_max:
            path_latest = path
    assert path_latest is not None
    return path_latest


class ExoMILDEnsemblingSampler:
    TYPE = "deterministic"

    def __init__(
        self,
        model,
        path_ckpts=None,
        path_calib=None,
        **kwargs,
    ):
        self.path_ckpts = path_ckpts
        self.path_calib = path_calib

        if self.path_ckpts is None:
            print(f"[ExoMILDEnsemblingSampler] No checkpoint to load")
            self.samplers = [ExoMILDSampler(model=model, **kwargs)]
        else:
            self.path_ckpts = to_absolute_path(self.path_ckpts)
            print(
                "[ExoMILDEnsemblingSampler] Loading checkpoints "
                f"from {path_ckpts=}"
            )
            folders = os.listdir(self.path_ckpts)

            self.samplers = []

            for folder in folders:
                path_ckpt = find_latest_ckpt(
                    os.path.join(self.path_ckpts, folder)
                )
                print(f"{path_ckpt=}")
                _model = copy.deepcopy(model)
                state_dict = torch.load(path_ckpt)["net"]
                _model.load_state_dict(state_dict)

                sampler = ExoMILDSampler(model=_model, **kwargs)
                self.samplers.append(sampler)

        if self.path_calib is not None:
            print(
                "[ExoMILDEnsemblingSampler] Using calibration "
                f"from {self.path_calib=}"
            )
            self.calibrator = Calibrator(
                path_calib=to_absolute_path(self.path_calib)
            )

    def run(self, **kwargs):
        """
        Args:
        -----
        * y: torch.tensor(b, C, H, W)
        * psf: torch.tensor(b, h, w)
        * rot: torch.tensor(b, C)
        * idx: torch.tensor(b,)
        * amplitude: torch.tensor(b,)

        Returns:
        --------
        * alpha: torch.tensor(b, H', W')
        * sigma_alpha: torch.tensor(b, H', W')
        """

        all_a = []
        all_b = []
        all_b_raw = []
        for i, sampler in enumerate(self.samplers):
            out = sampler.run(**kwargs)
            alpha = out["alpha"]
            alpha_raw = out["alpha_raw"]
            sigma = out["sigma_alpha"]
            a = 1 / sigma**2
            b = alpha * a
            b_raw = alpha_raw * a
            all_a.append(a)
            all_b.append(b)
            all_b_raw.append(b_raw)

        a = torch.stack(all_a).mean(dim=0)
        b = torch.stack(all_b).mean(dim=0)
        b_raw = torch.stack(all_b_raw).mean(dim=0)
        alpha = b / a
        alpha_raw = b_raw / a
        sigma = 1 / torch.sqrt(a)

        return {
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "sigma_alpha": sigma,
        }

    def run_inference(self, y, rot, mask_temporal, psf, lbdas, **kwargs):
        print(f"[Sampler] {y.shape=}")
        bs, C, T, H, W = y.shape
        assert bs == 1
        # channels_str = "".join([str(i) for i in range(C)])
        y = y[:, :, mask_temporal.flatten(), :, :]
        lbdas = lbdas.view(1, -1)
        out = self.run(
            y=y,
            s_0=y,
            rot=rot,
            # mask_temporal=mask_temporal,
            psf=psf,
            lbda=lbdas,
            output="sample",
        )
        snr = out["alpha"] / out["sigma_alpha"]
        snr_raw = out["alpha_raw"] / out["sigma_alpha"]
        results = {
            "alpha": out["alpha"].cpu().numpy(),
            "alpha_raw": out["alpha_raw"].cpu().numpy(),
            "snr": snr.cpu().numpy(),
            "snr_raw": snr_raw.cpu().numpy(),
            "sigma_alpha": out["sigma_alpha"].cpu().numpy(),
        }
        if self.path_calib is not None:
            results["snr_calib"] = self.calibrator(snr.cpu().numpy())

        return results
