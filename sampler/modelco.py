import torch
import numpy as np

from datasets.transforms.injection import gaussian_2d, fit_gaussian_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelCoSampler:
    TYPE = "deterministic"

    def __init__(self, model):
        self.update_model(model)

    def update_model(self, model):
        self.model = model
        self.model.eval()

    def fit_psf(self, psf):
        assert psf.shape[0] == 1
        psf_params = fit_gaussian_2d(psf[0].detach().cpu().numpy())
        psf_params["offset"] = 0
        psf_params["x0"] = 0
        psf_params["y0"] = 0
        self.psf_params = psf_params

    def sample_psf(self, size=9):
        yy, xx = np.mgrid[:size, :size] - size // 2
        xy = (xx, yy)
        psf = gaussian_2d(xy, **self.psf_params)
        return torch.tensor(psf, dtype=torch.float32, device=device).view(
            size, size
        )

    def run(self, **kwargs):
        with torch.no_grad():
            return self._run(**kwargs)

    def _run(
        self,
        y,
        psf,
        rot,
        idx,
        mask_temporal,
        mode="sb",
        prefix=None,
        s_0=None,
        lbdas=None,
        **kwargs,
    ):
        print(f"{y.shape=}")
        bs, C, T, H, W = y.shape
        assert C == 1, f"Only monospectral supported with MODEL&CO, {C=}"
        # y = y[:, 0]
        # psf = psf[:, 0]

        if psf is not None:
            assert psf.ndim == 4, f"{psf.shape=}"

        # bs, T, H, W = y.shape
        x_hat = self.model.forward_multi(
            x=y,
            psf=psf,
            rot=rot,
            s_0=s_0,
            mask_temporal=mask_temporal,
            mode=mode,
            lbdas=lbdas,
        )["snr"][:, 0]
        sigma = torch.ones_like(x_hat)

        out = {
            "alpha": x_hat.float(),
            "sigma_alpha": sigma.float(),
        }

        return out
