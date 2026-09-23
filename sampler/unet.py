import torch

"""Minimal Unet sampler adapter.

This module provides `UnetSampler`, a thin adapter around U-Net style
models that output tensors shaped (bs, T, 1, H, W). The sampler reduces
the temporal/channel dimension and returns the dictionary expected by the
detector/evaluator pipeline (alpha, alpha_raw, sigma_alpha).
"""


class UnetSampler:
    TYPE = "deterministic"

    def __init__(self, model, **kwargs):
        self.update_model(model)

    def update_model(self, model):
        self.model = model
        try:
            self.model.eval()
        except Exception:
            pass

    def _unwrap_model_output(self, out):
        # support models that return a tensor or a dict containing a tensor
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            # common keys: 'alpha','snr','out'
            for k in ("alpha", "snr", "out", "pred"):
                if k in out:
                    return out[k]
            # otherwise return the first tensor-like value
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    return v
        raise ValueError("Unrecognized model output from UNet-like model")

    def run(self, y, **kwargs):
        """Run the model on input `y` and return a dict with tensors.

        Expected `y` shape: (bs, T, 1, H, W) or (bs, C, T, H, W). The
        implementation squeezes the singleton channel dimension and
        averages over the temporal/channel axis to produce `alpha`.
        """
        with torch.no_grad():
            out = self.model(x=y, **kwargs) if hasattr(self.model, "__call__") else self.model(y, **kwargs)
        out = self._unwrap_model_output(out)

        # Expect out shape (bs, T, 1, H, W) or (bs, T, H, W)
        if out.ndim == 5:
            # (bs, T, 1, H, W) -> squeeze dim 2 -> (bs, T, H, W)
            out = out.squeeze(2)
        if out.ndim == 4:
            # reduce temporal/channel axis into a single 2D map
            alpha = out.mean(dim=1)
        else:
            raise ValueError(f"Unexpected model output shape for UnetSampler: {out.shape}")

        sigma = torch.ones_like(alpha)

        return {"alpha": alpha, "alpha_raw": alpha, "sigma_alpha": sigma}

    def run_inference(self, y, rot=None, mask_temporal=None, psf=None, lbdas=None, **kwargs):
        res = self.run(y=y, rot=rot, mask_temporal=mask_temporal, psf=psf, lbda=lbdas, **kwargs)
        alpha = res["alpha"]
        sigma = res["sigma_alpha"]
        snr = alpha / sigma
        return {
            "alpha": alpha.cpu().numpy(),
            "alpha_raw": alpha.cpu().numpy(),
            "snr": snr.cpu().numpy(),
            "snr_raw": snr.cpu().numpy(),
            "sigma_alpha": sigma.cpu().numpy(),
        }
