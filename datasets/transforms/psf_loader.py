import torch
import pickle
import os
import numpy as np


def preprocess_psf(psf, crop, crop_bg):
    """
    Args:
    -----
    * psf: np.array(h, w)
    * crop: int
    * crop_bg: int
    """
    H = 64
    C = psf.shape[0]
    assert psf.ndim == 3, f"{psf.shape=}"
    assert psf.shape[-1] == H
    assert crop < crop_bg
    assert crop % 2 == 0
    assert crop_bg % 2 == 0

    mask_in = np.zeros((H, H), dtype=bool)
    mask_in[
        (H - crop) // 2 : (H + crop) // 2,
        (H - crop) // 2 : (H + crop) // 2,
    ] = 1

    mask_bg = np.ones((H, H), dtype=bool)
    mask_bg[
        (H - crop_bg) // 2 : (H + crop_bg) // 2,
        (H - crop_bg) // 2 : (H + crop_bg) // 2,
    ] = 0
    psf_out = []
    for c in range(C):
        bg_mean = np.mean(psf[c, mask_bg])
        psf_c = psf[c, mask_in].reshape((crop, crop))
        psf_c -= bg_mean
        psf_out.append(psf_c)

    psf_out = np.stack(psf_out)

    # normalize
    # amplitude = np.max(psf)
    # psf = psf / amplitude

    # psf = np.clip(psf, a_min=0, a_max=1)
    assert psf_out.shape[-1] == crop
    assert psf_out.ndim == 3
    assert psf_out.shape[0] == C

    return psf_out


class PSFLoader:
    """Load and preprocess PSF, yields amplitude for normalization"""

    def __init__(self, path_db, crop, crop_bg):
        path_psfs = os.path.join(path_db, "psfs.pkl")
        with open(path_psfs, "rb") as f:
            self.psfs = pickle.load(f)
        assert crop % 2 == 0
        assert crop_bg % 2 == 0
        self.crop = crop
        self.crop_bg = crop_bg

    def __call__(self, item):
        obs_id = item["obs_id"]
        channel_idx = item["c"]

        psf = self.psfs[obs_id]
        # (2, h, w)

        # psf = psf[channel_idx]
        psf = np.stack([psf[c] for c in channel_idx])
        # (C, h, w)


        # psf, amplitude = preprocess_psf(
        psf = preprocess_psf(
            psf=psf, crop=self.crop, crop_bg=self.crop_bg
        )
        # (C, h', w')
        item["psf"] = psf.astype(np.float32)
        # item["amplitude"] = amplitude

        return item
