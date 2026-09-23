import torch
import numpy as np
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def idx_to_mask(idx, bsp):
    bs, n_sources, T, ppp, ppp = idx.shape
    mask_bsp = torch.zeros(bs, bsp, device=device, dtype=bool)
    mask_bsp[torch.arange(bs), idx.view(bs, -1)] = 1
    # (bs, bsp)

    return mask_bsp


def rot_matrix(theta):
    """
    Args:
    -----
    * theta: (b, T) angles in rad

    Returns:
    --------
    * rot_mat: (b, 2, 2)
    """
    theta = -theta
    _cos = torch.cos(theta)
    _sin = torch.sin(theta)
    out = torch.stack([_cos, -_sin, _sin, _cos], dim=-1)
    # (bs, T, 4)
    bs, T, _ = out.shape

    return out.view(bs, T, 2, 2)


def compute_trajectories(coords, rot, center=128, return_grad=False):
    assert coords.ndim == 3, f"{coords.shape=}"
    assert coords.shape[2] == 2, f"{coords.shape=}"
    assert rot.ndim == 2, f"{rot.shape=}"
    bs, n_sources, _ = coords.shape
    bs, T = rot.shape

    coords_c = (coords - center).view(bs, n_sources, 1, 2, 1)
    # (bs, n_sources, 1, 2, 1)

    theta = torch.deg2rad(rot)
    # (bs, T)

    rot_mat = rot_matrix(theta)
    # (bs, T, 2, 2)

    rot_mat = rot_mat.view(1, T, 2, 2)
    # (1, T, 2, 1) rot_mat is the same for all sources

    coords_c = rot_mat @ coords_c
    # (n_sources, T, 2, 1)

    coords_c = coords_c + center
    # (n_sources, T, 2, 1)
    # print(f"{coords_c[0]=}")
    # breakpoint()
    if return_grad:
        return coords_c, rot_mat

    return coords_c, None


class BatchPSFSampler(nn.Module):
    def __init__(self, psf_size, mode, dilation=1):
        super().__init__()
        self.psf_size = psf_size
        assert mode in ["centered", "all_pixels", "shift"]
        self.mode = mode
        self.dilation = dilation

        yy, xx = np.mgrid[: self.psf_size, : self.psf_size]
        yy = torch.tensor(yy)[None, ...].to(device)
        xx = torch.tensor(xx)[None, ...].to(device)
        # (1, h, w)

        self.register_buffer("yy", yy, persistent=False)
        self.register_buffer("xx", xx, persistent=False)

        if self.mode == "all_pixels":
            x0 = (
                torch.arange(self.psf_size)[None, ...]
                .expand(self.psf_size, -1)
                .reshape(-1, 1, 1)
                .float()
                .to(device)
            )
            y0 = (
                torch.arange(self.psf_size)[..., None]
                .expand(-1, self.psf_size)
                .reshape(-1, 1, 1)
                .float()
                .to(device)
            )
        elif self.mode == "centered":
            x0 = self.psf_size // 2
            y0 = self.psf_size // 2
        elif self.mode == "shift":
            return
        else:
            raise ValueError
        delta_x = (xx - x0)[None, ...].float() * self.dilation
        delta_y = (yy - y0)[None, ...].float() * self.dilation
        # (1, 1, ps * ps, ps, ps)

        self.register_buffer("delta_x", delta_x, persistent=False)
        self.register_buffer("delta_y", delta_y, persistent=False)

    def forward(
        self,
        amplitude,
        sigma_x,
        sigma_y,
        theta,
        offset,
        x0=None,
        y0=None,
        return_jacobian=False,
    ):
        """
        Args:
        ----
            amplitude: (bs,) or (1,)
            x0: (bs,) or (1,)
            y0: (bs,) or (1,)
            sigma_x: (bs,) or (1,)
            sigma_y: (bs,) or (1,)
            theta: (bs,) or (1,)

        Returns:
        -------
            psf: (bs, h * w)
        """
        # amplitude = amplitude.view(-1, 1, 1)
        # sigma_x = sigma_x.view(-1, 1, 1)
        # sigma_y = sigma_y.view(-1, 1, 1)
        # theta = theta.view(-1, 1, 1)
        # offset = offset.view(-1, 1, 1)
        assert amplitude.ndim == 1
        assert sigma_x.ndim == 1
        assert sigma_y.ndim == 1
        assert theta.ndim == 1
        assert offset.ndim == 1
        if self.mode == "shift":
            assert self.xx.ndim == 3
            assert x0 is not None
            assert y0 is not None
            bs, ns = x0.shape
            bs, ns = y0.shape
            delta_x = self.xx - x0.view(bs, ns, 1, 1)
            delta_y = self.yy - y0.view(bs, ns, 1, 1)
        else:
            delta_x = self.delta_x
            delta_y = self.delta_y
            # (1, h * w (pos), h, w)

        a = (torch.cos(theta) ** 2) / (2 * sigma_x**2) + (
            torch.sin(theta) ** 2
        ) / (2 * sigma_y**2)
        b = -(torch.sin(2 * theta)) / (4 * sigma_x**2) + (
            torch.sin(2 * theta)
        ) / (4 * sigma_y**2)
        c = (torch.sin(theta) ** 2) / (2 * sigma_x**2) + (
            torch.cos(theta) ** 2
        ) / (2 * sigma_y**2)
        # (bs)
        a = a.view(-1, 1, 1, 1)
        b = b.view(-1, 1, 1, 1)
        c = c.view(-1, 1, 1, 1)
        offset = offset.view(-1, 1, 1, 1)
        amplitude = amplitude.view(-1, 1, 1, 1)

        exp = torch.exp(
            -(a * delta_x**2 + 2 * b * delta_x * delta_y + c * delta_y**2)
        )
        all_psf = offset + amplitude * exp
        # (bs, h, w)
        if return_jacobian:
            # the grad must be return w.r.t. x_t, y_t, OK !
            # (bs, h, w, 2)
            assert self.mode == "shift"

            pre_x = -2 * a * delta_x - 2 * b * delta_y
            pre_y = -2 * c * delta_y - 2 * b * delta_x
            grad_x0 = amplitude * pre_x * exp
            grad_y0 = amplitude * pre_y * exp
            # (bs, ns, h, w)

            jacobian = torch.stack([grad_y0, grad_x0], dim=-1)
            # (bs, ns, h, w, 2)
        else:
            jacobian = None

        return all_psf, jacobian
