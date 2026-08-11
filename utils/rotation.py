import torch
from torch.nn import functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rot_matrix(theta):
    """
    Args:
    -----
    * theta: (b, ) angles in rad

    Returns:
    --------
    * rot_mat: (b, 2, 2)
    """
    theta = -theta
    _cos = torch.cos(theta)
    _sin = torch.sin(theta)
    out = torch.stack([_cos, -_sin, _sin, _cos])
    return out.T.view(-1, 2, 2)


class BatchRotationOperator:
    def __init__(
        self,
        device,
        out_size,
        in_size,
        zero_init,
        mode="bilinear",
        dtype=torch.float32,
    ):
        self.device = device
        assert out_size % 2 == 0
        assert in_size % 2 == 0
        self.out_size = out_size
        self.in_size = in_size
        self.mode = mode
        # print(f"[BatchRotationOperator] {self.mode=}")
        self.dtype = dtype
        # print(f"[BatchRotationOperator] {self.dtype=}")
        self.zero_init = zero_init
        # print(f"[BatchRotationOperator] {self.zero_init=}")

        yx = np.mgrid[: self.out_size, : self.out_size]
        # (2, out_size, out_size)

        center_out = (self.out_size - 1) / 2
        center_in = (self.in_size - 1) / 2

        yx_centered = yx - center_out
        # (2, out_size, out_size)
        yx_centered_scaled = yx_centered / center_in

        self.u = torch.tensor(
            yx_centered_scaled, dtype=self.dtype, device=self.device
        )

        self.shift_center = 0.5 / center_in

    def forward(self, mode="warp", **kwargs):
        if mode == "warp":
            return self.warp(**kwargs)
        else:
            raise ValueError(f"mode not recognized ({mode=})")

    def warp(self, x, rot, debug=False):
        """
        Forward model

        Args:
        -----
        * x: (b, t, H_in, W_in)
        * rotations: (b, t) in degrees

        Returns:
        --------
        * x_rotated: (b, t, H_out, W_out)
        """
        b, c, H, W = x.shape
        brot, t = rot.shape
        assert H == self.in_size, f"{H} != {self.in_size}"
        assert W == self.in_size
        assert b == brot, f"{b=}, {brot=}"
        # t = rot.shape[1]

        if self.zero_init:
            rot = rot - rot[:, 0].view(-1, 1)
        rot = torch.deg2rad(rot)
        # (b, t)

        u = self.u - self.shift_center
        # (2, h, w)

        x = x.reshape(-1, 1, H, W)
        # (b * t, 1, H, W)
        self.rot = rot.flatten()

        rot_mat = rot_matrix(self.rot)
        # (b * t, 2, 2)

        u = u.flatten(start_dim=1)[None, ...]
        # (1, 2, h * w)

        u = rot_mat @ u
        # (b * t, 2, h * w)

        u = u.view(b * t, 2, self.out_size, self.out_size)
        # (b * t, 2, h, w)

        u = u.permute(0, 3, 2, 1)
        # (b * t, w, h, 2)

        u = u + self.shift_center

        x = F.grid_sample(
            input=x,
            grid=u,
            mode=self.mode,
            padding_mode="zeros",
            align_corners=True,
        )
        # (b * t, 1, h, w)

        x = x.reshape(b, t, self.out_size, self.out_size)
        # (b, t, h, w)

        if debug:
            return x, u

        return x
