import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_padding(full_size, kernel_size, stride):
    if (full_size - kernel_size) % stride == 0:
        p = 0
    else:
        k = np.ceil((full_size - kernel_size) / stride)
        p = (kernel_size + k * stride - full_size) / 2
        assert p % 1 == 0, f"{p=}"
        p = int(p)
    assert ((full_size + 2 * p - kernel_size) / stride) % 1 == 0
    return p


def get_divisor(size, kernel_size, stride):
    inputs = torch.ones((1, 1, size, 1))

    patches = torch.nn.functional.unfold(
        inputs,
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
    )
    rec = torch.nn.functional.fold(
        patches,
        output_size=(size, 1),
        kernel_size=(kernel_size, 1),
        stride=(stride, 1),
    )
    return 1 / rec.flatten()


def extract_patches_3d(x, kernel_size, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(
        dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1
    ):
        dim_out = (
            dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1
        ) // dim_stride + 1
        return dim_out

    channels = x.shape[1]

    d_dim_in = x.shape[2]
    h_dim_in = x.shape[3]
    w_dim_in = x.shape[4]
    d_dim_out = get_dim_blocks(
        d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0]
    )
    h_dim_out = get_dim_blocks(
        h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1]
    )
    w_dim_out = get_dim_blocks(
        w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2]
    )
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    # (B, C, D, H, W)
    x = x.view(-1, channels, d_dim_in, h_dim_in * w_dim_in)
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(
        x,
        kernel_size=(kernel_size[0], 1),
        padding=(padding[0], 0),
        stride=(stride[0], 1),
        dilation=(dilation[0], 1),
    )
    # (B, C * kernel_size[0], d_dim_out * H * W)

    x = x.view(-1, channels * kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(
        x,
        kernel_size=(kernel_size[1], kernel_size[2]),
        padding=(padding[1], padding[2]),
        stride=(stride[1], stride[2]),
        dilation=(dilation[1], dilation[2]),
    )
    # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

    x = x.view(
        -1,
        channels,
        kernel_size[0],
        d_dim_out,
        kernel_size[1],
        kernel_size[2],
        h_dim_out,
        w_dim_out,
    )
    # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.contiguous().view(
        -1, channels, kernel_size[0], kernel_size[1], kernel_size[2]
    )
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    return x


def combine_patches_3d(
    x, kernel_size, output_shape, padding=0, stride=1, dilation=1
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    def get_dim_blocks(
        dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1
    ):
        dim_out = (
            dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1
        ) // dim_stride + 1
        return dim_out

    channels = x.shape[1]
    d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
    d_dim_in = get_dim_blocks(
        d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0]
    )
    h_dim_in = get_dim_blocks(
        h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1]
    )
    w_dim_in = get_dim_blocks(
        w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2]
    )
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

    x = x.view(
        -1,
        channels,
        d_dim_in,
        h_dim_in,
        w_dim_in,
        kernel_size[0],
        kernel_size[1],
        kernel_size[2],
    )
    # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
    # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

    x = x.contiguous().view(
        -1,
        channels * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2],
        h_dim_in * w_dim_in,
    )
    # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

    x = torch.nn.functional.fold(
        x,
        output_size=(h_dim_out, w_dim_out),
        kernel_size=(kernel_size[1], kernel_size[2]),
        padding=(padding[1], padding[2]),
        stride=(stride[1], stride[2]),
        dilation=(dilation[1], dilation[2]),
    )
    # (B, C * kernel_size[0] * d_dim_in, H, W)

    x = x.view(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
    # (B, C * kernel_size[0], d_dim_in * H * W)

    x = torch.nn.functional.fold(
        x,
        output_size=(d_dim_out, h_dim_out * w_dim_out),
        kernel_size=(kernel_size[0], 1),
        padding=(padding[0], 0),
        stride=(stride[0], 1),
        dilation=(dilation[0], 1),
    )
    # (B, C, D, H * W)

    x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
    # (B, C, D, H, W)

    return x


class TemporalPatchesHandler:
    def __init__(
        self,
        patch_size,
        stride,
        groups,
        temporal_stride=16,
        temporal_length=32,
        frame_size=256,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.frame_size = frame_size
        self.groups = groups
        self.temporal_length = temporal_length
        self.temporal_stride = temporal_stride
        self.unfolder = None
        self.folder = None
        self.divisor_t = None
        self.divisor_h = None
        self.divisor_w = None

        self.log(f"spatial stride: {self.stride}")
        self.log(f"spatial patch_size: {self.patch_size}")
        assert stride <= patch_size, f"{stride=}, {patch_size=}"

    def log(self, x):
        print(f"[SpatialPatchesHandler ({self.patch_size})] {x}")

    def to_patches(self, x):
        self.bs, self.T, self.C, self.H, self.W = x.shape

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # # (bs, C, T, H, W)

        x = extract_patches_3d(
            x,
            kernel_size=(
                self.temporal_length,
                self.patch_size,
                self.patch_size,
            ),
            stride=(
                self.temporal_stride,
                self.stride,
                self.stride,
            ),
        )
        # (bs * dt * dh * dw, C, t, ps, ps)

        x = x.permute(0, 2, 1, 3, 4)
        # (bs * dt * dh * dw, t, C, ps, ps)

        x = x.reshape(
            x.shape[0],
            self.temporal_length,
            self.C * self.patch_size**2,
        )
        # (bs * dt * dh * dw, t, C * ps ** 2)

        return x

    def from_patches(self, x):
        bsp, t, CK = x.shape

        x = x.view(bsp, t, 1, self.patch_size, self.patch_size)
        # (bs * dt * dh * dw, t, C, ps, ps)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # (bs * dt * dh * dw, C, t, ps, ps)

        x = combine_patches_3d(
            x,
            output_shape=(self.bs, self.C, self.T, self.H, self.W),
            kernel_size=(
                self.temporal_length,
                self.patch_size,
                self.patch_size,
            ),
            stride=(
                self.temporal_stride,
                self.stride,
                self.stride,
            ),
        )
        # (bs, C, T, H, W)

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # (bs, T, C, H, W)

        return x

    def init_folder(self, T, H, W, device):
        self.divisor_t = get_divisor(
            size=T,
            kernel_size=self.temporal_length,
            stride=self.temporal_stride,
        ).view(1, -1, 1, 1, 1)
        self.divisor_h = get_divisor(
            size=H,
            kernel_size=self.patch_size,
            stride=self.stride,
        ).view(1, 1, 1, -1, 1)
        self.divisor_w = get_divisor(
            size=W,
            kernel_size=self.patch_size,
            stride=self.stride,
        ).view(1, 1, 1, 1, -1)

    def unfold(self, x):
        """
        Args:
        -----
        * x: torch.tensor(b, T, C, H, W)

        Returns:
        --------
        * x: torch.tensor(b, dh * dw, T, C * s * s)
        """
        self.b, T, self.C, H, W = x.shape
        assert H == self.frame_size
        assert W == self.frame_size
        assert T >= self.C

        # x = x.permute(0, 2, 1, 3, 4)
        # (bs, C, T, H, W)

        if self.folder is None:
            self.init_folder(T=T, H=H, W=W, device=x.device)

        # xp = self.to_patches(x)
        x = self.to_patches(x)
        print(f"{x.shape=}")
        # (bsp, t, C * K)

        x = x.view(
            self.b, -1, self.temporal_length, self.C * self.patch_size**2
        )
        # (bsp, t, C * K)

        return x

    def fold(self, x):
        """
        Args:
        -----
        * x: torch.tensor(b, dh * dw, G, T, s * s)

        Returns:
        --------
        * x: torch.tensor(b, T, C, H, W)
        """
        # (b * dh * dw, C, T, s, s)
        # _, T, C, _, _ = x.shape
        bs, dthw, t, CK = x.shape
        # assert T >= self.C
        # assert self.groups == G

        x = x.view(bs * dthw, t, CK)
        # (bs * dthw, t, CK)

        x = self.from_patches(x)
        # (bs, T, C, H, W)

        x = x * self.divisor_t
        x = x * self.divisor_h
        x = x * self.divisor_w
        # (bs, C, T, H, W)

        x = x.permute(0, 2, 1, 3, 4)
        # (bs, T, C, H, W)

        return x


class SpatialPatchesHandler:
    def __init__(self, patch_size, stride, groups, frame_size=256):
        self.patch_size = patch_size
        self.stride = stride
        self.frame_size = frame_size
        self.groups = groups
        self.unfolder = None
        self.folder = None
        self.divisor_inv = None

        self.log(f"spatial stride: {self.stride}")
        self.log(f"spatial patch_size: {self.patch_size}")
        assert stride <= patch_size, f"{stride=}, {patch_size=}"

    def log(self, x):
        print(f"[SpatialPatchesHandler ({self.patch_size})] {x}")

    def init_folder(self, H, W, device):
        p_h = get_padding(
            full_size=H, kernel_size=self.patch_size, stride=self.stride
        )
        p_w = get_padding(
            full_size=W, kernel_size=self.patch_size, stride=self.stride
        )
        padding = (p_h, p_w)
        self.log(f"{padding=}")

        self.unfolder = nn.Unfold(
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=padding,
        )
        self.folder = nn.Fold(
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=padding,
        )
        input_ones = torch.ones(
            (1, 1, H, W), dtype=torch.float32, device=device
        )
        # divisor = self.folder(self.unfolder(input_ones))[None, ...]
        divisor = self.folder(self.unfolder(input_ones))
        self.divisor_inv = 1 / divisor
        # (1, 1, H, W)

    def unfold(self, x):
        """
        Args:
        -----
        * x: torch.tensor(b, C, T, H, W)

        Returns:
        --------
        * x: torch.tensor(b, dh * dw, T, C * s * s)
        """
        # self.b, T, self.C, H, W = x.shape
        self.b, self.C, T, H, W = x.shape
        assert H == self.frame_size
        assert W == self.frame_size
        assert T >= self.C

        if self.folder is None:
            self.init_folder(H, W, device=x.device)

        x = x.view(self.b, self.C * T, H, W)
        # (b, C * T, H, W)

        x = self.unfolder(x)
        # (b, C * T * s * s, dh * dw)

        dhw = x.shape[-1]

        x = x.permute(0, 2, 1)
        # (b, dh * dw, C * T * s * s)

        x = x.reshape(self.b, dhw, self.C * T, self.patch_size**2)
        # (b, dh * dw, C * T,  s * s)

        return x

    def fold(self, x):
        """
        Args:
        -----
        * x: torch.tensor(b, dh * dw, G, T, s * s)

        Returns:
        --------
        * x: torch.tensor(b, T, C, H, W)
        """
        # (b * dh * dw, C, T, s, s)
        # _, T, C, _, _ = x.shape
        bs, dhw, T, G, K = x.shape
        # assert T >= self.C
        assert self.groups == G

        x = x.view(bs, dhw, -1)
        # (b, dh * dw, T * G * s * s)

        x = x.permute(0, 2, 1)
        # (b, T * G * s *s, dh * dw)

        x = self.folder(x)
        # (b, T * G, H, W)

        _, _, H, W = x.shape

        x = x * self.divisor_inv
        # (b, T * G, H, W)

        x = x.view(self.b, T, self.groups, H, W)
        # (b, T, G, H, W)

        return x
