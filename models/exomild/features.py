import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.misc import make_orthogonal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JacobianAccumulator:
    def __init__(self):
        self.jacobians = []

    def add_jacobian(self, jacobian):
        self.jacobians.append(jacobian)

    def forward_right(self, x):
        bs, bsp, T, K, _ = x.shape

        for jacobian in self.jacobians:
            x = jacobian.forward_right(x)
        return x

    def forward_left(self, x):
        bs, bsp, T, G, fsp, fsp = x.shape
        if G > 1:
            id_block = torch.eye(G * fsp, device=device).view(
                1, 1, 1, G, fsp, G * fsp
            )
            # (1, 1, 1, G, fs, G * fs)
            x = x @ id_block
            # breakpoint()
            # (bs, bsp, T, G, fsp, G * fsp)

        for jacobian in self.jacobians[::-1]:
            x = jacobian.forward_left(x)
        # (bs, bsp, T, G, fsp, K)
        return x


class EfficientJacobianAccumulator:
    def set_jacobian(self, jacobian):
        self.jacobian_cum = jacobian
        self.K = jacobian.shape[-1]

    def forward_left(self, x):
        bs, bsp, T, G, fsp, fsp = x.shape
        if G > 1:
            id_block = torch.eye(G * fsp, device=device).view(
                1, 1, 1, G, fsp, G * fsp
            )
            # (1, 1, 1, G, fs, G * fs)
            x = x @ id_block
            # breakpoint()
            # (bs, bsp, T, G, fsp, G * fsp)

        x = x.view(bs, bsp, T, G * fsp, G * fsp)
        # (bs, bsp, T, G * fsp, G * fsp)

        x = x @ self.jacobian_cum
        # (bs, bsp, T, G * fsp, K)
        T_jx = x.shape[2]

        x = x.view(bs, bsp, T_jx, G, fsp, self.K)
        # (bs, bsp, T, G, fsp, K)

        return x

    def forward_left_temporal(self, x):
        bs, bsp, T_jx, G, t, t = x.shape

        x = x.view(bs, bsp, T_jx, G * t, G * t)
        # (bs, bsp, T_jx, G * fsp, G * fsp)

        x = x @ self.jacobian_cum
        # (bs, bsp, T_jx, G * t, K)
        T_jx = x.shape[2]

        x = x.view(bs, bsp, T_jx, G, t, self.K)
        # (bs, bsp, T, G, fsp, K)

        return x


class FeaturesPipeline(nn.Module):
    def __init__(self, params_pipeline, bsp, C):
        super().__init__()
        self.bsp = bsp
        self.C = C
        transforms = []
        for params in params_pipeline:
            params = dict(params)
            transform_type = params.pop("type")
            if transform_type == "linear":
                patch_wise = params.pop("patch_wise")
                transform = LinearFeatures(
                    # **params, bsp=self.bsp if patch_wise else 1, C=C
                    **params,
                    bsp=self.bsp if patch_wise else 1,
                )
            elif transform_type == "leaky_relu":
                patch_wise = params.pop("patch_wise")
                transform = LeakyReLU(
                    **params, bsp=self.bsp if patch_wise else 1
                )
            elif transform_type == "swish":
                patch_wise = params.pop("patch_wise")
                transform = Swish(**params, bsp=self.bsp if patch_wise else 1)
            elif transform_type == "identity":
                transform = Identity(**params)
            else:
                raise NotImplementedError
            transforms.append(transform)

        self.transforms = nn.ModuleList(transforms)
        print(f"[FeaturesPipeline] {self.transforms=}")

    def forward(self, x, idx, with_jacobian=False):
        bs, bsp, T, CK = x.shape
        if idx is not None:
            assert idx.ndim == 1
            assert len(idx) == bsp

        features = x
        # (bs, bsp, T, K)

        if with_jacobian:
            # jacobian_acc = JacobianAccumulator()
            # jacobian_acc = EfficientJacobianAccumulator(K=K, device=x.device)
            jacobian_acc = EfficientJacobianAccumulator()
        else:
            jacobian_acc = None

        # jac = torch.eye(K, device=x.device).view(1, 1, 1, K, K)
        jac = None

        for transform in self.transforms:
            # features, jacobian = transform(
            features, jac = transform(
                x=features,
                with_jacobian=with_jacobian,
                idx=idx,
                jac=jac,
                # jac=None,
            )
            # if features.isnan().any():
            # print("breakpoint()")
            # breakpoint()
            # else:
            # print("ok2")
            # if with_jacobian:
            # jacobian_acc.add_jacobian(jacobian)
            # del jacobian
            # torch.cuda.empty_cache()
        # jacobian_acc.jacobian_cum = jac

        if with_jacobian:
            jacobian_acc.set_jacobian(jac)

        return features, jacobian_acc


class Identity(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.size_in = size_in

    def forward(self, x, with_jacobian, **kwargs):
        bs, bsp, T, K = x.shape
        assert K == self.size_in

        if with_jacobian:
            jacobian = JacobianIdentity()
        else:
            jacobian = None
        return x, jacobian


class JacobianIdentity:
    def forward_right(self, x):
        return x

    def forward_left(self, x):
        return x


class LinearFeatures(nn.Module):
    def __init__(
        self,
        size_in,
        size_out,
        bsp,
        learnable=True,
        orthogonal=False,
        is_identity=False,
    ):
        super().__init__()
        self.size_in = size_in
        self.bsp = bsp
        # self.C = C
        self.learnable = learnable
        print(f"[LinearFeatures] {self.learnable=}")
        self.size_out = size_out
        # self.D = nn.Parameter(0.1 * torch.randn(self.size_out, self.size_in))
        if is_identity:
            assert (
                self.size_in == self.size_out
            ), f"{self.size_in=}, {self.size_out=}"
            self.D = torch.eye(self.size_in, device=device).view(
                1, 1, 1, self.size_in, self.size_in
            )
        else:
            D = (
                0.1
                * torch.randn(1, self.bsp, 1, self.size_out, self.size_in)
                / np.sqrt(self.size_in)
            )
            if orthogonal:
                assert self.bsp == 1
                D = D.view(self.size_out, self.size_in).T
                D = make_orthogonal(D).T
                D = D.view(1, self.bsp, 1, self.size_out, self.size_in)
            if self.learnable:
                self.D = nn.Parameter(D)
            else:
                # self.D = D
                self.register_buffer("D", D, persistent=True)


    def forward(self, x, with_jacobian, idx, jac=None):
        bs, bsp, d_s, d_pf = x.shape
        # assert CK == self.C * self.size_in, f"{CK} != {self.size_in}"
        assert d_pf == self.size_in, f"{d_pf=} != {self.size_in=}"
        if (idx is not None) and self.bsp > 1:
            # bs, bsp = mask_bsp.shape
            # assert bsp == self.bsp
            assert bs == 1, "Not compatible if bs > 1"
            D = self.D[:, idx, :, :, :]
        else:
            D = self.D
        features = (D @ x[..., None]).squeeze(-1)
        # (bs, bsp, T, size_out)

        # if features.isnan().any():
        # breakpoint()
        # else:
        # print("ok")

        assert features.shape[-1] == self.size_out
        if jac is not None:
            jac = D @ jac
            return features, jac
        else:
            return features, D[:, :, :, :, : self.size_in]


class JacobianLinearFeatures:
    def __init__(self, D):
        self.D = D
        _, self.bsp, _, self.size_out, self.size_in = self.D.shape

    def forward_cum(self, jac):
        bs, bsp, T, K, K = jac.shape
        # (bs, bsp, T, K, K)

        jac = self.D @ jac
        # (bs, bsp, T, size_out, size_in)

        return jac

    def forward_right(self, z):
        bs, bsp, T, _, K = z.shape
        # (bs, bsp, T, K (pos), size_out)

        z = (self.D.unsqueeze(3) @ z[..., None]).squeeze(-1)
        # (bs, bsp, T, K (pos), size_out)

        return z

    def forward_left(self, z):
        bs, bsp, T, G, fsp, fs = z.shape
        assert fs == self.size_out
        z = z @ self.D.view(1, self.bsp, 1, 1, self.size_out, self.size_in)
        # (bs, bsp, T, G, fs, K_in)
        assert z.shape[-1] == self.size_in

        return z


class LeakyReLU(nn.Module):
    def __init__(self, size, bsp):
        super().__init__()
        self.size = size
        self.bsp = bsp
        # self.slope = nn.Parameter(torch.tensor(0.01))
        # self.slope = nn.Parameter(0.1 * torch.ones(1, self.bsp, 1, self.size))
        self.slope = nn.Parameter(1 * torch.ones(1, self.bsp, 1, self.size))
        self.bias = nn.Parameter(0.01 * torch.randn(1, self.bsp, 1, self.size))
        # self.slope = nn.Parameter(1.0 * torch.ones(1, 1, 1, self.size))
        # self.bias = nn.Parameter(0.0 * torch.randn(1, 1, 1, self.size))

    def forward(self, x, idx, with_jacobian, jac=None):
        bs, bsp, T, K_in = x.shape

        if (idx is not None) and (self.bsp > 1):
            slope = self.slope[:, idx, :, :]
            bias = self.bias[:, idx, :, :]
        else:
            slope = self.slope
            bias = self.bias

        slope = 1e-3 + F.relu(slope)

        x_b = x + bias
        # (bs, bsp, T, K_in)

        mask_pos = (x_b > 0).float()
        # (bs, bsp, T, K_in)

        # features = x_b * mask_pos + slope * x_b * (1 - mask_pos)
        factor = mask_pos + slope * (1 - mask_pos)

        features = x_b * factor

        # (bs, bsp, T, K_in)
        if jac is not None:
            jac = factor.view(bs, bsp, T, K_in, 1) * jac
            return features, jac

        if with_jacobian:
            jacobian = JacobianLeakyReLU(mask_pos=mask_pos, slope=slope)
        else:
            jacobian = None
        return features, jacobian


class JacobianLeakyReLU:
    def __init__(self, mask_pos, slope):
        self.mask_pos = mask_pos
        self.slope = slope
        self.bs, self.bsp_data, self.T, self.size = self.mask_pos.shape
        self.bsp_slope = self.slope.shape[1]

        # (bs, bsp, T, K_in)

    def forward_cum(self, jac):
        bs, bsp, T, fs, fs = jac.shape
        # (bs, bsp, T, fs, fs)

        multiplier = self.mask_pos + (1 - self.mask_pos) * self.slope
        # (bs, bsp, T, size)

        multiplier = multiplier.view(
            self.bs, self.bsp_data, self.T, self.size, 1
        )
        # (bs, bsp, T, size, 1)

        jac = multiplier * jac
        # (bs, bsp, T, fs, fs)

        return jac

    def forward_right(self, z):
        # z: (bs, bsp, T, K (pos), K_in)
        mask_pos_fwd = self.mask_pos.unsqueeze(3)
        # (bs, bsp, T, 1, K_in)
        z = (
            mask_pos_fwd * z
            + (1 - mask_pos_fwd)
            * self.slope.view(1, self.bsp_slope, 1, 1, self.size)
            * z
        )
        return z

    def forward_left(self, z):
        # z: (bs, bsp, T, G, fsp, fs)
        bs, bsp, T_z, G, fsp, fs = z.shape
        mask_pos_adj = self.mask_pos.view(bs, bsp, self.T, 1, 1, self.size)
        z = (
            mask_pos_adj * z
            + (1 - mask_pos_adj)
            * self.slope.view(1, self.bsp_slope, 1, 1, 1, fs)
            * z
        )
        return z


class Swish(nn.Module):
    def __init__(self, size, bsp):
        super().__init__()
        self.size = size
        self.bsp = bsp
        self.slope = nn.Parameter(0.1 * torch.ones(1, self.bsp, 1, self.size))
        self.bias = nn.Parameter(0.01 * torch.randn(1, self.bsp, 1, self.size))

    def forward(self, x, idx, with_jacobian, jac=None):
        bs, bsp, T, K_in = x.shape

        if (idx is not None) and (self.bsp > 1):
            slope = self.slope[:, idx, :, :]
            bias = self.bias[:, idx, :, :]
        else:
            slope = self.slope
            bias = self.bias

        slope = 1e-3 + F.relu(slope)

        x_b = x + bias
        # (bs, bsp, T, K_in)

        sigmoid = torch.sigmoid(x_b)

        y = x_b * (slope + sigmoid)
        # (bs, bsp, T, K_in)

        dy = slope + sigmoid + sigmoid * (1 - sigmoid) * x_b
        # (bs, bsp, T, K_in)

        if jac is not None:
            jac = dy.view(bs, bsp, T, K_in, 1) * jac
            return y, jac

        if with_jacobian:
            jacobian = JacobianSwish(dy=dy)
        else:
            jacobian = None
        return y, jacobian


class JacobianSwish:
    def __init__(self, dy):
        self.dy = dy
        self.bs, self.bsp_data, self.T, self.size = self.dy.shape
        # self.bsp_slope = self.slope.shape[1]

        # (bs, bsp, T, K_in)

    def forward_cum(self, jac):
        # (bs, bsp, T, fs, fs)

        dy = self.dy[..., None]
        jac = jac * dy

        return jac

    def forward_right(self, z):
        # z: (bs, bsp, T, K (pos), K_in)
        dy = self.dy.unsqueeze(3)
        # (bs, bsp, T, 1, K_in)
        z = dy * z
        return z

    def forward_left(self, z):
        # z: (bs, bsp, T, G, fsp, fs)
        bs, bsp, T_z, G, fsp, fs = z.shape
        # mask_pos_adj = self.mask_pos.view(bs, bsp, self.T, 1, 1, self.size)
        dy = self.dy.view(bs, bsp, self.T, 1, 1, self.size)
        z = dy * z
        return z
