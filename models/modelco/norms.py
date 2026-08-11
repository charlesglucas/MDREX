import torch
import torch.nn as nn
from utils.misc import ContextPlaceHolder


class TemporalNorm(nn.Module):
    def __init__(self, n_channels, n_groups, affine):
        super().__init__()
        self.n_channels = n_channels
        self.n_groups = n_groups
        self.affine = affine
        print(f"[TemporalNorm] {self.affine=}")
        self.eps = 1e-5
        assert self.n_groups > 0, f"{n_groups=}"
        assert self.n_channels % self.n_groups == 0
        if self.affine:
            self.scale = nn.Parameter(torch.ones(1, self.n_channels, 1, 1, 1))
            self.shift = nn.Parameter(torch.zeros(1, self.n_channels, 1, 1, 1))

    def forward(self, x, mask_temporal):
        bs, C, T, H, W = x.shape
        assert mask_temporal.shape[1] == T
        assert mask_temporal.shape[0] == 1

        mask_temporal = mask_temporal.view(1, 1, 1, T, 1, 1).float()
        # (bs, 1, 1, T, 1, 1)

        T_real = torch.sum(mask_temporal, dim=3, keepdim=True)
        # (bs, 1, 1, 1, 1, 1)

        x = x.view(bs, self.n_groups, -1, T, H, W)
        # (bs, G, dC, T, H, W)
        dC = x.shape[2]

        mean = torch.sum(x * mask_temporal, dim=(2, 3), keepdim=True) / (
            T_real * dC
        )
        # (bs, G, 1, 1, H, W)

        x_c = (x - mean) * mask_temporal
        # (bs, G, dC, T, H, W)

        var = (
            torch.sum(x_c**2, dim=(2, 3), keepdim=True) / (dC * T_real - 1)
            + 1e-5
        )
        # (bs, G, 1, 1, H, W)

        std = torch.sqrt(var)
        # (bs, G, 1, 1, H, W)

        x_c = x_c / std
        # (bs, G, dC, T, H, W)

        x_c = x_c.view(bs, C, T, H, W)
        # (bs, C, T, H, W)

        if self.affine:
            x_c = x_c * self.scale + self.shift

        return x_c


class WhitenNorm(nn.Module):
    def __init__(
        self,
        n_groups,
        n_channels,
        affine,
        reg_C,
        joint_mu,
        joint_C,
        mode="cholesky",
        grad_params=False,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_channels = n_channels
        assert self.n_channels % self.n_groups == 0
        assert mode in ["cholesky", "zca", "iternorm"]
        self.mode = mode
        self.grad_params = grad_params
        self.reg_C = float(reg_C)

        self.affine = affine
        print(f"[WhitenNorm] {self.affine=}")
        self.joint_mu = joint_mu
        self.joint_C = joint_C
        print(f"[WhitenNorm] {self.joint_mu=}")
        print(f"[WhitenNorm] {self.joint_C=}")
        if self.affine:
            self.scale = nn.Parameter(torch.ones(1, self.n_channels, 1, 1, 1))
            self.shift = nn.Parameter(torch.zeros(1, self.n_channels, 1, 1, 1))

    def forward(self, x, mask_temporal):
        bs, L, T, H, W = x.shape

        _, T = mask_temporal.shape
        assert mask_temporal.shape[0] == 1

        if self.grad_params:
            context = ContextPlaceHolder()
        else:
            context = torch.no_grad()

        mask_temporal = mask_temporal.view(1, 1, T, 1, 1).float()
        # (bs, 1, T, 1, 1)

        T_real = torch.sum(mask_temporal, dim=2, keepdim=True)
        # (bs, 1, 1, 1, 1)

        x = x * mask_temporal
        # (bs, L, T, H, W)

        with context:
            if self.joint_mu:
                mean = torch.sum(x, dim=(0, 2), keepdim=True) / T_real
                # (1, L, 1, H, W)
            else:
                mean = torch.sum(x, dim=2, keepdim=True) / T_real
                # (bs, L, 1, H, W)

        x = (x - mean) * mask_temporal
        # (bs, L, T, H, W)

        x = x.view(bs, self.n_groups, -1, T, H, W)
        # (bs, G, dG, T, H, W)

        dG = x.shape[2]

        # x = x.permute(0, 1, 4, 5, 3, 2)
        # (bs, G, H, W, T, dG)
        x = x.permute(1, 4, 5, 0, 3, 2)
        # (G, H, W, bs, T, dG)

        G, H, W, bs, _, _ = x.shape

        x = x.reshape(-1, T, dG)
        # (bsp, T, dG)

        # mask_temporal = mask_temporal
        T_real = T_real.view(1, 1, 1, 1, 1, 1)
        T_real = T_real.expand(-1, G, H, W, -1, -1)
        # (bs, G, H, W, 1, 1)

        T_real = T_real.view(G * H * W, 1, 1)
        # (bsp, 1, 1)

        with context:
            if self.joint_C:
                T_real = bs * T_real
                x = x.view(-1, bs * T, dG)
                # (G H W, bs * T, dG)
            S_hat = torch.einsum("bti,btj->bij", x, x) / T_real
            # (bsp, dG, dG)
            # breakpoint()

            bsp = S_hat.shape[0]

            tr_S2 = torch.einsum("bii->b", S_hat**2)
            tr2_S = torch.einsum("bii->b", S_hat) ** 2
            tr_SS = torch.einsum("bii->b", S_hat @ S_hat)

            diag = torch.einsum("bii->bi", S_hat)
            # (bsp, dG)

            F = torch.diag_embed(diag)
            # (bsp, dG, dG)

            T_real = T_real.view(bsp)
            # (bsp)

            diag_id = torch.eye(dG, device=S_hat.device).view(1, dG, dG)

            num = tr_SS + tr2_S - 2 * tr_S2
            den = (T + 1) * (tr_SS - tr_S2)
            rho = torch.clip(num / den, 0, 1)
            rho = rho.view(-1, 1, 1)

            C_hat = (1 - rho) * S_hat + rho * F
            # (bsp, dG, dG)

            if self.reg_C > 0:
                C_hat = C_hat + self.reg_C * diag_id
                # C_hat = C_hat + self.reg_C * F

            C_inv, info_inv = torch.linalg.inv_ex(C_hat)
            Lw, info_chol = torch.linalg.cholesky_ex(C_inv, upper=False)
            # (bsp, dG, dG)

            mask_issue = (info_inv + info_chol) > 0
            mask_issue_float = mask_issue.float().view(-1, 1, 1)
            # (bsp)
            n_issues = torch.sum(mask_issue).item()
            if n_issues > 0:
                print(f"/!\ n_issues C_hat: {n_issues=}")
                print(f"    n_issues_inv: {info_inv.sum()=}")
                print(f"    n_issues_cholesky: {info_chol.sum()=}")
                # breakpoint()
            with torch.no_grad():
                # L[mask_issue, :, :] = 0
                Lw = torch.nan_to_num(Lw)
            Lw = Lw * (1 - mask_issue_float) + diag_id * mask_issue_float
            # L = L * (1 - mask_issue_float) + mask_issue_float / torch.sqrt(F)

            Lw = Lw.transpose(1, 2).unsqueeze(1)
            # (bsp, 1, dG, dG)

            if Lw.isnan().any():
                print(f"Lw is nan: {Lw.isnan().sum()=}")
                breakpoint()
        assert x.shape[0] == Lw.shape[0]

        x = x[..., None]
        # (G H W, bs T, dG, 1)
        # or (G H W bs, T, dG, 1)
        # float32

        x_w = Lw @ x
        # (bsp, T, dG, 1)
        # float16

        x_w = x_w.reshape(self.n_groups, H, W, bs, T, dG)
        # (G, H, W, bs, T, dG)

        x_w = x_w.permute(3, 0, 5, 4, 1, 2)
        # (bs, G, dG, T, H, W)

        x_w = x_w.reshape(bs, L, T, H, W)
        # (bs, L, T, H, W)

        if self.affine:
            x_w = self.scale * x_w + self.shift

        # if x_w.isnan().any():
            # print(f"x_w is nan: {x_w.isnan().sum()=}")
            # breakpoint()

        return x_w
