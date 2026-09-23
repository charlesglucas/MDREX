import math
import numpy as np
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

# from labml_helpers.module import Module


class Swish(nn.Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class ResidualBlock(nn.Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        skip_norm,
        n_groups: int = 32,
        dropout: float = 0.1,
    ):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        if skip_norm:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Group normalization and the second convolution layer
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1)
            )
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(
        self,
        n_channels: int,
        skip_norm,
        n_heads: int = 1,
        d_k: int = None,
        n_groups: int = 32,
    ):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        if skip_norm:
            self.norm = nn.Identity()
        else:
            self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(
            batch_size, -1, self.n_heads, 3 * self.d_k
        )
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        n_groups,
        skip_norm,
    ):
        super().__init__()
        self.res = ResidualBlock(
            in_channels,
            out_channels,
            time_channels,
            n_groups=n_groups,
            skip_norm=skip_norm,
        )
        if has_attn:
            self.attn = AttentionBlock(
                n_channels=out_channels, n_groups=n_groups, skip_norm=skip_norm
            )
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool,
        n_groups,
        skip_norm,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(
            in_channels + out_channels,
            out_channels,
            time_channels,
            n_groups=n_groups,
            skip_norm=skip_norm,
        )
        if has_attn:
            self.attn = AttentionBlock(
                n_channels=out_channels, skip_norm=skip_norm, n_groups=n_groups
            )
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(
        self, skip_norm, n_channels: int, time_channels: int, n_groups
    ):
        super().__init__()
        self.res1 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            n_groups=n_groups,
            skip_norm=skip_norm,
        )
        self.attn = AttentionBlock(
            n_channels, n_groups=n_groups, skip_norm=skip_norm
        )
        self.res2 = ResidualBlock(
            n_channels,
            n_channels,
            time_channels,
            n_groups=n_groups,
            skip_norm=skip_norm,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            n_channels, n_channels, (4, 4), (2, 2), (1, 1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(
        self,
        skip_norm=False,
        image_channels: int = 3,
        image_channels_out=None,
        n_channels: int = 64,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[int]] = (
            False,
            False,
            True,
            True,
        ),
        n_blocks: int = 2,
        n_groups=32,
        init_zero_last=False,
    ):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)
        self.cumprod_res = np.cumprod(ch_mults)[-1]
        self.skip_norm = skip_norm
        print(f"[U-Net] {self.skip_norm=}")

        # Project image into feature map
        self.image_proj = nn.Conv2d(
            image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1)
        )

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)
        if image_channels_out is None:
            image_channels_out = image_channels

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    DownBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        time_channels=n_channels * 4,
                        has_attn=is_attn[i],
                        n_groups=n_groups,
                        skip_norm=self.skip_norm,
                    )
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(
            n_channels=out_channels,
            time_channels=n_channels * 4,
            n_groups=n_groups,
            skip_norm=self.skip_norm,
        )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    UpBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        time_channels=n_channels * 4,
                        has_attn=is_attn[i],
                        n_groups=n_groups,
                        skip_norm=self.skip_norm,
                    )
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_channels=n_channels * 4,
                    has_attn=is_attn[i],
                    n_groups=n_groups,
                    skip_norm=self.skip_norm,
                )
            )
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        if self.skip_norm:
            self.norm = nn.Identity()
        else:
            self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(
            in_channels, image_channels_out, kernel_size=(3, 3), padding=(1, 1)
        )
        self.init_zero_last = init_zero_last
        print(f"[UNet] {self.init_zero_last}")
        if self.init_zero_last:
            self.final.bias.data.zero_()
            self.final.weight.data.zero_()

    def forward(self, x: torch.Tensor, t=None, **kwargs):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        # to_unsqueeze = False
        # if x.ndim == 5:
        #     assert x.shape[2] == 1
        # x = x.squeeze(2)
        # to_unsqueeze = True

        # (bs, T, C, H, W)
        # (bs, T, C, H, W)
        assert x.ndim == 5, f"{x.shape=}"
        x = x.squeeze(2)
        # (bs, T, H, W)
        assert x.ndim == 4

        bs, _, H, W = x.shape
        res_h = H % self.cumprod_res
        res_w = W % self.cumprod_res
        assert res_h == 0, f"{res_h=}, {self.cumprod_res=}, {H=}"
        assert res_w == 0, f"{res_w=}, {self.cumprod_res=}, {W=}"

        # (bs, C, H, W)
        # breakpoint()
        # Get time-step embeddings
        if t is None:
            device = x.device
            t = torch.zeros(bs, device=device)
        else:
            t = t.view(bs)

        t = self.time_emb(t)

        # breakpoint()
        # Get image projection
        x = self.image_proj(x)
        # (bs, d, H, W)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # First half of U-Net
        for m in self.down:
            # print(m)
            x = m(x, t)
            # (bs, d, H, W)
            h.append(x)

        # Middle (bottom)
        # breakpoint()
        x = self.middle(x, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                # breakpoint()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)

        # Final normalization and convolution
        out = self.final(self.act(self.norm(x)))
        # if to_unsqueeze:
        # out = out.unsqueeze(2)
        out = out.unsqueeze(2)
        # (bs, T, 1, H, W)
        return out


def get_unet(
    timesteps,
    n_channels,
    skip_norm,
    ch_mults,
    n_blocks=2,
    n_groups=32,
    image_channels_out=None,
    init_zero_last=False,
    **kwargs,
):
    unet = UNet(
        skip_norm=skip_norm,
        image_channels=timesteps,
        ch_mults=ch_mults,
        n_channels=n_channels,
        image_channels_out=image_channels_out,
        n_blocks=n_blocks,
        n_groups=n_groups,
        init_zero_last=init_zero_last,
    )
    return unet
