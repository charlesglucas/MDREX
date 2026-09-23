import torch
import torch.nn as nn
import math


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


class WeightsPredictor(nn.Module):
    def __init__(self, dim_out, dim_emb=128):
        super().__init__()
        self.dim_out = dim_out
        self.dim_emb = dim_emb
        self.encoder_x = TimeEmbedding(n_channels=self.dim_emb)
        self.encoder_y = TimeEmbedding(n_channels=self.dim_emb)
        self.encoder_rot = TimeEmbedding(n_channels=self.dim_emb)

        self.model = nn.Sequential(
            *[
                nn.Linear(self.dim_emb, self.dim_emb),
                nn.ReLU(),
                nn.Linear(self.dim_emb, self.dim_emb),
                nn.ReLU(),
                nn.Linear(self.dim_emb, self.dim_out),
                nn.Softmax(dim=1),
            ]
        )

    def forward(self, x, y, rot_range):
        bs, HW = x.shape
        bs, HW = y.shape
        x = x.flatten()
        y = y.flatten()
        emb_x = self.encoder_x(x)
        emb_y = self.encoder_y(y)
        # (bs * HW, dim)

        emb_rot = self.encoder_rot(rot_range)
        # (bs, dim)

        emb_x = emb_x.view(bs, HW, self.dim_emb)
        emb_y = emb_y.view(bs, HW, self.dim_emb)
        emb_rot = emb_rot.view(bs, 1, self.dim_emb)

        emb = emb_x + emb_y + emb_rot
        # (bs, HW, dim_emb)

        emb = emb.view(bs * HW, self.dim_emb)

        out = self.model(emb)
        # (bs * HW, dim_out)

        out = out.view(bs, HW, self.dim_out)
        # (bs, HW, dim_out)

        out = out.permute(0, 2, 1)
        # (bs, dim_out, HW)

        return out
