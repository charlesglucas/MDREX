import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, alpha, sigma, mask_target, alpha_target, **kwargs):
        bs, C, H, W = alpha.shape
        assert alpha.shape == alpha_target.shape
        assert sigma.shape == alpha_target.shape

        mask_loss = torch.ones((bs, C, H, W), device=device, dtype=bool)
        mask_loss &= mask_target.bool().view(bs, 1, H, W)

        alpha_gt = alpha_target[mask_loss]
        alpha_pred = alpha[mask_loss]
        sigma_pred = sigma[mask_loss]

        loss = 0.5 * (alpha_pred - alpha_gt) ** 2 / sigma_pred**2
        loss = loss + torch.log(sigma_pred)

        loss = loss.mean()

        return loss


class MODELCOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, snr, snr_target, mask_target):
        bs, C, H, W = snr.shape
        assert snr.shape == snr_target.shape

        loss = (snr - snr_target)[mask_target].pow(2).mean()

        return loss
