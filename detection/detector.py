from abc import ABC, abstractmethod
from collections import defaultdict
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils.prefix_log import TimeLogMixin


class BaseDetector(ABC):
    @abstractmethod
    def forward(self, y, rot, psf):
        """
        Returns a detection score (SNR) per pixel

        Args:
        -----
        * y: torch.tensor(b, T, H, W)
        * rot: torch.tensor(b, T)
        * psf: torch.tensor(b, h, w)

        Returns:
        --------
        * snr: torch.tensor(b, H', W')
        """
        pass


class SamplingDetector(BaseDetector, TimeLogMixin):
    def __init__(self, sampler, n_samples):
        self.sampler = sampler
        self.n_samples = n_samples

    def forward(
        self,
        y,
        rot,
        psf,
        use_tqdm=True,
        return_samples=False,
        prefix=None,
        **kwargs,
    ):
        """
        Args:
        -----
        * y: torch.tensor(b, T, H, W)
        * rot: torch.tensor(b, T)
        * psf: torch.tensor(b, c, c)

        Returns:
        --------
        * alpha: torch.tensor(b, H', W')
        * sigma_alpha: torch.tensor(b, H', W')
        """
        assert y.ndim == 4
        assert rot.ndim == 2
        assert psf.ndim == 3
        all_alpha = []
        all_samples = defaultdict(list)
        for n in tqdm(range(self.n_samples), disable=not use_tqdm):
            if (n == 0) or ((n + 1) % 5 == 0):
                self.log(f"Sample {n + 1}/{self.n_samples}")
            sample = self.sampler.run(
                y=y,
                rot=rot,
                psf=psf,
                idx=n,
                use_tqdm=False,
                prefix=prefix,
            )
            alpha = sample["alpha"]
            all_alpha.append(alpha)

            for k, v in sample.items():
                all_samples[k].append(v)
        self.log(f"Sampling finished")

        all_alpha = torch.stack(all_alpha)
        assert all_alpha.ndim == 4

        if self.n_samples == 1:
            print("/!\ N SAMPLES = 1")
            mean_alpha = torch.mean(all_alpha, dim=0)
            std_alpha = torch.ones_like(mean_alpha)
        else:
            std_alpha = torch.std(all_alpha, dim=0)
            mean_alpha = torch.mean(all_alpha, dim=0)
        if False:
            print(mean_alpha.shape)
            snr = mean_alpha / std_alpha
            plt.imshow(snr[0].cpu() > 5)
            plt.show()
        if return_samples:
            return mean_alpha, std_alpha, all_alpha
        return mean_alpha, std_alpha


class DeterministicDetector(BaseDetector):
    def __init__(self, sampler):
        self.sampler = sampler

    # def forward(self, y, rot, psf, amplitude, **kwargs):
    def forward(self, y, rot, psf, amplitude=None, **kwargs):
        """
        Args:
        -----
        * y: torch.tensor(b, T, H, W)
        * rot: torch.tensor(b, T)
        * psf: torch.tensor(b, c, c)
        * amplitude: torch.tensor(b,)

        Returns:
        --------
        * alpha: torch.tensor(b, H', W')
        * sigma_alpha: torch.tensor(b, H', W')
        """
        assert y.ndim == 5
        assert rot.ndim == 2
        assert psf.ndim == 4
        sample = self.sampler.run(
            y=y, rot=rot, psf=psf, amplitude=amplitude, **kwargs
        )
        alpha = sample["alpha"]
        sigma_alpha = sample["sigma_alpha"]
        if False:
            snr = alpha / sigma_alpha
            plt.imshow(snr[0].cpu().numpy())
            plt.show()
            breakpoint()

        return alpha, sigma_alpha
