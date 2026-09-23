import numpy as np
import torch


class ReproducibleNoiseSampler:
    def __init__(self, seed, T, deterministic):
        self.seed = seed
        self.T = T
        self.deterministic = deterministic

    def __call__(self, item):
        idx = item["idx"]
        try:
            frame = item["frame"]
        except KeyError:
            # frame = item["obj"]
            frame = item["s_0"]

        if self.deterministic:
            gen = np.random.RandomState(seed=self.seed + idx)
            noise = gen.randn(*frame.shape).astype(np.float32)
            gen = np.random.RandomState(seed=self.seed + idx)
            # t_diffusion = gen.rand()
            t = gen.randint(0, self.T)
        else:
            noise = np.random.randn(*frame.shape).astype(np.float32)
            t = np.random.randint(0, self.T)

        # item["noise"] = torch.tensor(noise, dtype=torch.float32)
        # item["noise"] = torch.tensor(noise, dtype=torch.float32) * 0
        item["noise"] = torch.tensor(noise, dtype=torch.float32)
        item["t"] = t
        # item["t"] = 0

        return item


class ReproducibleQuantileSampler:
    def __init__(self, seed, deterministic):
        self.seed = seed
        self.deterministic = deterministic

    def __call__(self, item):
        idx = item["idx"]

        if self.deterministic:
            gen = np.random.RandomState(seed=self.seed + idx)
            # noise = gen.randn(*frame.shape).astype(np.float32)
            q = gen.rand() * 100.0
        else:
            q = np.random.rand() * 100.0

        item["q"] = q

        return item


def check_reproducible():
    noise_sampler = ReproducibleNoiseSampler(seed=0)
    item = {"idx": 38, "frame": np.zeros((32, 32))}
    torch.manual_seed(22)
    item1 = noise_sampler(item)
    torch.manual_seed(34)
    item2 = noise_sampler(item)

    item = {"idx": 37, "frame": np.zeros((32, 32))}

    item3 = noise_sampler(item)

    assert np.all(item1["noise"] == item2["noise"])
    assert np.all(item1["t_diffusion"] == item2["t_diffusion"])
    assert not np.all(item1["noise"] == item3["noise"])
    print("OK")
