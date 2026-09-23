import torch
import numpy as np
import os

from utils.viz import save_gif


class SaveProgressCallback:
    def __init__(self, path_progress, save_interval):
        self.path_progress = path_progress
        self.save_interval = save_interval
        os.makedirs(self.path_progress)

    def __call__(self, x_t, x_0t, t):
        if t % self.save_interval != 0:
            return
        path_x_t = os.path.join(self.path_progress, f"x_t_{t:04d}.npy")
        path_x_0t = os.path.join(self.path_progress, f"x_0t_{t:04d}.npy")
        np.save(path_x_t, x_t[0, 0].detach().cpu().numpy())
        np.save(path_x_0t, x_0t[0, 0].detach().cpu().numpy())
        print(f"Saved to {path_x_t}")


class SaveGIFCallback:
    def __init__(self, path_gif, save_interval, n_samples, cmap="viridis"):
        self.path_gif = path_gif
        self.save_interval = save_interval
        self.n_samples = n_samples
        self.cmap = cmap
        os.makedirs(self.path_gif)

    def save(self, data, fn):
        n_tot = min(len(data), self.n_samples)
        for i in range(n_tot):
            path_folder = os.path.join(self.path_gif, f"{i}")
            os.makedirs(path_folder, exist_ok=True)
            path_gif = os.path.join(path_folder, fn)
            save_gif(
                data=data[i].cpu().numpy(),
                path=path_gif,
                cmap=self.cmap,
                caption="",
            )

    def __call__(self, x_t, x_0t, t):
        if t % self.save_interval != 0:
            return

        # for i in range(self.n_samples):
        self.save(data=x_t, fn=f"x_t_{t:04d}.gif")
        self.save(data=x_0t, fn=f"x_0t_{t:04d}.gif")
        # path_gif = os.path.join(self.path_gif, f"{i}")
        # os.makedirs(path_gif, exist_ok=True)

        # path_x_t = os.path.join(path_gif, f"x_t_{t:04d}.gif")
        # path_x_0t = os.path.join(path_gif, f"x_0t_{t:04d}.gif")

        # save_gif(
        # data=x_t[i].cpu().numpy(),
        # path=path_x_t,
        # cmap=self.cmap,
        # caption="",
        # )
        # save_gif(
        # data=x_0t[i].cpu().numpy(),
        # path=path_x_0t,
        # cmap=self.cmap,
        # caption="",
        # )
