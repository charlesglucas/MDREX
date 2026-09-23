import pandas as pd
import os
import numpy as np


def polar_to_cart(pa, sep, rot, center=512):
    if isinstance(sep, np.ndarray):
        sep = sep.reshape(1, -1)
    sep = sep / 1000
    if isinstance(pa, np.ndarray):
        pa = pa.reshape(1, -1)
    if isinstance(rot, np.ndarray):
        rot = rot.reshape(-1, 1)
    angle = np.deg2rad(-rot - pa + 180)
    y_coords = -np.cos(angle) * sep / 0.01225
    x_coords = -np.sin(angle) * sep / 0.01225
    x_coords += center
    y_coords += center
    return x_coords, y_coords


class KnownSourcesMasker:
    """ "Mask flagged sources"""

    def __init__(self, path_db, center=None, mask_only=False):
        self.path_db = path_db
        self.mask_only = mask_only
        self.center = center
        self.path_sources = os.path.join(self.path_db, "metadata/sources.csv")

        self.reload()

    def reload(self):
        self.df_sources = pd.read_csv(self.path_sources, header=0)

    def __call__(self, item):
        assert "mask" not in item.keys()
        # data = item["frame"]
        debug = False
        # debug = True
        frame = item["frame"]
        C, T, H, W = frame.shape
        if self.center is None:
            assert H == W
            center = H // 2
        else:
            center = self.center
            assert self.center == H // 2
        mask = np.ones((T, H, W), dtype=bool)

        obs_id = item["obs_id"]
        sub_df = self.df_sources[self.df_sources.obs_id == obs_id]

        rot = item["rot"]
        sep = sub_df.sep.to_numpy()
        pa = sub_df.pa.to_numpy()
        radius = sub_df.radius.to_numpy()
        status = list(sub_df.status)
        if not self.mask_only:
            item["sep"] = sep
            item["pa"] = pa
            item["radius"] = radius
            item["status"] = status

        n_sources = len(sub_df)
        if n_sources > 0:
            x_coords, y_coords = polar_to_cart(
                pa=pa, sep=sep, rot=rot, center=center
            )

            for t in range(T):
                for i in range(n_sources):
                    win = radius[i]
                    x = x_coords[t, i]
                    y = y_coords[t, i]

                    x_start = np.clip(x - win, 0, W).astype(int)
                    y_start = np.clip(y - win, 0, H).astype(int)
                    x_end = np.clip(x + win, 0, W).astype(int)
                    y_end = np.clip(y + win, 0, H).astype(int)
                    if np.abs(y_end - y_start) > (2 * win + 2):
                        breakpoint()
                    if np.abs(x_end - x_start) > (2 * win + 2):
                        breakpoint()

                    mask[t, y_start : y_end + 1, x_start : x_end + 1] = 0

            if debug:
                from utils import cube_3d_viewer

                frames = np.stack(frame)
                cube_3d_viewer(
                    frames,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    # status=status,
                    masks=mask,
                    alpha=0.1,
                )

        item["mask"] = mask[None, ...]
        # (1 ,T, H, W)
        return item


class KnownSourcesCoords:
    """ "Mask flagged sources"""

    def __init__(self, path_db, center=None, n_max=150):
        self.path_db = path_db
        self.n_max = n_max
        self.center = center
        self.path_sources = os.path.join(self.path_db, "metadata/sources.csv")

        self.reload()

    def reload(self):
        self.df_sources = pd.read_csv(self.path_sources, header=0)

    def __call__(self, item):
        assert "mask" not in item.keys()
        # data = item["frame"]
        # debug = True
        frame = item["frame"]
        C, T, H, W = frame.shape
        if self.center is None:
            assert H == W
            center = H // 2
        else:
            center = self.center
            assert self.center == H // 2
        # mask = np.ones(frame.shape, dtype=bool)

        obs_id = item["obs_id"]
        sub_df = self.df_sources[self.df_sources.obs_id == obs_id]

        rot = item["rot"]
        sep = sub_df.sep.to_numpy()
        pa = sub_df.pa.to_numpy()
        radius = sub_df.radius.to_numpy()
        # status = list(sub_df.status)
        # if not self.mask_only:
        # item["sep"] = sep
        # item["pa"] = pa
        # item["radius"] = radius
        # item["status"] = status

        n_sources = len(sub_df)
        if n_sources > self.n_max:
            breakpoint()
        assert n_sources <= self.n_max
        coords_real = np.ones((self.n_max, 2))
        radius_real = np.ones((self.n_max,))
        if n_sources > 0:
            x_coords, y_coords = polar_to_cart(
                pa=pa, sep=sep, rot=rot, center=center
            )
            # (T, n_sources)
            # x_coords = x_coords[0]
            # y_coords = y_coords[0]
            # breakpoint()
            concat = np.stack(
                [y_coords[0], x_coords[0]],
            ).T
            coords_real[:n_sources] = concat
            radius_real[:n_sources] = radius

        item["coords_real"] = coords_real
        # (n_max, 2) (y, x)
        item["radius_real"] = radius_real
        item["n_sources_real"] = n_sources
        return item
