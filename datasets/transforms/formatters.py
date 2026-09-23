import numpy as np

class KeyRenamer:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, item):
        for k_old, k_new in self.mapping:
            if k_old in list(item.keys()):
                item[k_new] = item.pop(k_old)
        return item


class KeyFilter:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, item):
        for k in list(item.keys()):
            if k not in self.keys:
                item.pop(k)
        return item


class LambdaFormatter:
    def __init__(self, keys, fn):
        self.keys = keys
        self.fn = fn

    def __call__(self, item):
        # breakpoint()
        for k in self.keys:
            item[k] = self.fn(item[k])
        return item


class FramePadder:
    def __init__(self, n_frames):
        self.n_frames = n_frames
        print(f"[FramePadder] {self.n_frames=}")

    def __call__(self, item):
        y = item["y"]
        s_0 = item["s_0"]
        mask = item["mask"]
        rot = item["rot"]
        assert y.shape == s_0.shape
        C, T, H, W = y.shape

        assert T <= self.n_frames
        assert T == len(rot)
        n_frames_missing = self.n_frames - T
        if n_frames_missing > 0:
            zeros_3d = np.zeros((C, n_frames_missing, H, W), dtype=np.float32)
            y = np.concatenate([y, zeros_3d], axis=1)
            s_0 = np.concatenate([s_0, zeros_3d], axis=1)
            mask = np.concatenate(
                [mask, zeros_3d[0, None, ...].astype(bool)], axis=1
            )

            pad_rot = [rot[-1]] * n_frames_missing
            rot = np.array(list(rot) + pad_rot).astype(np.float32)

        mask_temporal = np.zeros(self.n_frames, dtype=bool)
        mask_temporal[:T] = 1

        item["y"] = y
        item["s_0"] = s_0
        item["rot"] = rot
        item["mask_temporal"] = mask_temporal
        item["mask"] = mask

        return item
