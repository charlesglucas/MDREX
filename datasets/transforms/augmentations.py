import torch
import numpy as np


def pol2cart(rho, theta):
    dx = rho * np.cos(theta)
    dy = rho * np.sin(theta)
    return dx.astype(np.float32), dy.astype(np.float32)


def cart2pol(dx, dy):
    rho = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    return rho.astype(np.float32), theta.astype(np.float32)


def rot90pol(rho, theta, k):
    for _ in range(int(k)):
        theta = theta - np.pi / 2
        dx, dy = pol2cart(rho=rho, theta=theta)
        dy = dy + 1
        rho, theta = cart2pol(dx=dx, dy=dy)
    return rho, theta


class FramesShuffler:
    def __init__(self, deterministic):
        self.deterministic = deterministic
        print(f"AUGMENTATION: {self.__class__.__name__} instanciated")

    def __call__(self, item):
        s_0 = item["s_0"]
        mask = item["mask"]
        # breakpoint()
        C, T, H, W = s_0.shape
        assert T >= C
        assert mask.ndim == 4

        if self.deterministic:
            idx = item["idx"]
            gen = np.random.RandomState(seed=idx)
            perm = gen.permutation(T)
        else:
            perm = torch.randperm(T).numpy()

        item["s_0"] = s_0[:, perm]
        item["mask"] = mask[:, perm]

        return item


class RandomTimeFlip:
    def __init__(self, deterministic):
        self.deterministic = deterministic
        print(f"AUGMENTATION: {self.__class__.__name__} instanciated")

    def __call__(self, item):
        s_0 = item["s_0"]
        mask = item["mask"]
        rot = item["rot"]
        C, T, H, W = s_0.shape
        assert T >= C
        assert mask.ndim == 4
        assert rot.ndim == 1
        assert len(rot) == T

        if self.deterministic:
            idx = item["idx"]
            gen = np.random.RandomState(seed=idx)
            flip = gen.rand(1) > 0.5
        else:
            # perm = torch.randperm(T).numpy()
            flip = torch.rand(1).item() > 0.5

        if flip:
            item["s_0"] = s_0[:, ::-1].copy()
            item["mask"] = mask[:, ::-1].copy()
            item["rot"] = rot[::-1].copy()

        return item


class RandomRotationFlip:
    def __init__(self, deterministic):
        self.deterministic = deterministic
        print(f"AUGMENTATION: {self.__class__.__name__} instanciated")

    def __call__(self, item):
        s_0 = item["s_0"]
        mask = item["mask"]
        rot = item["rot"]
        C, T, H, W = s_0.shape
        assert T >= C
        assert mask.ndim == 4
        assert rot.ndim == 1
        assert len(rot) == T

        if self.deterministic:
            idx = item["idx"]
            gen = np.random.RandomState(seed=idx)
            flip = gen.rand(1) > 0.5
        else:
            # perm = torch.randperm(T).numpy()
            flip = torch.rand(1).item() > 0.5

        if flip:
            item["rot"] = rot[::-1].copy()

        return item


class RandomSpatialFlip:
    def __init__(self, deterministic, full_frame):
        self.deterministic = deterministic
        self.full_frame = full_frame
        print(f"AUGMENTATION: {self.__class__.__name__} instanciated")

    def __call__(self, item):
        s_0 = item["s_0"]
        mask = item["mask"]
        rot = item["rot"]

        if self.full_frame:
            rho = None
            theta = None
        else:
            rho = item["rho"]
            theta = item["theta"]
        C, T, H, W = s_0.shape
        assert T >= C
        assert mask.ndim == 4
        assert rot.ndim == 1
        assert len(rot) == T

        if self.deterministic:
            idx = item["idx"]
            gen = np.random.RandomState(seed=idx)
            vertical_flip = gen.rand(1) > 0.5
            horizontal_flip = gen.rand(1) > 0.5
        else:
            # perm = torch.randperm(T).numpy()
            vertical_flip = torch.rand(1).item() > 0.5
            horizontal_flip = torch.rand(1).item() > 0.5

        if "paco_a" in item.keys():
            paco_a = item["paco_a"]
            assert paco_a.ndim == 4

        if vertical_flip:
            s_0 = s_0[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
            rot = -rot
            if not self.full_frame:
                dx, dy = pol2cart(rho=rho, theta=theta)
                dy = -dy + 1
                rho, theta = cart2pol(dx=dx, dy=dy)
                item["theta"] = theta
            if "paco_a" in item.keys():
                paco_a = torch.flip(paco_a, dims=(2,))
        if horizontal_flip:
            s_0 = s_0[:, :, :, ::-1].copy()
            mask = mask[:, :, :, ::-1].copy()
            rot = -rot
            if not self.full_frame:
                dx, dy = pol2cart(rho=rho, theta=theta)
                dx = -dx + 1
                rho, theta = cart2pol(dx=dx, dy=dy)
                item["theta"] = theta
            if "paco_a" in item.keys():
                paco_a = torch.flip(paco_a, dims=(3,))

        item["s_0"] = s_0
        item["mask"] = mask
        item["rot"] = rot
        if "paco_a" in item.keys():
            item["paco_a"] = paco_a

        return item


class RandomRot90:
    def __init__(self, deterministic, full_frame):
        self.deterministic = deterministic
        self.full_frame = full_frame
        print(f"AUGMENTATION: {self.__class__.__name__} instanciated")

    def __call__(self, item):
        s_0 = item["s_0"]
        mask = item["mask"]
        rot = item["rot"]
        C, T, H, W = s_0.shape
        assert T >= C
        assert mask.ndim == 4
        assert rot.ndim == 1
        assert len(rot) == T

        if self.deterministic:
            idx = item["idx"]
            gen = np.random.RandomState(seed=idx)
            k = int(gen.rand(1) // 0.25)
        else:
            # perm = torch.randperm(T).numpy()
            k = int(torch.rand(1).item() // 0.25)

        assert k in [0, 1, 2, 3]

        s_0 = np.rot90(s_0, k=k, axes=(2, 3))
        mask = np.rot90(mask, k=k, axes=(2, 3))

        if "paco_a" in item.keys():
            paco_a = item["paco_a"]
            assert paco_a.ndim == 4
            paco_a = torch.rot90(paco_a, k=k, dims=(2, 3))
            item["paco_a"] = paco_a

        if not self.full_frame:
            rho = item["rho"]
            theta = item["theta"]
            rho, theta = rot90pol(rho=rho, theta=theta, k=k)
            item["rot"] = rot
            item["theta"] = theta

        item["s_0"] = s_0.copy()
        item["mask"] = mask.copy()

        return item


class RandomRotationReverse:
    def __init__(self, deterministic):
        self.deterministic = deterministic
        print(f"AUGMENTATION: {self.__class__.__name__} instanciated")

    def __call__(self, item):
        if self.deterministic:
            idx = item["idx"]
            gen = np.random.RandomState(seed=idx)
            reverse = gen.uniform() > 0.5

        else:
            reverse = torch.rand(1).item() > 0.5

        if reverse:
            item["rot"] *= -1

        return item


class FlipRot2:
    def __call__(self, item):
        item["rot"] *= -1
        return item


class ForceRot:
    def __init__(self, rot, reverse=True):
        # self.rot = -torch.linspace(0, rot, T)
        self.rot_range = rot
        self.reverse = reverse
        print(f"[ForceRot] initialized with rot_range={self.rot_range}")
        print(f"[ForceRot] {self.reverse=}")

    def __call__(self, item):
        T = len(item["rot"])
        sign = np.sign(item["rot"][-1] - item["rot"][0])
        if self.reverse:
            sign *= -1
        rot = sign * np.linspace(0, self.rot_range, T)
        item["rot"] = rot.astype(np.float32)
        return item


class ForceNFrames:
    def __init__(self, T):
        self.T = T

    def resize_frames(self, frames):
        T, H, W = frames.shape
        n_repeat = 1 + int(np.ceil(self.T / T))
        idx = np.concatenate([np.arange(T)] * n_repeat)
        idx = idx[: self.T]
        frames = frames[idx]
        assert frames.shape == (self.T, H, W)
        return frames

    def resize_rot(self, rot):
        assert rot.ndim == 1
        T = len(rot)
        n_repeat = 1 + int(np.ceil(self.T / (T - 1)))

        rot_diff = list(rot[1:] - rot[:-1])
        rot_new = [0] + rot_diff * n_repeat
        assert len(rot_new) >= self.T
        rot_new = np.array(rot_new[: self.T])
        rot_new = np.cumsum(rot_new) + rot[0]
        assert len(rot_new) == self.T
        T_check = np.minimum(self.T, T)
        assert np.abs(rot_new[:T_check] - rot[:T_check]).max() < 1e-3
        rot_new = rot_new.astype(np.float32)
        return rot_new

    def __call__(self, item):
        try:
            item["s_0"] = self.resize_frames(item["s_0"])
        except Exception:
            print(item.keys())
            item["frame"] = self.resize_frames(item["frame"])
        item["mask"] = self.resize_frames(item["mask"])
        item["rot"] = self.resize_rot(item["rot"])
        return item


class ForceRotater:
    def __init__(self, rot_rate):
        self.rot_rate = float(rot_rate)
        print(f"AUGMENTATION: {self.__class__.__name__} instanciated")
        self.rot = (self.rot_rate * np.arange(64)).astype(np.float32)
        print(f"AUGMENTATION: {self.__class__.__name__} {self.rot=}")

    def __call__(self, item):

        item["rot"] = self.rot

        return item
