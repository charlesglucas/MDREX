import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle

from lmdb_dataset import LMDBDataset
from transforms import (
    KnownSourcesMasker,
    FOVMasker,
    RelativeFluxNormalizer,
    StandardNormalizer,
)
from collections import defaultdict


def radial_averaging(mean, annular_width):
    pass


def main():
    path_db = "data/db"
    dual_filter = "B_H"
    channel_idx = 0
    split = "train"

    filter_obs_params = {
        "n_frames_min": 32,
        "rot_min": 5,
        "frame_type": "aggressive",
        "split": split,
        "dual_filter": dual_filter,
    }

    path_folder = os.path.join(
        path_db, "standard", f"{split}_{dual_filter}_{channel_idx}"
    )
    os.makedirs(path_folder, exist_ok=True)
    path_mean = os.path.join(path_folder, f"mean.npy")
    path_var = os.path.join(path_folder, f"var.npy")
    path_count_mean = os.path.join(path_folder, f"count_mean.npy")
    path_count_var = os.path.join(path_folder, f"count_var.npy")
    path_pixels = os.path.join(path_folder, f"pixels.pkl")
    # masker = Masker(channel=channel_idx, mask_known_sources=False)
    # normalizer = Normalizer()
    # transformations = []

    known_masker = KnownSourcesMasker(channel=channel_idx)
    fov_masker = FOVMasker(channel=channel_idx)
    rel_normalizer = RelativeFluxNormalizer()
    # std_normalizer = StandardNormalizer(
    # path_db=path_db, dual_filter=dual_filter
    # )
    transformations = [
        known_masker,
        rel_normalizer,
        # std_normalizer,
        fov_masker,
    ]
    dataset = LMDBDataset(
        path_db=path_db,
        n_frames=1,
        extent_frames=1,
        clip_per_obs_max=64,
        filter_obs_params=filter_obs_params,
        channel_idx=channel_idx,
        transforms=transformations,
    )
    print(f"length: {len(dataset)}")
    pixels = {
        # "right": {"x": 723, "y": 457},
        # "center": {"x": 614, "y": 512},
        # "bottom": {"x": 680, "y": 979},
        # "bottom-left": {"x": 436, "y": 553},
        "bottom-left+": {"x": 464, "y": 567},
        # "u": {"x": 511, "y": 376},
        # "uu": {"x": 482, "y": 21},
        # "rb": {"x": 853, "y": 710},
        "lb": {"x": 251, "y": 761},
        # "rb+": {"x": 862, "y": 774},
        # "rbb": {"x": 710, "y": 926},
        "center": {"x": 507, "y": 519},
        "rb": {"x": 563, "y": 569},
        "rb2": {"x": 560, "y": 567},
    }
    all_idx = []

    pixels_var = defaultdict(list)
    pixels_mean = defaultdict(list)

    # computing mean
    print("Computing mean ...")
    mean = None
    count_mean = None
    for i in tqdm(range(len(dataset))):
        item = dataset.__getitem__(i)
        frame = item["frame"][0]
        mask = item["mask"][0]
        if count_mean is None:
            mean = np.zeros_like(frame)
            mean[mask] = frame[mask]
            count_mean = np.zeros_like(frame)
            count_mean[mask] += 1
        else:
            mean[mask] = mean[mask] * count_mean[mask] / (count_mean[mask] + 1)
            mean[mask] += frame[mask] / (count_mean[mask] + 1)
            count_mean[mask] += 1
        for p, coords in pixels.items():
            pixels_mean[p].append(mean[coords["y"], coords["x"]])
        all_idx.append(item["idx0"])

        # if i > 10:
        # break
    # plt.imshow(np.log(1 + mean))
    # plt.show()
    # breakpoint()

    # computing variance
    print("Computing variance ...")
    var = None
    count_var = None
    for i in tqdm(range(len(dataset))):
        item = dataset.__getitem__(i)
        frame = item["frame"][0]
        mask = item["mask"][0]
        if count_var is None:
            var = np.zeros_like(frame)
            var[mask] = (frame[mask] - mean[mask]) ** 2
            count_var = np.zeros_like(frame)
            count_var[mask] += 1
        else:
            var[mask] = var[mask] * count_var[mask] / (count_var[mask] + 1)
            var[mask] += (frame[mask] - mean[mask]) ** 2 / (
                count_var[mask] + 1
            )
            count_var[mask] += 1
        for p, coords in pixels.items():
            pixels_var[p].append(var[coords["y"], coords["x"]])
    np.save(path_mean, mean)
    np.save(path_var, var)
    np.save(path_count_mean, count_mean)
    np.save(path_count_var, count_var)

    for p in pixels.keys():
        pixels[p]["mean"] = pixels_mean[p]
        pixels[p]["var"] = pixels_mean[p]
    pixels["all_idx"] = all_idx
    with open(path_pixels, "wb") as f:
        pickle.dump(pixels, f)


if __name__ == "__main__":
    main()
