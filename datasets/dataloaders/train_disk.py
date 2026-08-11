import numpy as np
import torch
import random
from torch.utils.data import DataLoader, ConcatDataset
from hydra.utils import to_absolute_path
from datasets.lmdb_dataset import LMDBDataset
from datasets.transforms.coronograph_loader import CoronographLoader
from datasets.transforms.formatters import (
    FramePadder,
    KeyFilter,
    KeyRenamer,
)
from datasets.transforms.augmentations import (
    RandomTimeFlip,
    RandomSpatialFlip,
    RandomRotationFlip,
    RandomRotationReverse,
    FramesShuffler,
    RandomRot90,
    ForceRotater,
)
from datasets.transforms.masker import (
    KnownSourcesMasker,
    KnownSourcesCoords,
)
from datasets.transforms.psf_loader import PSFLoader
from datasets.transforms.disk_simulator import DiskParamsSamplerSobol, DiskInjector

import torch.multiprocessing as mp

generators = {
    "train": torch.Generator(),
    "val": torch.Generator(),
    "test": torch.Generator(),
}


def infinite_generator(dataloader):
    while True:
        yield from dataloader


def identity(x):
    return x


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ReproducibleDataLoader(DataLoader):
    def __init__(self, dataset, split, **kwargs):
        assert split in ["train", "test", "val"]
        super().__init__(
            dataset,
            **kwargs,
            worker_init_fn=seed_worker,
            generator=generators[split],
        )


def get_dataloaders_train_disk(
    batch_size,
    obj_size,
    repeat_train,
    num_workers,
    crop,
    crop_bg,
    n_min,
    n_max,
    alpha_min,
    alpha_max,
    scale,
    sep_max,
    sampling_mode,
    symmetric_flux_train,
    interpolation,
    # coronograph_params_train,
    dataset_params_train,
    dataset_params_val,
    distance_min,
    psf_model,
    psf_size_inj,
    use_adaptive,
    range_max_adaptive,
    rv_mode,
    exp_coeff,
    force_rot,
    augmentations,
    infinite=True,
    subset=None,
):
    # Prepare dataset parameters
    path_db = to_absolute_path(dataset_params_train.path_db)
    print(f"{path_db=}")
    dataset_params_train.path_db = path_db
    dataset_params_val.path_db = path_db
    dataset_params_train.suffix_lmdb = f"lmdb_{obj_size}"
    dataset_params_val.suffix_lmdb = f"lmdb_{obj_size}"

    # Load coordinates of real sources to remove them from data
    real_coords = KnownSourcesCoords(path_db=path_db)
    sources_masker = KnownSourcesMasker(path_db=path_db, mask_only=True)

    # Load PSF and coronograph transmission mask
    psf_loader = PSFLoader(path_db=path_db, crop=crop, crop_bg=crop_bg)
    path_coronograph = to_absolute_path("/scratch2/clear/chalucas/codes/DiscRec/data/coronograph")
    #path_coronograph = to_absolute_path(coronograph_params_train.path_coronograph)
    coronograph_loader = CoronographLoader(path_coronograph=path_coronograph)

    # Load augmentation transform of nuisance data
    for aug in augmentations:
        assert aug in [
            "rot90",
            "time_flip",
            "spatial_flip",
            "rot_flip",
            "random_rot_reverse",
            "rot_reverse",
            "shuffle_frames",
        ]

    if "rot90" in augmentations:
        rot90_train = RandomRot90(deterministic=False, full_frame=True)
    else:
        rot90_train = identity

    if "time_flip" in augmentations:
        time_flip_train = RandomTimeFlip(deterministic=False)
    else:
        time_flip_train = identity

    if "spatial_flip" in augmentations:
        spatial_flip_train = RandomSpatialFlip(
            deterministic=False, full_frame=True
        )
    else:
        spatial_flip_train = identity

    if "rot_flip" in augmentations:
        rot_flip_train = RandomRotationFlip(deterministic=False)
    else:
        rot_flip_train = identity

    if "random_rot_reverse" in augmentations:
        random_rot_reverse_train = RandomRotationReverse(deterministic=False)
    else:
        random_rot_reverse_train = identity

    if force_rot is not None:
        rot_rate = force_rot / 64
        force_rotater = ForceRotater(rot_rate=rot_rate)
    else:
        force_rotater = identity

    if "shuffle_frames" in augmentations:
        print("Train shuffler initialized")
        shuffler_train = FramesShuffler(deterministic=False)
    else:
        print("No train shuffler")
        shuffler_train = identity

    # Load synthetic disk injector
    disk_params_sampler = DiskParamsSamplerSobol()
    injector = DiskInjector(obj_size)

    # Key renamer and filter to keep only necessary data
    key_renamer_s = KeyRenamer(mapping=([("frame", "s_0")]))
    key_filter = KeyFilter(
        keys=[
            "s_0",
            "y",
            "rot",
            "psf",
            "lbda",
            "alpha",
            "amplitude",
            #  "coords_real",
            "radius_real",
            #  "n_sources_real",
            "idx",
            "obs_id",
            "mask",
            "weight_obs",
            "mask_temporal",
            "obj",
            "coronograph_mask"
        ]
    )

    # Frame padder to ensure consistent number of frames
    n_frames_train = dataset_params_train.n_frames
    n_frames_val = dataset_params_val.n_frames
    frame_padder_train = FramePadder(n_frames=n_frames_train)
    frame_padder_val = FramePadder(n_frames=n_frames_val)

    # Concatenate all transforms to apply consecutively
    transforms_train = [
        force_rotater,
        psf_loader,
        coronograph_loader,
        real_coords,
        sources_masker,
        key_renamer_s,
        rot90_train,
        time_flip_train,
        spatial_flip_train,
        rot_flip_train,
        random_rot_reverse_train,
        shuffler_train,
        disk_params_sampler,
        injector,
        key_filter,
        frame_padder_train,
    ]

    transforms_val = [
        force_rotater,
        psf_loader,
        coronograph_loader,
        real_coords,
        sources_masker,
        key_renamer_s,
        disk_params_sampler,
        injector,
        key_filter,
        frame_padder_val,
    ]

    # Initialize datasets and apply transforms
    print("\nINITIALIZING TRAIN DATASET")
    dataset_train = LMDBDataset(
        **dataset_params_train,
        transforms=transforms_train,
    )
    if repeat_train is not None:
        print(f"Repeating training set: {repeat_train}")
        dataset_train = ConcatDataset([dataset_train] * repeat_train)

    print("\nINITIALIZING VAL DATASET")
    dataset_val = LMDBDataset(
        **dataset_params_val,
        transforms=transforms_val,
    )

    # Initialize dataloaders using transformed datasets
    print(f"{batch_size=}")
    dataloader_train = DataLoader(
        dataset_train,
        shuffle=True,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        # multiprocessing_context=mp.get_context("spawn"),
    )

    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        # multiprocessing_context=mp.get_context("spawn"),
    )
    if infinite:
        dataloader_train = infinite_generator(dataloader_train)

    return {
        "train": dataloader_train,
        "val": dataloader_val,
        "test": None,
    }
