from torch.utils.data import DataLoader, Subset, ConcatDataset
from hydra.utils import to_absolute_path
from datasets.lmdb_dataset import LMDBDataset
from datasets.transforms.formatters import (
    LambdaFormatter,
    FramePadder,
)
from datasets.transforms.augmentations import (
    FlipRot2,
    FramesShuffler,
    ForceRot,
    ForceNFrames,
)
from datasets.transforms.masker import (
    KnownSourcesMasker,
    KnownSourcesCoords,
)
from datasets.transforms.formatters import KeyFilter, KeyRenamer
from datasets.transforms.coords_sampler import (
    CoordsSamplerTransform,
)
from datasets.transforms.injection import Injector3D
from datasets.transforms.psf_loader import PSFLoader


def identity(x):
    return x


def get_dataloaders_detection(
    batch_size,
    obj_size,
    repeat_idx,
    n_repeat,
    # p_unc,
    num_workers,
    crop,
    crop_bg,
    n_min,
    n_max,
    alpha_min,
    alpha_max,
    sampling_mode,
    range_max_adaptive,
    use_adaptive,
    scale,
    sep_max,
    interpolation,
    dataset_params,
    distance_min,
    psf_model,
    psf_size_inj,
    shuffle,
    flip_rot,
    obs_id,
    n_cubes,
    force_rot_pre,
    force_rot_post,
    reverse_rot_post,
    force_n_frames_pre,
    force_n_frames_post,
    rv_mode,
    exp_coeff,
    # rot_scale_post,
    seed=None,
    subset=None,
):
    # assert obs_id is not None
    path_db = to_absolute_path(dataset_params.path_db)
    print(f"{path_db=}")
    dataset_params.path_db = path_db
    dataset_params.suffix_lmdb = f"lmdb_{obj_size}"

    # load coords real sources
    real_coords = KnownSourcesCoords(path_db=path_db)

    sources_masker = KnownSourcesMasker(path_db=path_db, mask_only=True)

    # load PSF
    psf_loader = PSFLoader(path_db=path_db, crop=crop, crop_bg=crop_bg)

    # unsqueezer
    # unsqueezer = LambdaFormatter(keys=["s_0"], fn=lambda x: x[None, ...])
    unsqueezer = LambdaFormatter(
        keys=["s_0", "mask"], fn=lambda x: x[:, None, ...]
    )
    # squeezer = LambdaFormatter(keys=["s_0", "y"], fn=lambda x: x.squeeze(0))
    squeezer = LambdaFormatter(keys=["s_0", "y"], fn=lambda x: x.squeeze(1))

    if shuffle:
        print("Val shuffler initialized")
        shuffler = FramesShuffler(deterministic=True)
    else:
        print("No val shuffler")
        shuffler = identity

    # sampler synthetic coords
    synthetic_coords_sampler = CoordsSamplerTransform(
        name="uniform",
        deterministic=True,
        n_min=n_min,
        n_max=n_max,
        size=obj_size,
        crop=crop,
        radius=distance_min,
        sampling_mode=sampling_mode,
        adaptive=use_adaptive,
        filter_real=True,
        sep_max=sep_max,
        n_channels=1,
        range_max=range_max_adaptive,
        rv_mode=rv_mode,
        exp_coeff=exp_coeff,
        alpha_range=[alpha_min, alpha_max],
        seed=seed,
    )

    # assert (force_rot_pre is not None) == (force_rot_post is not None)
    # assert (force_rot_post is not None) == (force_n_frames is not None)

    if force_rot_post is not None:
        # assert force_n_frames is not None
        # assert force_rot_pre is not None
        # forced_rot_pre = ForceRot(rot=force_rot_pre, T=64)
        print(f"Forced_rot_pre")
        forced_rot_pre = ForceRot(rot=force_rot_pre)
        print(f"Forced_rot_post")
        forced_rot_post = ForceRot(
            rot=force_rot_post, reverse=reverse_rot_post
        )
    else:
        forced_rot_post = identity
        forced_rot_pre = identity
        # forced_n_frames = identity

    if force_n_frames_pre:
        forced_n_frames_pre = ForceNFrames(T=force_n_frames_pre)
    else:
        forced_n_frames_pre = identity

    if force_n_frames_post:
        forced_n_frames_post = ForceNFrames(T=force_n_frames_post)
    else:
        forced_n_frames_post = identity

    if flip_rot:
        rot_flipper = FlipRot2()
    else:
        rot_flipper = identity

    # normalize amplitude speckles
    # amplitude_normalizer = AmplitudeNormalizer()

    # inject synthetic sources in frame
    injector = Injector3D(
        method=interpolation,
        ensure_key_error=False,
        psf_size_inj=psf_size_inj,
        scale=scale,
        # n_channels=1,
        psf_model=psf_model,
    )

    key_renamer_s = KeyRenamer(mapping=([("frame", "s_0")]))
    key_filter = KeyFilter(
        keys=[
            "s_0",
            "y",
            "rot",
            "psf",
            "alpha",
            "lbda",
            "coords",
            "n_sources",
            "amplitude",
            "coords_real",
            "radius_real",
            "n_sources_real",
            "idx",
            "obs_id",
            "mask",
        ]
    )

    # rot_scaler_post = RotScaler(scale=rot_scale_post)

    n_frames = dataset_params.n_frames
    frame_padder = FramePadder(n_frames=n_frames)

    transforms_val = [
        rot_flipper,
        psf_loader,
        # amplitude_normalizer,
        real_coords,
        sources_masker,
        forced_n_frames_pre,
        forced_rot_pre,
        synthetic_coords_sampler,
        key_renamer_s,
        forced_n_frames_post,
        forced_rot_post,
        # unsqueezer,
        shuffler,
        # rot_scaler_post,
        injector,
        # squeezer,
        key_filter,
        frame_padder,
    ]

    # dataset_params.filter_obs_params.split = "val"

    dataset_val = LMDBDataset(
        **dataset_params,
        transforms=transforms_val,
        n_repeat=n_cubes,
    )
    print(f"{len(dataset_val)=}")
    if repeat_idx is not None:
        assert subset is None
        print(f"Repeating idx: {repeat_idx}")
        dataset_val = Subset(dataset_val, indices=[int(repeat_idx)])
        dataset_val = ConcatDataset([dataset_val] * n_repeat)

    if subset is not None:
        assert repeat_idx is None
        subset = str(subset).split(" ")
        subset = [int(k) for k in subset]
        print(f"Applying subset: {subset}")
        dataset_val = Subset(dataset_val, indices=subset)
    else:
        print("No subset applied")

    dataloader_val = DataLoader(
        dataset_val,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
    )
    return {"train": None, "val": dataloader_val, "test": None}
