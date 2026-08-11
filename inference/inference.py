import torch
from astropy.io import fits
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
from hydra.utils import to_absolute_path

from sampler import get_sampler
from models import get_model
from inference.routine2 import process_folder, get_sortframes_vector
from inference.contrast_curve import contrast_curve_sigma


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SourcesMasker:
    def __init__(self, use_disks, center, H):
        self.use_disks = use_disks
        self.center = center
        self.H = H

        self.yy, self.xx = np.mgrid[: self.H, : self.H]

    def cart2pol(self, x, y):
        x_c = x - self.center
        y_c = y - self.center
        rho = np.sqrt(x_c**2 + y_c**2).astype(np.float32)
        theta = np.arctan2(y_c, x_c).astype(np.float32)
        return rho, theta

    def pol2cart(self, rho, theta):
        dx = rho * np.cos(theta)
        dy = rho * np.sin(theta)
        y = self.center + dy
        x = self.center + dx
        return x.astype(np.float32), y.astype(np.float32)

    def rotate_coords(self, x, y, angle_deg):
        rho, theta = self.cart2pol(x=x, y=y)
        theta = theta + np.radians(angle_deg)
        x, y = self.pol2cart(rho=rho, theta=theta)
        return x, y

    def __call__(self, x, y, radius, rot):
        assert len(x) == len(y)
        assert len(x) == len(radius)
        T = len(rot)
        n = len(x)

        mask = np.ones((T, self.H, self.H), dtype=bool)

        # zero init as coords are already north aligned
        # rot = rot - rot[0]

        coords_x = []
        coords_y = []
        for i in range(n):
            _x = x[i]
            _y = y[i]
            _radius = radius[i]
            for t in range(T):
                x_t, y_t = self.rotate_coords(x=_x, y=_y, angle_deg=rot[t])
                if t == 0:
                    # coordinates in non north aligned
                    coords_x.append(x_t)
                    coords_y.append(y_t)

                if self.use_disks:
                    rr = np.sqrt((self.xx - x_t) ** 2 + (self.yy - y_t) ** 2)
                    mask[t, rr < _radius] = 0
                else:
                    x_start = np.clip(x_t - _radius, 0, self.H).astype(int)
                    y_start = np.clip(y_t - _radius, 0, self.H).astype(int)
                    x_end = np.clip(x_t + _radius, 0, self.H).astype(int)
                    y_end = np.clip(y_t + _radius, 0, self.H).astype(int)
                    if np.abs(y_end - y_start) > (2 * _radius + 2):
                        breakpoint()
                    if np.abs(x_end - x_start) > (2 * _radius + 2):
                        breakpoint()
                    mask[t, y_start : y_end + 1, x_start : x_end + 1] = 0

        coords = np.stack((coords_y, coords_x)).T
        return mask, coords


def load_folder(
    path_folder, use_centered, channel_idx, channel_sortframes=0, crop=256
):
    print(f"{path_folder=}")
    assert path_folder is not None
    all_files = os.listdir(path_folder)
    if len(all_files) < 3:
        print(f"Not enough files: {all_files}")
        return

    if use_centered:
        print("Using recentered data")
        path_centered = os.path.join(path_folder, "centering2")
        # use_centered = True
        if not os.path.exists(path_centered):
            try:
                print("Centering data ..")
                process_folder(path_folder)
                path_y = os.path.join(path_centered, "shifted_cube.fits")
                path_sortframes = os.path.join(
                    path_centered, "new_sortframes_vec.fits"
                )
                print("Centering data done")
            except Exception as e:
                print("Centering failed")
                print(e)
                use_centered = False
        else:
            n_files = len(os.listdir(path_centered))
            if n_files <= 5:
                print("Number of files too low")
                use_centered = False
    else:
        print("Not using recentered data")

    is_mc = ["reduced_master_cube" in f.lower() for f in all_files]
    assert np.sum(is_mc) == 1, f"{is_mc=}, {all_files=}"
    is_psf = ["psf_master_cube" in f.lower() for f in all_files]
    # assert np.sum(is_psf) == 1, f"{is_psf=}, {all_files=}"
    if np.sum(is_psf) == 0:
        print(f"PSF not found")
        psf = None
    elif np.sum(is_psf) > 1:
        print(f"Multiple PSF: {is_psf} {all_files}")
        psf = None
    else:
        path_psf = os.path.join(path_folder, all_files[np.argmax(is_psf)])
        psf = fits.getdata(path_psf)
        if psf.ndim == 3:
            psf = psf[:, None, :, :]
        assert psf.ndim == 4
        # psf = psf[channel, 0]
        psf = psf[:, 0]
    is_rot = [
        (("rotation" in f.lower()) and ("reverse" not in f.lower()))
        for f in all_files
    ]
    assert np.sum(is_rot) == 1, f"{is_rot=}, {all_files=}"

    if use_centered:
        print("Using centered data")
        path_y = os.path.join(path_centered, "shifted_cube.fits")
        path_sortframes = os.path.join(
            path_centered, "new_sortframes_vec.fits"
        )
        sortframes = fits.getdata(path_sortframes)
    else:
        print("Not using centered data")
        path_y = os.path.join(path_folder, all_files[np.argmax(is_mc)])
        sortframes = get_sortframes_vector(path_folder)
    path_rot = os.path.join(path_folder, all_files[np.argmax(is_rot)])

    y = fits.getdata(path_y)
    rot = fits.getdata(path_rot)
    header = fits.getheader(path_y)
    infrared_filter = header["HIERARCH ESO INS COMB IFLT"]
    if infrared_filter == "DB_H23":
        lbdas = np.array([1.593, 1.667])
    elif infrared_filter == "DB_K12":
        # from CONVERT_PARAMETER_LOG
        lbdas = np.array([2.110, 2.251])
    elif infrared_filter == "BB_H":
        lbdas = np.array([1.625, 1.625])
    elif infrared_filter == "BB_J":
        lbdas = np.array([1.245, 1.245])
    elif infrared_filter == "DB_J23":
        lbdas = np.array([1.19, 1.273])
    else:
        raise NotImplementedError(f"{infrared_filter=}")
    try:
        waffle_amplitude = header["HIERARCH ESO OCS WAFFLE AMPL"]
        print(f"{waffle_amplitude=}")
    except:
        print(f"No waffle amplitude found")
    print(y.shape)

    if sortframes is not None:
        print(f"{sortframes.shape=}")
        print(f"{sortframes}")
        if sortframes.ndim == 1:
            pass
        elif sortframes.ndim == 2:
            print(f"Selecting channel {channel_sortframes} of sorting vector")
            d1, d2 = sortframes.shape
            if d1 > d2:
                sortframes = sortframes[:, channel_sortframes]
            else:
                sortframes = sortframes[channel_sortframes, :]
            print(sortframes)
        else:
            raise NotImplementedError
        if np.sum(sortframes) == 0:
            print("Sortframes is zero, replacing by ones")
            sortframes = np.ones_like(sortframes).astype(bool)
            print(sortframes)
    else:
        C, T, H, W = y.shape
        sortframes = np.ones(T, dtype=bool)
        print("sorting vector not found")
        print(f"{y.shape=}")

    assert y.ndim == 4
    if rot.ndim == 2:
        assert rot.shape[0] == 1
        rot = rot.flatten()

    assert rot.ndim == 1, f"{rot.shape=}"
    # y = y[channel]
    if channel_idx is not None:
        print(f"Selecting channel: {channel_idx=}")
        y = y[channel_idx, None, ...]
        psf = psf[channel_idx, None, ...]
        lbdas = lbdas[channel_idx, None, ...]
    else:
        print("Using all channels")

    C, T, H, W = y.shape
    assert H == 1024
    assert W == 1024
    center = H // 2
    hs = crop // 2

    y = y[:, :, center - hs : center + hs, center - hs : center + hs].copy()

    mask = np.ones((T, crop, crop), dtype=bool)
    coords_real = None
    radius_real = None
    n_sources_real = 0

    try:
        path_sources = os.path.join(
            path_folder, "sources_256_northaligned.csv"
        )
        print(f"{path_sources=}")
        df = pd.read_csv(path_sources)
        coords_x = df.loc[:, "x"].to_numpy()
        coords_y = df.loc[:, "y"].to_numpy()
        radius_real = df.loc[:, "radius"].to_numpy()
        n_sources_real = len(coords_x)
        if n_sources_real > 0:
            assert crop == 256, "coords are in 256 frame"
            source_masker = SourcesMasker(
                use_disks=True, center=crop // 2, H=crop
            )
            mask, coords_real = source_masker(
                x=coords_x, y=coords_y, radius=radius_real, rot=rot
            )
    except FileNotFoundError:
        print("No source file found")
        pass

    assert mask.shape == (T, crop, crop)

    return {
        "y": y,
        "rot": rot,
        "psf": psf,
        "header": header,
        "sortframes": sortframes,
        "lbdas": lbdas,
        "mask": mask,
        "coords_real": coords_real,
        "radius_real": radius_real,
        "n_sources_real": n_sources_real,
    }


def make_inference(inputs, sampler):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    y = torch.tensor(inputs["y"].astype(np.float32), device=device)[None, ...]
    rot = torch.tensor(inputs["rot"].astype(np.float32), device=device)[
        None, ...
    ]
    lbdas = torch.tensor(inputs["lbdas"].astype(np.float32), device=device)
    psf = inputs["psf"]
    if psf is not None:
        psf = torch.tensor(psf.astype(np.float32), device=device)[None, ...]
    sortframes = torch.tensor(inputs["sortframes"].astype(bool), device=device)
    bs, C, T, H, W = y.shape
    print(f"{T=}")
    mask_temporal = torch.ones((bs, T), device=device, dtype=bool)

    # breakpoint()
    try:
        y = y[:, :, sortframes, :, :]
    except IndexError:
        breakpoint()
        pass
    rot = rot[:, sortframes]
    mask_temporal = mask_temporal[:, sortframes]
    with torch.no_grad():
        out = sampler.run_inference(
            y=y,
            psf=psf,
            # psf=None,
            rot=rot,
            mask_temporal=mask_temporal,
            lbdas=lbdas,
        )
    return out


def save_outputs(out, folder_name, header=None):
    cwd = os.getcwd()
    print(f"{cwd=}")
    # breakpoint()
    path_folder = os.path.join(cwd, folder_name)
    print(f"{path_folder=}")
    os.makedirs(path_folder)

    for k, v in out.items():
        path = os.path.join(path_folder, f"{k}.fits")
        fits.writeto(filename=path, data=v, header=header)
        # hdu = fits.PrimaryHDU(v)
        # hdul = fits.HDUList([hdu])
        # hdul.writeto(path)
        print(f"saved to {path}")


def inference_folder(path_folder, cfg_data, sampler):
    folder_name = os.path.split(path_folder.rstrip("/"))[1]
    inputs = load_folder(
        path_folder=path_folder,
        use_centered=cfg_data.use_centered,
        channel_sortframes=0,
        channel_idx=cfg_data.channel_idx,
    )

    if inputs is None:
        return
    try:
        if cfg_data.use_amp:
            with torch.cuda.amp.autocast():
                out = make_inference(inputs=inputs, sampler=sampler)
        else:
            out = make_inference(inputs=inputs, sampler=sampler)
    except torch.cuda.OutOfMemoryError as e:
        print(e)
        print("CUDA out of memory")

    sep_as, contrast = contrast_curve_sigma(sigma_map=out["sigma_alpha"][0])
    out["contrast_curve_sigma"] = np.stack([sep_as, contrast])
    save_outputs(out=out, folder_name=folder_name, header=inputs["header"])

    return out


def filter_subdirs(subdirs, filter_name):
    out = []
    for subdir in subdirs:
        if filter_name in subdir:
            splits = subdir.split("/")
            splits = [filter_name in s for s in splits]
            if splits[-1]:
                out.append(subdir)
    print("Filtered:")
    for sub in out:
        print(sub)
    return out


def inference(cfg):
    domain = "paired"

    cfg_model = cfg.model
    cfg_data = cfg.data.inference

    model = get_model(
        image_channels=25,
        out_channels=25,
        **cfg_model,
    )

    sampler = get_sampler(
        diffusion=None,
        model=model,
        mode=cfg.mode,
        symmetry=None,
        mixing_operator=None,
        **cfg.sampler,
    )

    path_root = to_absolute_path(cfg_data.path_root)

    if cfg_data.recursive:
        path_root = path_root.rstrip("/") + "/"
        subdirs = [p[0] for p in os.walk(path_root, followlinks=True)]
        subdirs = [p for p in subdirs if p != path_root]
        subdirs = filter_subdirs(subdirs, filter_name=cfg_data.filter_name)
        print(f"{subdirs=}")
        for i, subdir in enumerate(subdirs):
            inference_folder(
                path_folder=subdir, cfg_data=cfg_data, sampler=sampler
            )
    else:
        inference_folder(
            path_folder=path_root, cfg_data=cfg_data, sampler=sampler
        )

    return

    filter_name = cfg_data.filter_name

    all_outputs = defaultdict(list)

    # NOTE: path_root should end with /
    if path_root is not None:
        print(f"{filter_name=}")
        print(f"{path_root=}")
        subdirs = [p[0] for p in os.walk(path_root, followlinks=True)]
        subdirs = [p for p in subdirs if p != path_root]
        subdirs = filter_subdirs(subdirs, filter_name=filter_name)
        print(f"{subdirs=}")
        subdirs_leaf = [p.split(path_root)[1] for p in subdirs]
        subdirs = subdirs[::-1]
        for i, subdir in enumerate(subdirs):
            print(f"\n({i + 1}/ {len(subdirs)}) {subdir=}")
            inputs = load_folder(
                path_folder=subdir,
                use_centered=cfg_data.use_centered,
                channel_sortframes=0,
                channel_idx=cfg_data.channel_idx,
            )
            if inputs is None:
                continue
            try:
                if cfg_data.use_amp:
                    with torch.cuda.amp.autocast():
                        out = make_inference(inputs=inputs, sampler=sampler)
                else:
                    out = make_inference(inputs=inputs, sampler=sampler)
            except torch.cuda.OutOfMemoryError as e:
                print(e)
                print("CUDA out of memory")
                continue
            leaf = subdirs_leaf[i]
            save_outputs(out=out, folder=leaf, header=inputs["header"])
            for k, v in out.items():
                all_outputs[k].append(v)
            all_outputs["subdir"].append(subdir)
        cwd = os.getcwd()
        path_outputs = os.path.join(cwd, "outputs.pt")
        torch.save(all_outputs, path_outputs)
        print(f"Outputs saved to {path_outputs}")
