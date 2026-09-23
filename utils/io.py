import os
import numpy as np
import subprocess
from astropy.io import fits


class MissingFile(Exception):
    def __init__(self, path_folder, keyword, comment=""):
        self.path_folder = path_folder
        self.keyword = keyword
        self.comment = comment
        msg = f"{self.path_folder} ({self.keyword}), {comment}"
        super().__init__(msg)


def load_fits(path_folder, keyword, check_zero=True):
    from astropy.io import fits

    assert keyword.lower() == keyword
    files = [f for f in os.listdir(path_folder) if keyword in f.lower()]

    assert len(files) <= 1
    if len(files) == 0:
        raise MissingFile(
            path_folder=path_folder, keyword=keyword, comment="no file"
        )
    assert len(files) == 1, f"{path_folder} ({keyword}): {files}"
    full_path = os.path.join(path_folder, files[0])

    hdul = fits.open(full_path)
    assert len(hdul) == 1
    if check_zero:
        data = hdul[0].data
        if np.all(data == 0):
            raise MissingFile(
                path_folder=path_folder, keyword=keyword, comment="psf is zero"
            )
    return hdul[0]


def save_fits(data, path, header=None, header_dict=None):
    assert data.ndim == 3, f"{data.shape}"
    T, H, W = data.shape
    data = np.flip(data, axis=1)
    if os.path.exists(path):
        print(f"Removing {path}")
        os.remove(path)
    if header is None:
        hdu = fits.PrimaryHDU(data=data)
    else:
        hdu = fits.PrimaryHDU(data=data, header=header)
    if header_dict:
        for k, v in header_dict.items():
            hdu.header[k] = v
    print(f"saving fits: {os.path.abspath(path)}")
    hdu.writeto(path)


def sync_file(path, force=False):
    split = path.split(":")
    if len(split) == 1:
        path = split[-1]
        fetch_jz = False
    elif len(split) == 2:
        machine, path = split
        assert machine == "jeanzay"
        fetch_jz = True
    else:
        raise ValueError()
    root_vasher = "/home/tbodrito/exo/exodiff"
    root_jz = "/gpfswork/rech/ofy/uzm62av/exo/exodiff"

    assert not os.path.isabs(path), "path should be relative"
    assert "data_new" not in path

    if os.path.exists(path) and (not force):
        print(f"file {path} found locally")
        return
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    path_vasher = os.path.join(root_vasher, path)
    # if fetch_jz or (not exists_remote(host="vasher", path=path_vasher)):
    if fetch_jz:
        print("path do not exists, fetching on JZ")
        path_dir_vasher = os.path.join(root_vasher, path_dir)
        path_jz = os.path.join(root_jz, path)
        cmd = f"ssh vasher mkdir -p {path_dir_vasher}"
        print(f"Creating remote directory on vasher: {path_dir_vasher}")
        subprocess.check_output(cmd.split(" "))
        cmd_sync = f"ssh vasher rsync -rv -P jeanzay:{path_jz} {path_vasher}"
        print(f"syncing: {cmd_sync}")
        subprocess.check_output(cmd_sync.split(" "))

    cmd = f"rsync -rv -P vasher:{path_vasher} {path}"
    print(cmd)
    subprocess.check_output(cmd.split(" "))
    print(f"file {path}: downloaded locally")


def sync_file_direct(path, force=False, exclude=None):
    split = path.split(":")
    if len(split) == 1:
        path = split[-1]
        fetch_jz = False
    elif len(split) == 2:
        machine, path = split
        assert machine == "jeanzay"
        fetch_jz = True
    else:
        raise ValueError()

    if os.path.exists(path) and (not force):
        print(f"file {path} found locally")
        return
    path_dir = os.path.dirname(path)

    if fetch_jz:
        os.makedirs(path_dir, exist_ok=True)
        print(f"Fetching on JZ..")
        root_jz = "/gpfswork/rech/ofy/uzm62av/exo/exodiff"
        path_jz = os.path.join(root_jz, path)
        if exclude is None:
            exclude = ""
        else:
            exclude = f"--exclude=\"{exclude}\""
        print(f"{path=}")
        cmd = f"rsync -rv -P {exclude} jeanzay:{path_jz} {path}"
        print(f"syncing:")
        print(f"{cmd}")
        subprocess.check_output(cmd.split(" "))
        print(f"file {path}: downloaded locally")
    else:
        sync_file(path, force=force)


def sync_file_vasher(path, force=False):
    split = path.split(":")
    if len(split) == 1:
        path = split[-1]
        fetch_jz = False
    elif len(split) == 2:
        machine, path = split
        assert machine == "jeanzay"
        fetch_jz = True
    else:
        raise ValueError()
    assert not os.path.isabs(path), "path should be relative"
    root_vasher = "/home/tbodrito/exo/exodiff"
    root_jz = "/gpfswork/rech/ofy/uzm62av/exo/exodiff"

    assert not os.path.isabs(path), "path should be relative"
    assert "data_new" not in path
    path_vasher = os.path.join(root_vasher, path)

    if os.path.exists(path_vasher) and (not force):
        print(f"file {path} found locally")
        return path_vasher

    path_dir = os.path.dirname(path_vasher)
    os.makedirs(path_dir, exist_ok=True)
    # path_vasher = os.path.join(root_vasher, path)
    # if fetch_jz or (not exists_remote(host="vasher", path=path_vasher)):
    if fetch_jz:
        print("path do not exists, fetching on JZ")
        # path_dir_vasher = os.path.join(root_vasher, path_dir)
        path_jz = os.path.join(root_jz, path)
        # cmd = f"ssh vasher mkdir -p {path_dir_vasher}"
        # print(f"Creating remote directory on vasher: {path_dir_vasher}")
        # subprocess.check_output(cmd.split(" "))
        cmd_sync = f"rsync -rv jeanzay:{path_jz} {path_vasher}"
        print(f"syncing: {cmd_sync}")
        subprocess.check_output(cmd_sync.split(" "))

        print(f"file {path}: downloaded locally")
    return path_vasher
