import numpy as np
import pandas as pd
import logging
import os
from distutils.dir_util import copy_tree
from collections import defaultdict
import torch
import pickle
import gc
from tqdm import tqdm
from utils import load_fits, MissingFile
from database.hash import hash_folder
from datasets.lmdb_utils import LMDBWriter

dtype_torch = torch.float32
dtype_np = np.float32

path_cubes = "data/f150/all_missing"
path_sort = "data/f150/sort/SPHERE_DC_DATA"

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class ScienceData:
    def __init__(self, path_cube, path_sort=None):
        self.path_cube = path_cube
        # self.path_sort = path_sort

        # self.psf_params = []
        self.load_data()
        self.current = 0

        self.sep = []
        self.pa = []
        self.dm = []
        self.absm = []
        self.snr = []
        self.status = []

    def load_data(self):
        # print("Loading cube")
        hdul_mc = load_fits(self.path_cube, keyword="reduced_master_cube")
        # Cube is not loaded in RAM as this point

        hdul_rot = load_fits(self.path_cube, keyword="rotation")

        header_mc = hdul_mc.header
        header_rot = hdul_rot.header

        # object
        assert header_rot["object"].strip() == header_mc["object"].strip()

        # Actual loading of the cube in RAM
        print("Loading cube")
        self.cube = hdul_mc.data.astype(np.float32)
        print("Loading cube ok")
        # /!\ from_numpy is important to avoid copying the array
        # print("To pytorch")
        # self.cube = torch.from_numpy(cube)
        # print("To pytorch ok")

        self.rotations = hdul_rot.data.astype(dtype_np)

        C, T, H, W = self.cube.shape
        self.T = T
        self.C = C

        assert self.rotations.shape == (T,)
        print(f"n_frames: {T}")

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.C * self.T:
            raise StopIteration

        t = self.current // self.C
        c = self.current % self.C

        # TODO: use temporality in PSF
        out = {
            "frame": self.cube[c, t].copy(),
            "rot": self.rotations[t],
            "t": t,
            "c": c,
        }
        self.current += 1
        return t, c, out


def main():
    small = True
    # small = False
    # path_metadata = "metadata/metadata_join_known.csv"
    path_metadata = "metadata/metadata_cubes.csv"
    # path_out = "data/db/"
    path_out = "data/db_new/"
    path_interactive = "metadata/interactive"
    # obs_id_small = [
    # "HD_103599__2015-06-02__23-29-37",
    # "HD_104125__2015-06-03__23-27-24",

    # ]

    if small:
        path_out = "data/db_small_test_2"

    df = pd.read_csv(path_metadata, header=0)
    if small:
        df = df.iloc[18:22]
    path_interactive_out = os.path.join(path_out, "metadata")
    # df = df.iloc[:10]
    # df_sources = pd.read_csv(path_sources, header=0)
    print(len(df))
    missing_files = defaultdict(list)
    n_frames = np.sum(df["n_frames"])
    print(f"n_frames: {n_frames}")

    lmdb_writer = LMDBWriter(path=path_out, n_frames=n_frames)
    to_write = []
    for i, row in df.iterrows():
        obs_id = row.obs_id
        print(f"\n {i + 1}/{len(df)} {obs_id}")
        path_folder_cube = os.path.join(path_cubes, row.folder)
        try:
            science_data = ScienceData(path_cube=path_folder_cube)
        except MissingFile as e:
            print(f"/!\ MISSING FILE: {e}")
            missing_files[obs_id].append(
                (e.keyword, e.path_folder.split("/")[-1])
            )
            continue

        for i, (t, c, item) in enumerate(tqdm(science_data)):
            frame_id = f"{obs_id}__{t:04d}_{c}"
            item["frame_id"] = frame_id
            to_write.append(
                # (f"{global_index:07d}".encode("ascii"), pickle.dumps(item))
                (frame_id.encode("ascii"), pickle.dumps(item))
            )
            if len(to_write) >= 256:
                lmdb_writer.add_frames(to_write)
                to_write = []

        del science_data
        gc.collect()

    lmdb_writer.add_frames(to_write)

    # copying sources and sort_obs
    copy_tree(path_interactive, path_interactive_out)
    print(f"Copied {path_interactive!r} to {path_interactive_out!r}")

    df.to_pickle(os.path.join(path_interactive_out, "df_global.pkl"))
    print(f"Missing files: {missing_files}")
    hash_folder(path_interactive_out)


if __name__ == "__main__":
    logging.info("")
    main()
