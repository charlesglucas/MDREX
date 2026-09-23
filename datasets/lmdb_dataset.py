import os
from torch.utils.data import Dataset
from omegaconf.listconfig import ListConfig
import pandas as pd
import numpy as np
from hydra.utils import get_original_cwd

from database.utils import get_hash, get_filter_id
from database.lmdb_utils import LMDBReader, KeyNotFoundError


def visualize_frame(frame, x_coords, y_coords):
    import matplotlib.pyplot as plt

    center = 1024 // 2

    plt.imshow(frame)
    plt.scatter(center + x_coords, center + y_coords)
    plt.show()


def load_split(path_db, **filter_obs_params):
    hash_meta = get_hash(path_db)
    filter_id = get_filter_id(**filter_obs_params)
    path_split = os.path.join(
        path_db, "splits", hash_meta, filter_id, "split.csv"
    )
    print(f"{path_split=}")
    df = pd.read_csv(path_split)
    return df


FILTERS = ["DB_H23", "DB_K12", "DB_Y23", "DB_J23", "BB_H", "DB_NDH23"]
LBDAS_IRDIS = [1.593, 1.667]


class LMDBDataset(Dataset):
    def __init__(
        self,
        path_db,
        n_frames,
        stride,
        dilation,
        rot_scale,
        drop_short_clips,
        filter_obs_params,
        channel_idx,
        clips_per_obs_max,
        multibands,
        n_obs_max=None,
        # suffix_lmdb="lmdb",
        suffix_lmdb="lmdb",
        n_repeat=None,
        transforms=[],
    ):
        """
        Fetch frames from LMDB database

        Args:
        -----
        * path_db: path to databse, str
        * n_frames: number of frames in a clip (review mode), int
        * stride: temporal stride for clip starts, int
        * dilation: temporal dilation within a clip, int
        * drop_short_clips: discard clips shorter than n_frames, bool
        * clips_per_obs_max: max number of clips per observation, int
        * filter_obs_params: parameters for filtering obs, dict
        * channel_idx: index of channel (0 or 1), int
        """
        super().__init__()
        # assert n_frames is not None
        self.path_db = path_db
        self.n_frames = n_frames
        self.stride = stride
        self.dilation = dilation
        self.drop_short_clips = drop_short_clips
        self.clips_per_obs_max = clips_per_obs_max
        self.rot_scale = rot_scale
        self.n_obs_max = n_obs_max
        self.multibands = multibands
        self.channel_idx = channel_idx
        self.log(f"{self.n_frames=}")
        self.log(f"{self.stride=}")
        self.log(f"{self.dilation=}")
        self.log(f"{self.drop_short_clips=}")
        self.log(f"{self.clips_per_obs_max=}")
        self.log(f"{self.rot_scale=}")
        self.log(f"{self.n_obs_max=}")
        self.log(f"{self.channel_idx=}")
        self.log(f"{self.multibands=}")

        try:
            path_cwd_init = get_original_cwd()
        except Exception:
            path_cwd_init = os.getcwd()
            # raise e
        self.path_meta_global = os.path.join(
            path_cwd_init, "database/metadata/df_global_merged.csv"
        )
        # breakpoint()
        self.path_frames_quality = os.path.join(
            path_db, "metadata/frames_quality.pkl"
        )
        self.n_repeat = n_repeat

        self.transforms = transforms
        # self.frames_per_obs_max = frames_per_obs_max

        # self.lmdb_reader = LMDBReader(path=self.path_db)
        path_lmdb = os.path.join(self.path_db, suffix_lmdb)
        self.lmdb_reader = LMDBReader(path=path_lmdb)

        self.filter_obs_params = filter_obs_params
        self.reload()
        print(f"[LMDBDataset] length dataset: {len(self)}")

    def log(self, txt):
        cls_name = self.__class__.__name__
        print(f"[{cls_name}] {txt}")

    def reload(self):
        df_global = pd.read_csv(self.path_meta_global)
        self.df_global = df_global.sort_values(by="obs_id")
        self.df_quality = pd.read_pickle(self.path_frames_quality)
        # breakpoint()

        print("[LMDBDataset] Filtering obs..")
        self.filter_obs(**self.filter_obs_params)

        print("[LMDBDataset] Init frames index..")
        try:
            n_frames_min = self.filter_obs_params.n_frames_min
        except:
            n_frames_min = self.filter_obs_params["n_frames_min"]
        self.init_frames_index(
            n_frames=self.n_frames,
            n_frames_min=n_frames_min,
            frame_quality=self.frame_quality,
            stride=self.stride,
            dilation=self.dilation,
            drop_short_clips=self.drop_short_clips,
            clips_per_obs_max=self.clips_per_obs_max,
            n_obs_max=self.n_obs_max,
        )

    def filter_obs(
        self,
        rot_min,
        n_frames_min,
        frame_quality,
        dual_filter,
        split,
        obs_category,
        obs_id=None,
        status=None,
        sel_obs=[],
    ):
        assert frame_quality in [-1, 0, 1, 2]
        # assert dual_filter in [None, "B_H", "B_Ks"]
        # assert n_frames_min >= self.extent_frames
        assert split in [None, "train", "val", "test", "calib"]
        assert self.channel_idx in [None, 0, 1]
        print(f"[LMDBDataset] {split=}")
        print(f"[LMDBDataset] {obs_id=}")
        print(f"[LMDBDataset] {rot_min=}")
        print(f"[LMDBDataset] {n_frames_min=}")
        print(f"[LMDBDataset] {frame_quality=}")
        print(f"[LMDBDataset] {dual_filter=}")
        print(f"[LMDBDataset] {status=}")
        print(f"[LMDBDataset] {obs_category=}")

        # obs_id_0 = self.df_global.obs_id
        mask_missing_psf = (1 - self.df_global.missing_psf).astype(bool)
        # breakpoint()
        self.df_global = self.df_global[mask_missing_psf]
        n_calib = (self.df_global.split == "calib").sum()
        print(f"{n_calib=}")
        # breakpoint()

        # self.split = split

        if status is not None:
            assert isinstance(status, list) or isinstance(status, ListConfig)
            status = list(status)
            for s in status:
                assert s in [-1, 0, 1, 2, 3]
            mask_na = self.df_global.status_obs.isna()
            self.df_global.loc[mask_na, "status_obs"] = -1
            # breakpoint()
            if 0 in status:
                assert (
                    not self.df_global.status_obs.isna().any()
                ), f"missing status, {status}"
                # breakpoint()
                assert not (
                    self.df_global.status_obs == -1
                ).any(), f"\n{self.df_global[self.df_global.status_obs == -1]['obs_id']}"

        self.frame_quality = frame_quality
        mask_quality = self.df_quality.quality >= self.frame_quality
        self.df_quality["valid_frames"] = mask_quality
        df_n_frames = (
            self.df_quality[["obs_id", "valid_frames"]].groupby("obs_id").sum()
        )
        # breakpoint()
        assert "valid_frames" not in self.df_global.keys()
        self.df_global = pd.merge(
            left=self.df_global, right=df_n_frames, on="obs_id", how="left"
        )
        # obs_id_1 = self.df_global.obs_id
        # # breakpoint()

        # TODO: pb with cubes without sorting vectors (2)
        mask_na_quality = self.df_global.valid_frames.isna()
        self.df_global.loc[mask_na_quality, "valid_frames"] = -1
        assert not self.df_global.valid_frames.isna().any()

        mask_n_frames = self.df_global.valid_frames >= n_frames_min
        print(f"mask_n_frames: {np.sum(mask_n_frames)}/{len(mask_n_frames)}")
        mask_rot = np.abs(self.df_global.rot_max) >= rot_min
        print(f"mask_rot: {np.sum(mask_rot)}/{len(mask_rot)}")

        mask_obs = mask_rot
        assert np.sum(mask_obs) > 0
        mask_obs &= mask_n_frames
        assert np.sum(mask_obs) > 0
        if dual_filter is not None:
            if isinstance(dual_filter, list) or isinstance(
                dual_filter, ListConfig
            ):
                mask_filter = np.zeros(len(self.df_global), dtype=bool)
                for filt in dual_filter:
                    assert filt in FILTERS, f"{filt=}, {FILTERS=}"
                    mask_filter |= (
                        self.df_global.infrared_filter.str.strip() == filt
                    )
            elif isinstance(dual_filter, str):
                mask_filter = (
                    # self.df_global.irdis_dual_filter.str.strip() == dual_filter
                    self.df_global.infrared_filter.str.strip()
                    == dual_filter
                )
            else:
                raise ValueError(f"{dual_filter=}")
            print(f"mask_filter: {np.sum(mask_filter)}/{len(mask_filter)}")
            mask_obs &= mask_filter
            # assert np.sum(mask_obs) > 0
            if np.sum(mask_obs) == 0:
                breakpoint()
            assert np.sum(mask_obs) > 0

        # obs_id_0 = self.df_global[mask_obs].obs_id
        # mask_obs &= mask_missing_psf
        # obs_id_1 = self.df_global[mask_obs].obs_id
        # breakpoint()

        if status is not None:
            mask_status = [s in status for s in self.df_global.status_obs]
            mask_status = np.array(mask_status)
            print(f"mask_status: {np.sum(mask_status)}/{len(mask_status)}")
            mask_obs &= mask_status
            assert np.sum(mask_obs) > 0

        if obs_id is not None:
            mask_obs_id = [obs == obs_id for obs in self.df_global.obs_id]
            mask_obs_id = np.array(mask_obs_id)
            print(f"mask_obs_id: {np.sum(mask_obs_id)}/{len(mask_obs_id)}")
            mask_obs &= mask_obs_id
            assert np.sum(mask_obs) > 0

        if obs_category is not None:
            cat = self.df_global["obs_category3"]
            if isinstance(obs_category, ListConfig):
                mask_cat = np.array([_cat in obs_category for _cat in cat])
            else:
                mask_cat = cat == obs_category
            print(f"mask_cat: {np.sum(mask_cat)}/{len(mask_cat)}")
            mask_obs &= mask_cat
            assert np.sum(mask_obs) > 0

        assert np.sum(mask_obs) > 0

        self.df_global = self.df_global[mask_obs]
        # breakpoint()

        # loading splits
        if split is not None:
            if self.df_global.split.isna().any():
                mask_split_na = self.df_global.split.isna()
                df_na = self.df_global[mask_split_na]
                print(df_na[["obs_id", "split"]].to_string())
                breakpoint()
            assert not self.df_global.split.isna().any()
            print(f"N observations: {len(self.df_global)}")
            print(
                f"pre-filter: {self.df_global[['obs_id', 'rot_max', 'n_frames', 'infrared_filter', 'missing_psf', 'split', 'status_obs']].to_string()}"
            )

            mask_split = self.df_global.split == split
            print(f"mask_cat: {np.sum(mask_split)}/{len(mask_split)}")
            self.df_global = self.df_global[mask_split]
            assert len(self.df_global) > 0

        self.df_global = self.df_global.reset_index()

        # selected observations
        if len(sel_obs) > 0:
            mask = np.zeros(len(self.df_global), dtype=bool)
            for i in range(len(self.df_global)):
                if self.df_global.iloc[i].obs_id in sel_obs:
                    mask[i] = 1
            self.df_global = self.df_global[mask]
        self.df_global = self.df_global.reset_index(drop=True)

        # self.df_global = self.df_global.iloc[3:]
        print(f"N observations: {len(self.df_global)}")
        print(
            f"obs_id: {self.df_global[['obs_id', 'rot_max', 'n_frames', 'infrared_filter']].to_string()}"
        )
        # assert len(self.df_global) > 0
        if len(self.df_global) == 0:
            breakpoint()
        # breakpoint()

    def get_idx_obs(self, obs_id):
        sub_df = self.df_global[self.df_global.obs_id == obs_id]
        # assert len(sub_df) == 1
        if len(sub_df) == 0:
            return -1, f"obs_id: {obs_id!r} not found"
        obs_idx = sub_df.index.astype(int)[0]

        all_obs_idx = self.all_frames[self.idx_start, 0]
        mask = all_obs_idx == obs_idx
        # assert np.sum(mask) > 0
        if np.sum(mask) == 0:
            return -1, f"obs_id: {obs_id!r} no valid frames"
        start_idx = np.arange(len(self))[mask][0]

        return start_idx, "0"

    def init_frames_index(
        self,
        n_frames,
        n_frames_min,
        stride,
        dilation,
        frame_quality,
        drop_short_clips,
        clips_per_obs_max,
        n_obs_max,
    ):
        """
        Init all frames, idx_start, and clip_lengths


        df_global:
        ----------
        obs_id | .. | rot_max |valid_frames | split | valid_frames (count)

        df_quality:
        ----------
        obs_id | t | quality |valid_frames
        """
        all_clips = {
            "obs_id": [],
            "frame_idx": [],
            "frame_quality": [],
            "n_valid": [],
        }
        for i, row in self.df_global.iterrows():
            obs_id = row.obs_id

            # get frames quality of obs_id
            sub_df = self.df_quality[self.df_quality.obs_id == obs_id]
            # quality = self.df_quality["quality"].to_numpy()

            mask_valid = sub_df["valid_frames"]

            idx_valid = sub_df["t"][mask_valid].to_numpy()
            quality = sub_df["quality"][mask_valid].to_numpy()
            n_valid = len(idx_valid)

            if False:
                print(f"{n_valid=}")
                breakpoint()
            if n_frames is None:
                all_clips["obs_id"].append(obs_id)
                all_clips["frame_idx"].append(idx_valid)
                all_clips["frame_quality"].append(quality)
                all_clips["n_valid"].append(n_valid)
                continue

            # compute idx_start
            idx_valid_new = np.arange(n_valid)
            # (n_valid)
            # (clip_idx, frames_idx)
            idx_start_new = idx_valid_new[::stride]
            offset = np.arange(0, n_frames * dilation, step=dilation)
            idx_table_new = idx_start_new.reshape(-1, 1) + offset.reshape(
                1, -1
            )
            # mask_inrange = idx_table_new < n_valid
            min_frames = n_frames_min
            if drop_short_clips:
                min_frames = n_frames

            max_clip = np.inf
            if clips_per_obs_max is not None:
                max_clip = clips_per_obs_max

            clip_counter = 0
            for clip in idx_table_new:
                if clip_counter >= max_clip:
                    break
                mask_inrange = clip < n_valid
                if np.sum(mask_inrange) < min_frames:
                    continue
                clip = clip[mask_inrange]
                _idx_valid = idx_valid[clip]
                _quality = quality[clip]
                all_clips["obs_id"].append(obs_id)
                all_clips["frame_idx"].append(_idx_valid)
                all_clips["frame_quality"].append(_quality)
                all_clips["n_valid"].append(n_valid)

                clip_counter += 1
        # breakpoint()
        all_clips = pd.DataFrame(all_clips)
        if n_obs_max is not None:
            print(f"[LMDBDataset] filtering n_obs_max")
            # breakpoint()
            # all_obs_id = list(set(all_clips["obs_id"]))
            all_obs_id = list(sorted(list(set(all_clips["obs_id"]))))
            print(f"{all_obs_id=}")
            n_obs = len(all_obs_id)
            print(f"n_obs init={n_obs}")
            gen = np.random.RandomState(seed=0)
            idx_obs = gen.permutation(n_obs)
            obs_valid = [all_obs_id[_idx] for _idx in idx_obs[:n_obs_max]]
            print(f"n_obs filter={len(obs_valid)}")
            print(f"{idx_obs=}")
            mask_n_obs = [
                obs_id in obs_valid for obs_id in all_clips["obs_id"]
            ]
            all_clips = all_clips.iloc[mask_n_obs]
            assert len(set(all_clips["obs_id"])) <= n_obs_max
        else:
            print(f"[LMDBDataset] no n_obs_max")

        self.all_clips = all_clips
        print("All clips:")
        print(f"{len(self.all_clips)=}")
        # print(f"{self.all_clips['frame_idx']=}")
        print("all_clips['obs_id']:")

        if self.n_repeat:
            print(f"Repeating: {self.n_repeat}")
            self.all_clips = pd.concat([self.all_clips] * self.n_repeat)
            print(f"{len(self.all_clips)=}")

        all_obs_id = list(set(all_clips["obs_id"]))
        for obs_id in all_obs_id:
            print(obs_id)
        # breakpoint()
        if len(all_clips) == 0:
            breakpoint()

        if True:
            mask_obs_selected = [
                (obs_id in list(self.all_clips["obs_id"]))
                for obs_id in self.df_global.obs_id
            ]
            # breakpoint()
            df_global_selected = self.df_global.iloc[
                mask_obs_selected
            ].set_index("obs_id")
            cwd = os.getcwd()
            split_name = self.filter_obs_params.get("split")
            path_csv = os.path.join(
                # cwd, f"df_global_selected_{self.filter_obs_params.split}.csv"
                cwd,
                f"df_global_selected_{split_name}.csv",
            )
            df_global_selected.to_csv(path_csv)
            print(f"df_global_selected saved to: \n{path_csv}")
        # breakpoint()
        self.obs_unique = list(set(self.all_clips.obs_id))
        print(f"[LMDBDataset] counting frames ..")
        all_frames = []
        for _, row in self.all_clips.iterrows():
            obs_id = row.obs_id
            frame_idx = row.frame_idx
            all_frames += [(obs_id, i) for i in frame_idx]

        n_frames = len(set(all_frames))
        if not self.multibands:
            if self.channel_idx is None:
                n_frames *= 2
        print(f"[LMDBDataset] {n_frames=}")

    def check_keys(self):
        keys_db = self.lmdb_reader.get_keys()
        obs_id_db = list(
            set(
                [
                    "__".join(k.split("__")[:3]).replace("b'", "")
                    for k in keys_db
                ]
            )
        )
        for obs_id in self.df_global.obs_id:
            assert obs_id in obs_id_db, f"{obs_id}"
        print("Keys OK!")

    def next_obs(self, idx):
        print("Next obs")
        idx_start = self.idx_frames[idx:]
        obs_idx = self.all_frames[idx_start, 0]
        offset = np.argmax(obs_idx != obs_idx[0])
        return idx + offset

    def prev_obs(self, idx):
        print("Prev obs")
        idx_start = self.idx_start[: idx + 1]
        obs_idx = self.all_frames[idx_start, 0][::-1]
        offset = np.argmax(obs_idx != obs_idx[0])
        return idx - offset

    def __len__(self):
        if self.multibands:
            return len(self.all_clips)
        if self.channel_idx is None:
            return 2 * len(self.all_clips)
        else:
            return len(self.all_clips)

    def __getitem__(self, idx):
        if self.multibands:
            channel_idx = [0, 1]
            idx_clip = idx
        else:
            if self.channel_idx is None:
                channel_idx = idx % 2
                idx_clip = idx // 2
            else:
                channel_idx = self.channel_idx
                idx_clip = idx
            channel_idx = [channel_idx]

        lbda = np.array([LBDAS_IRDIS[c] for c in channel_idx])

        obs_id = self.all_clips.iloc[idx_clip].obs_id
        frame_idx = self.all_clips.iloc[idx_clip].frame_idx
        quality = self.all_clips.iloc[idx_clip].frame_quality
        idx_obs = self.obs_unique.index(obs_id)
        weight_obs = 1 / self.all_clips.iloc[idx_clip].n_valid

        T = len(frame_idx)

        # frame_ids = [f"{obs_id}__{t:04d}_{channel_idx}" for t in frame_idx]
        frame_ids = [
            [f"{obs_id}__{t:04d}_{c}" for t in frame_idx] for c in channel_idx
        ]
        frame_ids_flat = []
        for c in channel_idx:
            frame_ids_flat += frame_ids[c]

        try:
            data = self.lmdb_reader.get(frame_ids_flat)
        except KeyNotFoundError:
            print(f"idx_clip: {idx_clip}")
            raise
        frame = np.stack([d["frame"] for d in data])
        CT, H, W = frame.shape
        frame = frame.reshape((len(channel_idx), T, H, W))
        rot = self.rot_scale * np.array([d["rot"] for d in data[:T]])
        assert len(rot) == T
        if self.n_frames and self.drop_short_clips:
            assert (
                frame.shape[1] == self.n_frames
            ), f"{frame.shape=}, {self.n_frames=}"
            assert len(rot) == self.n_frames, f"{len(rot)=}, {self.n_frames=}"

        # rot_diff = np.abs(rot.max() - rot.min())
        # print(f"{obs_id}: {rot_diff}")

        item = {
            "frame": frame.copy(),
            "rot": rot,
            "times": frame_idx.copy(),
            "quality": quality,
            # "psf_params": data[0]["psf_params"],
            # "idx": idx,
            "frame_ids": frame_ids,
            "obs_id": obs_id,
            "c": channel_idx,
            "split": -1,
            "idx": idx,
            "idx_obs": idx_obs,
            "weight_obs": 100 * weight_obs,
            "lbda": lbda,
        }
        for trans in self.transforms:
            item = trans(item)
            # breakpoint()
            pass

        return item


def main():
    from datasets.transforms import (
        StaticMasker,
        KnownSourcesMasker,
        KnownSourcesMaskerNew,
        FOVMasker,
        RelativeFluxNormalizer,
        StandardNormalizer,
        visualize_all,
    )
    from utils import view_dataset_item
    from scripts.train_test_split import to_ban
    from viz.reductions import ReductionLoader

    # path_db = "/home/theo/thoth/exo/new_project/code/data/db_small"
    path_db = "data/db_small"
    # path_db = "data/remote/db"
    # path_db = "data/db"
    dual_filter = "B_H"
    channel_idx = 0
    # n_frames = 16
    n_frames = 16
    sel_obs = [
        # lot of sources, negative rotation
        # "HD_104125__2015-06-03__23-27-24",
        # "HD_110411__2016-06-11__00-13-20",
        # lot of sources, positive rotation
        # "HD_159911__2016-04-16__09-34-10",
        # "HD_95086__2015-05-05__01-27-44",
        # "HD_95086__2015-05-11__23-44-00",
        # "HD_95086__2016-04-16__01-47-40",
        # "HD_56022__2016-03-30__00-11-02",
        # "HD_147553__2018-06-18__03-58-47",
    ]
    suspicious = [
        "2MASS_J20013718-3313139__2015-09-30__00-10-10",
        "HD_210049__2016-09-17__03-22-07",
        "HD_147553__2015-05-30__05-02-09",
        "HD_201919__2015-06-03__09-04-45",
        "HD_210049__2016-09-17__03-22-07",
        "HD_39060__2015-02-05__02-12-39",
        "TYC_5882-1169-1__2016-09-18__09-13-52",
        "TYC_7183-1477-1__2016-03-30__02-14-07",
    ]
    suspicious2 = [
        # "HD_108767B__2016-06-11__23-43-59",
        # "HD_141190__2015-03-31__08-08-20",
        # "HD_143637__2015-06-01__04-01-40",
        # "HD_143637__2015-06-04__03-02-47",
        # "HD_168210__2017-06-03__05-47-51",
        # "HD_197481__2015-05-30__09-11-51",
        "HD_143637__2017-03-20__08-45-00",
    ]
    suspicious3 = [
        "2MASS_J00172353-6645124",
        "TYC_5164-567-1__2015-05-15__09-17-27",
        "HD_29391__2016-12-13__03-48-07",
        "HD_29391__2016-12-12__04-52-59",
        "HD_147553__2016-03-28__08-37-36",
        "HD_32195__2015-09-30__09-05-02",
        "TYC_8047-232-1__2016-01-17__00-50-17",
        "HD_100453__2016-01-21__05-34-59",
        "HD_29391__2016-12-12__04-52-59",
        "TYC_5882-1169-1__2016-09-18__09-13-52",
        "TYC_486-4943-1__2015-05-31__07-41-39",
        "HD_326277__2016-03-30__09-20-08",
    ]

    filter_obs_params = {
        "n_frames_min": 32,
        # "n_frames_min": 16,
        # "n_frames_min": 64,
        "rot_min": 5,
        "frame_type": "all",
        # "frame_type": "aggressive",
        "dual_filter": dual_filter,
        # "split": "train",
        "split": None,
        # "split": "train",
        # "sel_obs": sel_obs,
        # "sel_obs": suspicious3,
        # "sel_obs": to_ban,
    }
    known_masker = KnownSourcesMaskerNew(path_db=path_db, channel=channel_idx)
    fov_masker = FOVMasker(channel=channel_idx)
    static_masker = StaticMasker(path_db=path_db)
    rel_normalizer = RelativeFluxNormalizer()
    reduction_loader = ReductionLoader()
    std_normalizer = StandardNormalizer(
        path_db=path_db,
        dual_filter=dual_filter,
        channel=channel_idx,
        # debug=True,
        debug=False,
    )
    transformations = [
        known_masker,
        rel_normalizer,
        # std_normalizer,
        fov_masker,
        # reduction_loader,
        static_masker,
    ]
    dataset = LMDBDataset(
        path_db=path_db,
        n_frames=n_frames,
        # extent_frames=1,
        # frames_per_obs_max=64,
        clip_per_obs_max=1,
        filter_obs_params=filter_obs_params,
        # extent_frames=16,
        # extent_frames=64,
        # split=None,
        channel_idx=channel_idx,
        transforms=transformations,
    )
    print(f"length: {len(dataset)}")

    # for i in tqdm(range(len(dataset))):
    for i in range(len(dataset)):
        item = dataset.__getitem__(i)
        print(f"{i+1}/{len(dataset)} {item['idx'][0]}")
        view_dataset_item(item)
        # visualize_all(item)
        # frame = item["frame"]
        # mask = item["mask"]
        # quality = item["quality"]
        # status = item["status"]
        # cube_3d_viewer(cube=frame, masks=mask, quality=quality, status=status)

    # 2717
    # for i in np.arange(2700, 2800):
    # print(i)
    # item = dataset.__getitem__(i)


def check_dataset():
    from datasets.transforms import RelativeFluxNormalizer

    # path_db = "data/db_small_test_2"
    path_db = "data/db_new"
    n_frames = 64
    cpo = None
    # dual_filter = "B_Ks"
    # dual_filter = None
    dual_filter = "DB_H23"
    channel_idx = 0
    filter_obs_params = {
        "n_frames_min": 16,
        "rot_min": 0,
        "frame_quality": 2,
        "dual_filter": dual_filter,
        "split": "train",
        "status": [0],
        "obs_category": None,
    }
    # transforms = [RelativeFluxNormalizer(path_db)]
    transforms = []

    dataset = LMDBDataset(
        suffix_lmdb="lmdb_256",
        path_db=path_db,
        n_frames=n_frames,
        stride=n_frames,
        drop_short_clips=False,
        clips_per_obs_max=cpo,
        filter_obs_params=filter_obs_params,
        multibands=True,
        channel_idx=None,
        transforms=transforms,
        dilation=1,
        rot_scale=1,
    )
    item = dataset.__getitem__(0)
    frame = item["frame"]
    rot = item["rot"]
    print(f"{frame.shape=}")
    print(f"{rot.shape=}")
    breakpoint()


if __name__ == "__main__":
    # main()
    check_dataset()
