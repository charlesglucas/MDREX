import torch
import sys
import pickle
import logging
import os
import lmdb

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class KeyNotFoundError(Exception):
    def __init__(self, msg=""):
        super().__init__(msg)


class LMDBWriter:
    def __init__(self, path, n_frames, item=None, multi=True):
        self.n_frames = n_frames
        self.multi = multi
        self.verbose = False

        # self.path_db = os.path.join(path, "lmdb")
        self.path_db = path
        self.log(f"Path db: {self.path_db}")
        if item is None:
            self.estimate_bytes()
        else:
            self._estimate_bytes(item)

        os.makedirs(self.path_db, exist_ok=True)
        self.env = lmdb.open(
            self.path_db,
            map_size=self.size_db_bytes,
            writemap=True,
            create=True,
            readahead=False,
            # map_async=True,
        )

    def log(self, txt):
        # log.info(f"[LMDBWriter] {txt}")
        print(f"[LMDBWriter] {txt}")

    def estimate_bytes(self):
        fake_item = {
            "frame": torch.randn((1024, 1024)).float(),
            "t": 100,
            "c": 2,
        }

        self._estimate_bytes(item=fake_item)

    def _estimate_bytes(self, item, margin=1.5):
        size_item_bytes = sys.getsizeof(pickle.dumps(item))
        self.size_db_bytes = int(size_item_bytes * 2 * self.n_frames * margin)
        size_db_Gib = self.size_db_bytes / (1024 ** 3)

        self.log(f"size item: {size_item_bytes} bytes")
        self.log(f"n_frames: {self.n_frames}")
        self.log(f"margin: {margin}")
        self.log(
            f"size db: {self.size_db_bytes} bytes ({size_db_Gib:.4f} Gib)"
        )

    def _write_multi(self, to_write):
        with self.env.begin(write=True) as txn:
            with txn.cursor() as curs:
                curs.putmulti(to_write)

    def _write_single(self, to_write):
        with self.env.begin(write=True) as txn:
            for (k, v) in to_write:
                txn.put(k, v)

    def add_frames(self, to_write):
        """
        Args:
        -----
        * to_write: list
        """
        if len(to_write) == 0:
            return
        if self.verbose:
            print("Writing to db")
        if self.multi:
            self._write_multi(to_write)
        else:
            self._write_single(to_write)
        if self.verbose:
            print("Writing to db ok")


class LMDBReader:
    def __init__(self, path):
        self.path = path
        print(f"lmdb_path: {self.path}")
        self.env = lmdb.open(
            # os.path.join(self.path, "lmdb"),
            self.path,
            readonly=True,
            lock=False,
            create=False,
            readahead=False,
        )
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

    def get_keys(self):
        print("Loading keys")
        with self.env.begin(write=False) as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        print("Loading keys ok")
        # return [str(k) for k in self.keys]
        return [k.decode("utf-8") for k in self.keys]

    def get(self, keys):
        """list"""
        assert isinstance(keys, list)
        with self.env.begin(write=False) as txn:
            with txn.cursor() as curs:
                keys_b = [k.encode("ascii") for k in keys]
                data = curs.getmulti(keys_b)

        # assert len(data) == len(keys), f"{len(data)}!={len(keys)}"
        if len(data) != len(keys):
            print(f"{len(data) != {len(keys)}}")
            print(keys)
            print([k in self.keys for k in keys_b])
            raise KeyNotFoundError()

        for i in range(len(keys)):
            # breakpoint()
            assert data[i][0] == keys_b[i]
            data[i] = pickle.loads(data[i][1])
        return data
