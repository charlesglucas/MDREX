import numpy as np
import yaml
import os
from yaml import Loader


def get_hash(path_db):
    path_hash = os.path.join(path_db, "metadata/hashes.yaml")
    with open(path_hash, "rb") as f:
        hashes = yaml.load(f, Loader=Loader)
    return hashes["all"]


def get_filter_id(**params):
    out = []
    keys = sorted(list(params.keys()))
    # for k, v in params.items():
    for k in keys:
        v = params[k]
        if k == "split":
            continue
        if k == "status":
            v = v[0]
        out.append(f"{k[0]}{v}")

    return "_".join(out)


def get_extraction_id(**params):
    out = []
    keys = sorted(list(params.keys()))
    # for k, v in params.items():
    for k in keys:
        v = params[k]
        out.append(f"{k[0]}{v}")

    return "_".join(out)


def get_path_stats(path_db, filter_obs_params):
    hash_meta = get_hash(path_db)
    filter_id = get_filter_id(**filter_obs_params)
    path_stats = os.path.join(path_db, "stats", hash_meta, filter_id)
    return path_stats
