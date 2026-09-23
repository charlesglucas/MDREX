import hashlib
import os
import yaml
import pandas as pd
import pickle


def hash_folder(path_folder, n_digits=16):
    path_hash = os.path.join(path_folder, "hashes.yaml")

    filenames = {
        "frames_quality": "frames_quality.pkl",
        "psf_params": "psf_params.csv",
        "sources": "sources.csv",
        "static": "static.csv",
        "status_obs": "status_obs.csv",
    }

    hashes = {}
    hash_concat = ""
    for name, fn in filenames.items():
        path = os.path.join(path_folder, fn)
        if "csv" in fn:
            df = pd.read_csv(path)
        elif "pkl" in fn:
            df = pd.read_pickle(path)
        else:
            continue

        bytes_df = pickle.dumps(df)
        hash_str = hashlib.md5(bytes_df).hexdigest()
        hash_concat += hash_str
        hashes[name] = hash_str

    fn_global = "df_global.pkl"
    if fn_global in os.listdir(path_folder):
        print("df_global found")
        df = pd.read_pickle(os.path.join(path_folder, fn_global))
        bytes_df = pickle.dumps(df)
        hash_str = hashlib.md5(bytes_df).hexdigest()
        hash_concat += hash_str
        hashes["global"] = hash_str

    hash_all = hashlib.md5(hash_concat.encode()).hexdigest()[:n_digits]
    hashes["all"] = hash_all
    with open(path_hash, "w") as f:
        yaml.dump(hashes, f, default_flow_style=False)
    print(f"Hashes saved to {path_hash}")
    print(f"hash_all: {hash_all}")


if __name__ == "__main__":
    # path_folder = "metadata/interactive"
    path_folder = "data/db_new/metadata"
    hash_folder(path_folder)
