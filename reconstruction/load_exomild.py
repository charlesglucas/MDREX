"""Load the learned projections of an ExoMILD checkpoint into a model with another set of active terms.

The terms of ExoMILD are the active entries (repeat = 1) of the 12 resolution x symmetry configurations
p8/p16/p32/p64 x s1/s2/s4 of conf/model/exomild.yaml, in this order; the checkpoint stores them as
all_terms.<k> with k the rank among the active terms of its training configuration (model.repeats of the
config.yaml of the checkpoint), which in general differs from the rank in the reconstruction model. The
projection D of each configuration active in both is copied to its new rank; a configuration not in the
checkpoint keeps its random initialisation (0.1 randn / sqrt(size_in)), and is reported.
"""
import pathlib
import torch
from omegaconf import OmegaConf

D_KEY = "engine.features_pipeline.transforms.0.D"


def ranks(repeats):
    """configuration index (0..11) -> rank of the term among the active ones"""
    act = [i for i, r in enumerate(repeats) if r]
    return {c: k for k, c in enumerate(act)}


def remap_learned_projections(ckpt_path, model_repeats, model_state, learned=True):
    """model_state: state_dict of the reconstruction model (modified in place and returned).
    learned=False keeps all projections random (behaviour before 2026-11-11)."""
    ckpt_path = pathlib.Path(ckpt_path)
    if not learned:
        print("[load_exomild] projections: random (not loaded)")
        return model_state
    train_repeats = list(OmegaConf.load(ckpt_path.parents[1] / "config.yaml").model.repeats)
    net = torch.load(ckpt_path, map_location="cpu")["net"]
    net = {k.replace(".module.", "."): v for k, v in net.items()}
    r_train, r_model = ranks(train_repeats), ranks(model_repeats)
    loaded, random = [], []
    for c, k_model in r_model.items():
        key_model = f"all_terms.{k_model}.{D_KEY}"
        if c in r_train:
            D = net[f"all_terms.{r_train[c]}.{D_KEY}"]
            assert D.shape == model_state[key_model].shape, (c, D.shape, model_state[key_model].shape)
            model_state[key_model] = D.to(model_state[key_model].device, model_state[key_model].dtype).clone()
            loaded.append(c)
        else:
            random.append(c)
    name = lambda c: f"p{8 * 2 ** (c // 3)}_s{[1, 2, 4][c % 3]}"
    print(f"[load_exomild] checkpoint terms {[name(c) for c in r_train]}")
    print(f"[load_exomild] learned projections: {[name(c) for c in loaded]}; random: {[name(c) for c in random]}")
    return model_state
