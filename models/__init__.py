import torch
from omegaconf import ListConfig
from hydra.utils import to_absolute_path

from models.modelco.modelco import ModelCo
from models.exomild.exomild import ExoMILD


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(
    name,
    ckpt=None,
    **kwargs,
):
    if name == "modelco":
        model = ModelCo(**kwargs)
    elif name == "exomild":
        model = ExoMILD(**kwargs)
    elif name == "unet":
        # map to UNet implementation
        model = UNet(**kwargs)
    else:
        raise ValueError(f"Model not recognized: {name}")

    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_params_M = n_params / 1e6
    print(f"N params model: {n_params_M:.3f} M")
    print(model)
    if ckpt not in [None, -1]:
        print(f"Loading checkpoint {ckpt}")
        state = torch.load(to_absolute_path(ckpt), map_location=device)[
            "net"
        ]
        model.load_state_dict(state)
    return model
