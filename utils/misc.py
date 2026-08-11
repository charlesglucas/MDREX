import torch
import numpy as np
import random
from datetime import datetime


def set_seed(seed):
    print(f"SETTING SEED TO {seed}")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def get_timestamp(format="all"):
    now = datetime.now()
    if format == "all":
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    elif format == "time":
        return now.strftime("%H:%M:%S")


def make_orthogonal(A):
    """Assume that A is a tall matrix.

    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative.
    """
    X, tau = torch.geqrf(A)
    Q = torch.linalg.householder_product(X, tau)
    # The diagonal of X is the diagonal of R (which is always real) so we normalise by its signs
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_sample_speckles():
    device = torch.device("cpu")
    path_speckles = "data/other/speckles.pt"
    speckles = torch.load(path_speckles, map_location=device)
    path_psf = "data/other/psf.pt"
    psf = torch.load(path_psf, map_location=device)
    return speckles, psf


class ContextPlaceHolder:
    def __enter__(self):
        return

    def __exit__(self, *args):
        return
