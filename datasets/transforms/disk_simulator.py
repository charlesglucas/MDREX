import torch
import numpy as np
from scipy.stats import qmc

from DDiT import Disk
from utils.rotation import BatchRotationOperator
import torch.nn.functional as F

class DiskParamsSampler:
    """
    Randomly sample disk parameters
    """
    def __init__(self):
        pass

    def __call__(self, item):
        C = item["psf"].shape[0]
        gpol=0
        # a=1; incl=0; e=0.2; omega=0; pa=90; pin=10; gsca=0.2; opang=0.2
        rng = np.random.default_rng()
        a = rng.uniform(0.6, 2)
        e = rng.uniform(0.0, 0.3)
        pa = rng.uniform(0.0, 360.0)
        incl = rng.uniform(-90.0, 90.0)
        omega = rng.uniform(0.0, 360.0)
        opang = rng.uniform(0.01, 0.2)
        gsca = rng.uniform(-0.8, 0.8)
        pin = rng.uniform(3.0, 30.0)
        pout = -pin    
        alpha = 10**rng.uniform(-6.5, -5, C)
        item["disk_params"] = dict(incl=incl, e=e, a=a, omega=omega, pa=pa, pin=pin, pout=pout, gsca=gsca, gpol=gpol, opang=opang, alpha=alpha)
        return item

class DiskParamsSamplerSobol:
    """
    Sample disk parameters using a Sobol quasi-random sequence,
    including alpha, generating points progressively without knowing N_iter.
    """
    def __init__(self, seed=None):
        """
        seed : random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.d = 9  # 8 geometric parameters + alpha

        # Initialize Sobol sequence sampler
        self.sampler = qmc.Sobol(d=self.d, scramble=True, seed=seed)

        # Define physical bounds for each parameter
        self.bounds = np.array([
            [0.6, 2.0],        # a
            [0.0, 0.3],        # e
            [0.0, 360.0],      # pa
            [-90.0, 90.0],     # incl
            [0.0, 360.0],      # omega
            [0.01, 0.2],       # opang
            [-0.8, 0.8],       # gsca
            [3.0, 30.0],       # pin
            [-6.5, -5.0],      # log10(alpha)
        ])

    def __call__(self, item):
        """
        Sample disk parameters for a single item.
        item: dictionary containing 'psf' array of shape (C, H, W)
        Returns item with 'disk_params' dictionary added.
        """
        C = item["psf"].shape[0]  # number of channels
        gpol = 0  # polarization parameter, fixed

        # Generate next Sobol point (shape 1x9)
        u = self.sampler.random(n=1)

        # Scale Sobol point to physical bounds
        Theta = qmc.scale(u, self.bounds[:, 0], self.bounds[:, 1])
        Theta = Theta[0]  # extract 1D array

        # Unpack parameters
        a, e, pa, incl, omega, opang, gsca, pin, log_alpha = Theta
        pout = -pin

        # Convert log_alpha to linear scale and replicate for all channels
        alpha = np.full(C, 10 ** log_alpha)
        # log_alpha_base = log_alpha
        # alpha = 10 ** (log_alpha_base + self.rng.normal(0, 0.2, size=C))

        # Store disk parameters in the item dictionary
        item["disk_params"] = dict(
            incl=incl, e=e, a=a, omega=omega, pa=pa,
            pin=pin, pout=pout, gsca=gsca, gpol=gpol,
            opang=opang, alpha=alpha
        )

        return item


class DiskParamsSamplerLHS:
    """
    Sample disk parameters using Latin Hypercube Sampling (LHS),
    with block-wise generation.
    """
    def __init__(self, seed=None, block_size=256):
        self.rng = np.random.default_rng(seed)
        self.d = 9
        self.block_size = block_size
        self.buffer = []

        self.sampler = qmc.LatinHypercube(d=self.d, seed=seed)

        self.bounds = np.array([
            [0.6, 2.0],        # a
            [0.0, 0.3],        # e
            [0.0, 360.0],      # pa
            [-90.0, 90.0],     # incl
            [0.0, 360.0],      # omega
            [0.01, 0.2],       # opang
            [-0.8, 0.8],       # gsca
            [3.0, 30.0],       # pin
            [-6.5, -5.0],      # log10(alpha)
        ])

    def __call__(self, item):
        C = item["psf"].shape[0]
        gpol = 0

        # Génération par blocs
        if len(self.buffer) == 0:
            u = self.sampler.random(n=self.block_size)
            Theta_block = qmc.scale(
                u, self.bounds[:, 0], self.bounds[:, 1]
            )
            self.buffer = list(Theta_block)

        Theta = self.buffer.pop(0)

        # Unpack
        a, e, pa, incl, omega, opang, gsca, pin, log_alpha = Theta
        pout = -pin

        # Convert log_alpha to linear scale and replicate for all channels
        alpha = np.full(C, 10 ** log_alpha)

        item["disk_params"] = dict(
            incl=incl, e=e, a=a, omega=omega, pa=pa,
            pin=pin, pout=pout, gsca=gsca, gpol=gpol,
            opang=opang, alpha=alpha
        )

        return item


class DiskInjector:
    """
    Transform that injects a synthetic disk into the data
    """
    def __init__(self, obj_size): 
        self.obj_size = obj_size
        self.disk = None
        self.rot_op = None

    def _init_disk(self):
        self.disk = Disk(nx=self.obj_size)

    def _init_rot_op(self, H):
        self.rot_op = BatchRotationOperator(
            device="cpu",
            in_size=H,
            out_size=H,
            mode="bicubic",
            zero_init=True,
        )

    def __call__(self, item):
        self.disk = Disk(nx=self.obj_size)

        psf = item["psf"]
        s_0 = item["s_0"]
        rot = item["rot"]
        coronograph_mask = item["coronograph_mask"]
        params = item["disk_params"]
    
        rot = torch.tensor(rot, dtype=torch.float32).unsqueeze(0)
        psf = torch.tensor(psf, dtype=torch.float32)
        s_0 = torch.tensor(s_0, dtype=torch.float32)
        coronograph_mask = torch.tensor(coronograph_mask.astype('float'), dtype=torch.float32)
        _ , p2, p3 = psf.shape
        psf = psf.view(1, 1, p2, p3)
        # psf = psf.expand(2, 1, p2, p3)

        C, T, H, W = s_0.shape

        # Lazy init dans le worker 
        if self.disk is None: 
            self._init_disk() 
        if self.rot_op is None: 
            self._init_rot_op(H)

        # Load disk parameters and compute model
        self.disk.compute_model(
            incl=params["incl"], 
            e=params["e"], 
            a=params["a"], 
            omega=params["omega"], 
            pa=params["pa"], 
            pin=params["pin"], 
            pout=params["pout"], 
            gsca=params["gsca"], 
            gpol=params["gpol"], 
            opang=params["opang"]
            )
        alpha = torch.tensor(params["alpha"], dtype=torch.float32)
        
        # Normalize disk intensity and scale with alpha
        im = self.disk.intensity
        x_min = im.min(); x_max = im.max();  disk_norm = (im - x_min) / (x_max - x_min) 
        disk_norm = torch.tensor(disk_norm, dtype=torch.float32)
        disk_norm = disk_norm.view(1, H, W).expand(C, -1, -1) # (C, H, W)
        disk = disk_norm * alpha

        # Rotate disk, convolve with psf and apply coronograph mask
        im = disk.view(C, 1, H, W).expand(-1, T, -1, -1)  # (C, T, H, W)
        im = self.rot_op.forward(x=im, rot=rot)
        im = im.permute(1, 0, 2, 3) # (T, C, H, W)
        im = F.conv2d(im, weight=psf, padding='same', groups=C)    # (T, C, H, W)
        im = im.permute(1, 0, 2, 3)  # (C, T, H, W)
        im = im * coronograph_mask.unsqueeze(0).unsqueeze(0)

        # Inject disk
        y = s_0 + im
        item["y"] = np.array(y)
        item["obj"] = np.array(disk)

        # item["y"] = np.zeros((1,64,256,256)).astype('float32')
        # item["obj"] = np.zeros((256,256)).astype('float32')

        return item
