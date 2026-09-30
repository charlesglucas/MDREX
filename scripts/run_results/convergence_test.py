import gc
import sys, pathlib, os
import argparse

# Convergence test of the optimizer: one synthetic configuration (shape, angle, contrast, couple), reconstructed
# from x = 0 with several stopping tolerances (ftol:xtol) of VMLMB. Same reconstruction as syntheticdata_shape.py.
# One file per tolerance, saved as soon as it is computed (relaunching resumes):
#   results/convergence/<shape>_angle<angle>_alpha<flux>_musmooth<mu>_musparse<mu>/ftol<v>_xtol<v>.npz
#   (x, objective f, number of evaluations, status, time, PSNR whole image / support)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mu-smooth", type=float, default=5e6)
parser.add_argument("--mu-sparse", type=float, default=1e5)
parser.add_argument("--shape", type=str, default="spiral", choices=["medium_ellipse", "spiral", "circle"])
parser.add_argument("--angle", type=int, default=0, choices=list(range(0, 325, 36)))
parser.add_argument("--flux", type=float, default=5e-6)
parser.add_argument("--tols", type=str, nargs="+", default=["1e-8:1e-6", "0:1e-6", "0:1e-8", "0:1e-10"],
                    help="ftol:xtol pairs")
args, remaining = parser.parse_known_args()
# Remove parsed args so Hydra doesn't complain.
sys.argv = [sys.argv[0]] + remaining

MU_SMOOTH = args.mu_smooth
MU_SPARSE = args.mu_sparse
SHAPE = args.shape
ANGLE = args.angle
FLUX = args.flux
TOLS = [tuple(float(v) for v in t.split(":")) for t in args.tols]

# Configure GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

import torch
import hydra
import numpy as np
import torch.nn.functional as F
import utils.optm as optm

from inference.inference import load_folder
from astropy.io import fits
from utils.rotation import BatchRotationOperator
from models import get_model

from reconstruction.mdrex import MDREX

from astropy.io import fits
from pathlib import Path

@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ------------------------------------------------------------
    # 1. Load nuisance data and pre-process it for the forward model
    # ------------------------------------------------------------
    
    # Load only selected frames
    path_folder = ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data"
    path_frame = ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/frame_selection_vector/"
    hdul = fits.open(path_frame / "ird_sortframes_vector_dc-IRD_FRAME_SELECTION_VECTOR-frame_selection_vector.fits")
    frames = hdul[0].data

    # Indices of frames to keep
    frames = np.asarray(frames).reshape(-1)
    idx = np.where(frames == 1)[0]

    ## Load data
    inputs = load_folder(path_folder=path_folder, use_centered=False, channel_sortframes=0, channel_idx=None,)
    y = inputs["y"].astype(np.float32) # (C, T, H, W)
    C, T, H, W = y.shape
    lbda = inputs["lbdas"].astype(np.float32)
    rot = inputs["rot"].astype(np.float32)
    psf = inputs["psf"].astype(np.float32)
    
    # Convert to torch tensors
    y = torch.tensor(y, device=device).unsqueeze(0) # (b, C, T, H, W)
    lbda = torch.tensor(lbda, device=device).unsqueeze(0) # (b, C)
    rot = torch.tensor(rot, device=device) #.unsqueeze(0)
    rot = rot.expand(C, -1)  # (C, T)
    psf = torch.tensor(psf, device=device)

    # Forward model preprocessing of PSF
    psf_size = psf.shape[-1]
    border = 15
    mask_out = torch.ones_like(psf)[0].bool()
    mask_out[border : psf_size - border, border : psf_size - border] = 0
    mean_0 = psf[0, mask_out].mean()
    mean_1 = psf[1, mask_out].mean()
    psf[0] = psf[0] - mean_0
    psf[1] = psf[1] - mean_1
    crop_size = 13
    psf_crop = psf[
        :,
        1 + (psf_size - crop_size) // 2 : 1 + (psf_size + crop_size) // 2,
        1 + (psf_size - crop_size) // 2 : 1 + (psf_size + crop_size) // 2,
    ]   
    psf_crop = psf_crop.view(2, 1, crop_size, crop_size) # (1, 1, crop, crop)
    device = y.device
    
    # Load coronograph mask
    path_coronograph = ROOT / "data/coronograph"
    k1k2_path = os.path.join(path_coronograph,"sphere_irdis_k1_k2_coronagraph_transmission_map.fits")
    with fits.open(k1k2_path, memmap=False) as hdul:
        mask = np.array(hdul[0].data, dtype=np.float32)
    deltaH = (mask.shape[1] - H) // 2; deltaW = (mask.shape[2] - W) // 2; 
    mask = mask[:, deltaH:deltaH+H, deltaW:deltaW+W] # (C, H, W)
    mask = torch.tensor(mask, device=device) # (C, H, W)
    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 2. Set-up EXOMILD coniguration and load pre-trained weights
    # ------------------------------------------------------------

    ## Data-fidelity term
    cfg_model = cfg.model
    # cfg_model.repeats = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # cfg_model.repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # cfg_model.repeats = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cfg_model.repeats = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = 256 # None
    cfg_model.n_channels = 2
    # ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/1_asdi/2024-11-09_20-10-01/banger_ms_bs16_lr5e-4_unetnormal_aug_111_100_100_100_seed5/ckpt/ckpt_40000.pt"
    ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/exomild_H2/ckpt/ckpt_40000.pt"
    print(f"Loading checkpoint {ckpt_path}")
    # new_repeats = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    new_repeats = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # new_repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # new_repeats = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    keep_indices = [i for i, v in enumerate(new_repeats) if v == 1]
    print("Conserved blocks :", keep_indices)
    state = torch.load(ckpt_path, map_location=device)["net"]
    state = {k.replace(".module.", "."): v for k, v in state.items()}
    state_cleaned = {}
    BLOCK_PREFIX = "model.blocks."   
    for key, val in state.items():
        if key.startswith(BLOCK_PREFIX):
            parts = key[len(BLOCK_PREFIX):].split(".")
            idx = int(parts[0])
            if idx in keep_indices:
                state_cleaned[key] = val
        else:
            state_cleaned[key] = val
    print("Conserved weights :", len(state_cleaned), "out of", len(state))
    cfg_model.repeats = new_repeats
    cfg_dict = dict(cfg_model)
    cfg_dict.pop('name', None)
    cfg_dict.pop('ckpt', None)
    model = get_model(name="exomild", ckpt=None, **cfg_dict)
    keys_to_remove = [k for k in state_cleaned.keys() if "weights_terms" in k or "features_pipeline.transforms.0.D" in k]
    for k in keys_to_remove:
        state_cleaned.pop(k, None)
    model.load_state_dict(state_cleaned, strict=False)
    model_state = model.state_dict()
    del model

    # ------------------------------------------------------------
    # 3. Ground truth loading
    # ------------------------------------------------------------

    mdrex = MDREX(y=y, rot=rot, psf=psf_crop, mask=mask, lbda=lbda, model_state=model_state, **cfg_model)

    # ------------------------------------------------------------
    # 4. Reconstructions with several tolerances
    # ------------------------------------------------------------
    import time

    def mu_to_str(mu):  # 1e6 -> "1e6", 5e5 -> "5e5"; 0 -> "0"
        if mu == 0:
            return "0"
        mantissa, exp = f"{mu:.0e}".split("e")
        return f"{mantissa}e{int(exp)}"

    def psnr(gt, x, support_threshold=0.04):  # same definition as plot_results/disk_rec_comp.py
        mask = gt > support_threshold * gt.max()
        whole = -20 * np.log10(np.sqrt(np.sum((gt - x)**2) / np.sum(gt**2)))
        supp = -20 * np.log10(np.sqrt(np.sum((gt[mask] - x[mask])**2) / np.sum(gt[mask]**2)))
        return whole, supp

    path_disk = ROOT / f"data/synthetic_disks/{SHAPE}/"
    with fits.open(path_disk / f"hid_fake_disk_image_{SHAPE}_{ANGLE}degrees.fits") as hdul:
        data = hdul[0].data
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128].astype(np.float32)
    with torch.no_grad():
        x_gt = FLUX*torch.tensor(datadisk, dtype=torch.float32, device=device)
        data_y = mdrex.forward_model(x_gt.unsqueeze(0).repeat(C, 1, 1)) + y
    gt = FLUX * datadisk

    outdir = (ROOT / "results" / "convergence"
              / f"{SHAPE}_angle{ANGLE}_alpha{mu_to_str(FLUX)}_musmooth{mu_to_str(MU_SMOOTH)}_musparse{mu_to_str(MU_SPARSE)}")
    outdir.mkdir(parents=True, exist_ok=True)
    for ftol, xtol in TOLS:
        outfile = outdir / f"ftol{mu_to_str(ftol)}_xtol{mu_to_str(xtol)}.npz"
        if outfile.exists():
            print(f"Skipping {outfile.name} (already computed)")
            continue
        t0 = time.time()
        with torch.no_grad():
            (xdisc, fx, gx, status) = mdrex.run_bfgs(np.zeros((C, H, W)), data_y, MU_SPARSE, MU_SMOOTH, ftol=ftol, xtol=xtol)
        elapsed = time.time() - t0
        x_mean = np.mean(xdisc, axis=0)
        pw, ps = psnr(gt, x_mean)
        print(f"ftol={ftol:g} xtol={xtol:g}: f={float(fx):.10e}  evaluations={mdrex.n_eval}  status={status}  "
              f"time={elapsed:.0f}s  PSNR whole={pw:.3f} support={ps:.3f}", flush=True)
        np.savez(outfile, x=xdisc, fx=fx, n_eval=mdrex.n_eval, status=str(status), time=elapsed, psnr_whole=pw,
                 psnr_supp=ps, ftol=ftol, xtol=xtol, mu_smooth=MU_SMOOTH, mu_sparse=MU_SPARSE, shape=SHAPE,
                 angle=ANGLE, flux=FLUX)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
