import sys, pathlib, os
import argparse

# Parse a few command-line options before Hydra consumes the rest.
# This allows launching multiple processes (one per GPU) with different parameters.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--data", type=str, required=True, help="Real data relative folder name")
# parser.add_argument("--frame", type=str, required=True, help="Real data relative folder name")
parser.add_argument("--datares", type=str, required=True, help="Real data relative folder name")
parser.add_argument("--band", type=str, required=True, help="Real data relative folder name")
parser.add_argument("--mu-smooth", type=float, nargs="+", required=True, help="Value(s) of mu_smooth, e.g. --mu-smooth 1e3 1e4")
parser.add_argument("--mu-sparse", type=float, nargs="+", required=True, help="Value(s) of mu_sparse, e.g. --mu-sparse 1e3 1e4")
parser.add_argument("--overwrite", action="store_true", help="Recompute existing result files")
args, remaining = parser.parse_known_args()
# Remove parsed args so Hydra doesn't complain.
sys.argv = [sys.argv[0]] + remaining

DATA = args.data
# FRAME = args.frame
BAND = args.band
DATARES = args.datares
MU_SMOOTHS = args.mu_smooth
MU_SPARSES = args.mu_sparse
OVERWRITE = args.overwrite
if DATA is None:
    raise ValueError("Missing --data argument for real data path")

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

def fmt_mu(mu):
    """1000.0 -> '1e3', 5e4 -> '5e4', 0.5 -> '5e-1' (used in file names)."""
    mant, exp = f"{mu:e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"

@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Load data
    #path_folder = ("/scratch/vasher/tbodrito/exo/data/real_data/HR_4796/2015-02-03")
    path_folder = ROOT / "data" / "real_data" / "DISKS_IRDIS_CHARLES" / DATA 
    if not path_folder.exists():
        raise FileNotFoundError(f"Real data path not found: {path_folder}")
    # path_folder = ("/scratch/vasher/tbodrito/exo/data/real_data/HIP_60074/2015-04-08")
    
    #
    # path_frame = ROOT / "data" / "real_data" / "DISKS_IRDIS_CHARLES" / FRAME
    # hdul = fits.open(path_frame / "ird_sortframes_vector_dc-IRD_FRAME_SELECTION_VECTOR-frame_selection_vector.fits")
    # frames = hdul[0].data

    # Indices of frames to keep
    # frames = np.asarray(frames).reshape(-1)
    # idx = np.where(frames == 1)[0]
    
    inputs = load_folder(path_folder=path_folder, use_centered=False, channel_sortframes=0, channel_idx=None,)
    y = inputs["y"].astype(np.float32) #[:,idx,:,:] # (C, T, H, W)
    C, T, H, W = y.shape
    lbda = inputs["lbdas"].astype(np.float32)
    rot = inputs["rot"].astype(np.float32) #[idx]
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
    path_coronograph = ROOT / f"data/coronograph/sphere_irdis_{BAND}_coronagraph_transmission_map.fits"
    k1k2_path = os.path.join(path_coronograph)
    with fits.open(k1k2_path, memmap=False) as hdul:
            mask = np.array(hdul[0].data, dtype=np.float32)
    deltaH = (mask.shape[1] - H) // 2; deltaW = (mask.shape[2] - W) // 2; 
    mask = mask[:, deltaH:deltaH+H, deltaW:deltaW+W] # (C, H, W)
    mask = torch.tensor(mask, device=device) # (C, H, W)
    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 2. Set-up forward model and regularization
    # ------------------------------------------------------------

    cfg_model = cfg.model
    # cfg_model.repeats = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    cfg_model.repeats = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = 128
    cfg_model.n_channels = 2
    # ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/1_asdi/2024-11-09_20-10-01/banger_ms_bs16_lr5e-4_unetnormal_aug_111_100_100_100_seed5/ckpt/ckpt_40000.pt"
    ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/exomild_H2/ckpt/ckpt_40000.pt"
    print(f"Loading checkpoint {ckpt_path}")
    new_repeats = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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
    # 3. Scheme over regularization parameters (one file per (mu_smooth, mu_sparse))
    # ------------------------------------------------------------

    mdrex = MDREX(y=y, rot=rot, psf=psf_crop, mask=mask, lbda=lbda, model_state=model_state, **cfg_model)

    outdir = ROOT / "results" / "realdata" / DATARES
    outdir.mkdir(parents=True, exist_ok=True)

    # Data saved once (not duplicated in every hyperparameter file)
    # Reference angle for north alignment and wavelengths, saved with each result
    # so that plots do not need the raw data (same convention as disk_rec_realdata.py)
    path_rotnth = path_folder / "ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits"
    if path_rotnth.exists():
        parallactic_angles = np.asarray(fits.getdata(path_rotnth), dtype=np.float32).reshape(-1)
        rotation_deg = -float(np.median(parallactic_angles[0]))
    else:
        print(f"Warning: {path_rotnth} not found, rotation_deg not saved")
        rotation_deg = np.nan
    lambda_files = list(path_folder.glob("**/*-lam.fits"))
    if len(lambda_files) == 1:
        wavelengths = np.asarray(fits.getdata(lambda_files[0])).reshape(-1)
    else:
        wavelengths = lbda.detach().cpu().numpy().reshape(-1)

    y_file = outdir / "y.npz"
    if not y_file.exists():
        # atomic write: several jobs (one per couple) may start at the same time
        tmp_file = outdir / f"y.tmp{os.getpid()}.npz"
        np.savez(tmp_file, y=y.detach().cpu().numpy())
        os.replace(tmp_file, y_file)

    for mu_smooth in MU_SMOOTHS:
        for mu_sparse in MU_SPARSES:
            outfile = outdir / f"musmooth{fmt_mu(mu_smooth)}_musparse{fmt_mu(mu_sparse)}.npz"
            if outfile.exists() and not OVERWRITE:
                print(f"Skipping {outfile.name} (already exists)")
                continue
            print(f"Running mu_smooth={mu_smooth:g}, mu_sparse={mu_sparse:g}")
            xdisc_0 = np.zeros((C, H, W))
            (xdisc_opt, fx, gx, status) = mdrex.run_bfgs(xdisc_0, y, mu_sparse, mu_smooth)
            with torch.no_grad():
                xtensor_opt = torch.tensor(xdisc_opt, dtype=torch.float32, device=y.device)
                Ax = mdrex.forward_model(xtensor_opt).cpu().numpy()
            np.savez(
                outfile,
                x=xdisc_opt,
                Ax=Ax,
                mu_smooth=mu_smooth,
                mu_sparse=mu_sparse,
                fx=fx,
                status=str(status),
                rotation_deg=rotation_deg,
                wavelengths=wavelengths,
            )
            print(f"Saved {outfile}")
            del xtensor_opt, Ax
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
