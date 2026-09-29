import gc
import sys, pathlib, os
import argparse

# Grid of (mu_smooth, mu_sparse) for ONE geometry and ONE angle, at the 3 contrasts
# (same reconstruction as syntheticdata_all.py / syntheticdata_shape.py). One file per couple,
# saved as soon as it is computed:
#   results/grid_shape/<shape>_angle<angle>/musmooth<mu>_musparse<mu>.npz
#   (x_opt (3 fluxes, 256, 256), PSNR whole image / support, x_gt)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mu-smooth", type=float, nargs="+", default=[1e6, 5e6, 1e7], help="Values of mu_smooth")
parser.add_argument("--mu-sparse", type=float, nargs="+", default=[5e4, 1e5, 5e5], help="Values of mu_sparse")
parser.add_argument("--shape", type=str, default="spiral", choices=["medium_ellipse", "spiral", "circle"], help="Geometry to reconstruct")
parser.add_argument("--angle", type=int, default=0, choices=list(range(0, 325, 36)), help="Position angle of the disk (degrees)")
args, remaining = parser.parse_known_args()
# Remove parsed args so Hydra doesn't complain.
sys.argv = [sys.argv[0]] + remaining

MU_SMOOTH_LIST = args.mu_smooth
MU_SPARSE_LIST = args.mu_sparse
SHAPE = args.shape
ANGLE = args.angle

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
    # 4. Grid of (mu_smooth, mu_sparse) for one geometry and one angle
    # ------------------------------------------------------------
    def mu_to_str(mu):  # 1e6 -> "1e6", 5e5 -> "5e5"
        mantissa, exp = f"{mu:.0e}".split("e")
        return f"{mantissa}e{int(exp)}"

    def psnr(gt, x, support_threshold=2e-7):  # same definition as plot_results/disk_rec_comp.py
        mask = gt > support_threshold
        whole = -20 * np.log10(np.sqrt(np.sum((gt - x)**2) / np.sum(gt**2)))
        supp = -20 * np.log10(np.sqrt(np.sum((gt[mask] - x[mask])**2) / np.sum(gt[mask]**2)))
        return whole, supp

    path_disk = ROOT / f"data/synthetic_disks/{SHAPE}/"
    filename = f"hid_fake_disk_image_{SHAPE}_{ANGLE}degrees.fits"
    with fits.open(path_disk / filename) as hdul:
        data = hdul[0].data
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128].astype(np.float32)
    fluxes = [1e-6, 5e-6, 1e-5]

    outdir = ROOT / "results" / "grid_shape" / f"{SHAPE}_angle{ANGLE}"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    for mu_smooth in MU_SMOOTH_LIST:
        for mu_sparse in MU_SPARSE_LIST:
            outfile = outdir / f"musmooth{mu_to_str(mu_smooth)}_musparse{mu_to_str(mu_sparse)}.npz"
            if outfile.exists():  # allows resuming after a walltime interruption
                print(f"Skipping {outfile.name} (already computed)")
                continue
            x_opt = np.zeros((3, 256, 256), dtype=np.float32)
            psnr_whole = np.zeros(3); psnr_supp = np.zeros(3)
            for (f, flux) in enumerate(fluxes):
                with torch.no_grad():
                    xdisc = np.zeros((C, H, W))
                    x_gt = flux*torch.tensor(datadisk, dtype=torch.float32, device=device)
                    x_gt = x_gt.unsqueeze(0).repeat(C, 1, 1)
                    data_y = mdrex.forward_model(x_gt) + y
                    (xdisc, fx, gx, status) = mdrex.run_bfgs(xdisc, data_y, mu_sparse, mu_smooth)
                    x_opt[f] = np.mean(xdisc, axis=0)
                psnr_whole[f], psnr_supp[f] = psnr(flux * datadisk, x_opt[f])
                print(f"mu_smooth={mu_smooth:g} mu_sparse={mu_sparse:g} flux={flux:g}: "
                      f"PSNR whole={psnr_whole[f]:.2f} support={psnr_supp[f]:.2f}", flush=True)
                torch.cuda.empty_cache()
            np.savez(outfile, x_opt=x_opt, psnr_whole=psnr_whole, psnr_supp=psnr_supp, fluxes=fluxes,
                     x_gt=np.stack([flux * datadisk for flux in fluxes]),
                     mu_smooth=mu_smooth, mu_sparse=mu_sparse, shape=SHAPE, angle=ANGLE)
            print(f"Saved {outfile}", flush=True)
            summary.append((mu_smooth, mu_sparse, psnr_whole, psnr_supp))

    print(f"\nSummary {SHAPE}, angle {ANGLE} (PSNR whole / support for alpha = 1e-6, 5e-6, 1e-5)")
    for mu_smooth, mu_sparse, pw, ps in summary:
        print(f"  ({mu_smooth:g}, {mu_sparse:g}): " + "  ".join(f"{a:6.2f}/{b:6.2f}" for a, b in zip(pw, ps)))

if __name__ == "__main__":
    main()
