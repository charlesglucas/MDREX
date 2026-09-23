import gc
import sys, pathlib, os
import argparse

# Parse a few command-line options before Hydra consumes the rest.
# This allows launching multiple processes (one per GPU) with different parameters.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mu-smooth", type=float, default=None, help="Regularization weight smoothness (overrides grid search)")
parser.add_argument("--mu-sparse", type=float, default=None, help="Regularization weight sparsity (overrides grid search)")
parser.add_argument("--out", type=str, default=None, help="Output NPZ path (overrides default naming)")
args, remaining = parser.parse_known_args()
# Remove parsed args so Hydra doesn't complain.
sys.argv = [sys.argv[0]] + remaining

MU_SMOOTH = args.mu_smooth
MU_SPARSE = args.mu_sparse
OUTPUT_PATH = args.out

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[1]

# Configure GPU memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import hydra
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils.optm as optm

from models import get_model
from inference.inference import load_folder
from models.exomild.exomild import ExoMILD
from utils.rotation import BatchRotationOperator
from astropy.io import fits


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # 1. Real-world data loading and preprocessing
    # ------------------------------------------------------------

    # Load only selected frames
    path_folder = ROOT / "SyntheticDisks/DataAssessment/HIP_72192/2015-06-11/IRDIS/data"
    path_frame = ROOT / "SyntheticDisks/DataAssessment/HIP_72192/2015-06-11/IRDIS/frame_selection_vector/"
    hdul = fits.open(path_frame / "ird_sortframes_vector_dc-IRD_FRAME_SELECTION_VECTOR-frame_selection_vector.fits")
    frames = hdul[0].data

    # Indices of frames to keep
    frames = np.asarray(frames).reshape(-1)
    idx = np.where(frames == 1)[0]
    # idx = idx[:40]  # Keep only the first 40 frames for memory reasons

    # Load data
    inputs = load_folder(path_folder=path_folder, use_centered=False, channel_sortframes=0, channel_idx=None,)
    y = inputs["y"].astype(np.float32)[:,idx,:,:] # (C, T, H, W) 
    C, T, H, W = y.shape
    print(f"{y.shape=}")
    lbda = inputs["lbdas"].astype(np.float32)
    print(f"{lbda=}")
    rot = inputs["rot"].astype(np.float32)[idx] 
    print(f"{rot.shape=}")
    psf = inputs["psf"].astype(np.float32)
    print(f"{psf.shape=}")

    # Convert to torch tensors
    y = torch.tensor(y, device=device).unsqueeze(0) # (b, C, T, H, W)
    lbda = torch.tensor(lbda, device=device).unsqueeze(0) # (b, C)
    rot = torch.tensor(rot, device=device).unsqueeze(0)
    psf = torch.tensor(psf, device=device)
    print(f"{y.shape=}")
    print(f"{lbda.shape=}")
    print(f"{rot.shape=}")
    print(f"{psf.shape=}")

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

    # ------------------------------------------------------------
    # 2. Set-up forward model and regularization
    # ------------------------------------------------------------

    # Initialize rotation operator
    batch_rotation = BatchRotationOperator(device=device, in_size=H, out_size=H, mode="bicubic", zero_init=True,)

    # Forward model
    def forward(x_disc):
        # x_disc : (C, H, W)
        im = x_disc.unsqueeze(0)   # (1, C, H, W)
        im = im.expand(T, -1, -1, -1)        # (T, C, H, W)
        im = im.permute(1, 0, 2, 3)          # (C, T, H, W)
        im = batch_rotation.forward(x=im, rot=rot.expand(C,-1))  # (C, T, H, W)
        im = im.permute(1, 0, 2, 3)   
        im = F.conv2d(im, weight=psf_crop, padding="same", groups=C)  # (T, C, H, W)
        im = im.permute(1, 0, 2, 3) 
        im = im * mask.unsqueeze(1)  # mask: (C, 1, H, W)
        return im.unsqueeze(0) # (1, C, T, H, W)

    ## Data-fidelity term
    # Configuration parameters
    
    # cfg_model = cfg.model
    # cfg_model.repeats = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    # cfg_model.use_dataparallel = False
    # cfg_model.batch_size = None
    # cfg_model.n_channels = 2
    # # Load trained model
    # cfg_dict = dict(cfg_model)
    # cfg_dict.pop('name', None)
    # cfg_dict.pop('ckpt', None)
    # model = get_model(name="exomild",ckpt=None, **cfg_dict)
    # # Load checkpoint manually after cleaning the keys
    # ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/1_asdi/2024-11-09_20-10-01/banger_ms_bs16_lr5e-4_unetnormal_aug_111_100_100_100_seed5/ckpt/ckpt_40000.pt"
    # print(f"Loading checkpoint {ckpt_path}")
    # state = torch.load(ckpt_path, map_location=device)["net"]
    # # Clean keys .module. from checkpoint DataParallel
    # state_cleaned = {key.replace('.module.', '.'): val for key, val in state.items()}
    # model.load_state_dict(state_cleaned, strict=False)
    # model_state = model.state_dict()
    # cfg_model.use_dataparallel = False
    # del model
    # gc.collect()


    cfg_model = cfg.model
    cfg_model.repeats = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None
    cfg_model.n_channels = 2
    ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/1_asdi/2024-11-09_20-10-01/banger_ms_bs16_lr5e-4_unetnormal_aug_111_100_100_100_seed5/ckpt/ckpt_40000.pt"
    print(f"Loading checkpoint {ckpt_path}")
    new_repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
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
    gc.collect()


    # Regularization terms
    def l2_l1_edge_preserving_2d(x, epsilon=10**(-6)): 
        dx = x[:, :-1, 1:] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        dy = x[:, 1:, :-1] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        grad_sq = dx**2 + dy**2  # (C, H-1, W-1) 
        grad_sq = grad_sq.mean(dim=0)  # (H-1, W-1) 
        return torch.sum(torch.sqrt(grad_sq + epsilon**2))
    
    def l2_l1_sparse_2d(x):
        return torch.sum(abs(x))
    
    # ------------------------------------------------------------
    # 3. Ground truth loading
    # ------------------------------------------------------------

    # Load disk
    flux = 1e-5
    path_disk = ROOT / "SyntheticDisks/DataAssessment/medium_ellipse/"
    hdul = fits.open(path_disk / "hid_fake_disk_image_medium_ellipse_0degrees.fits")
    data = hdul[0].data

    # Ensure the byte order is native
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128]
    with torch.no_grad():
        x_gt = flux*torch.tensor(datadisk, dtype=torch.float32, device=device) 
        x_gt = x_gt.unsqueeze(0).repeat(C, 1, 1)
        y = forward(x_gt) + y 

    # ------------------------------------------------------------
    # 5. BFGS scheme for one set of regularization parameters
    # ------------------------------------------------------------
    
    def run_bfgs(x_disc_0, y, mu_sparse, mu_smooth, exomild):
        """
        Run BFGS optimization for given regularization parameters and return the optimized solution, function value, gradient, and status.
        """
        # --- objective + gradient ---
        def fg(x_disc):
            x_tensor = torch.tensor(x_disc, dtype=torch.float32, device=device, requires_grad=True)
            with torch.set_grad_enabled(True):
                im = forward(x_tensor)
                diff = y - im
                params = exomild.fit_params(diff, lbda)
                log_likelihood = exomild.get_log_likelihood(diff, lbda, params)
                phi = torch.stack([torch.sum(-ll) for ll in log_likelihood]).mean()
                reg_sparse = mu_sparse * l2_l1_sparse_2d(x_tensor)
                reg_smooth = mu_smooth * l2_l1_edge_preserving_2d(x_tensor) 
                loss = phi + reg_sparse + reg_smooth
            (grad_x,) = torch.autograd.grad(loss, x_tensor, retain_graph=False, create_graph=False, allow_unused=False, )
            fx = loss.detach().cpu().numpy().astype(np.float32)
            gx = grad_x.detach().cpu().numpy().astype(np.float32)
            del loss, grad_x, im, diff, log_likelihood, params, x_tensor
            torch.cuda.empty_cache()
            return fx, gx
        xdisc_opt, fx, gx, status = optm.vmlmb(fg, x_disc_0, verb=1, lower=0, maxiter=1, observer=None)
        return xdisc_opt, fx, gx, status

    # ------------------------------------------------------------
    # 6. Regularization parameter handling
    # ------------------------------------------------------------

    exomild = ExoMILD(**cfg_model).to(device)
    exomild.load_state_dict(model_state)

    # If the user specified parameters on the command line, run only that pair.
    if MU_SMOOTH is not None and MU_SPARSE is not None:
        mu_smooth = MU_SMOOTH
        mu_sparse = MU_SPARSE
        print(f"Running single combination: mu_smooth={mu_smooth}, mu_sparse={mu_sparse}")

        xdisc_0 = np.zeros((C, H, W))
        (xdisc_opt, fx, gx, status) = run_bfgs(xdisc_0, y, mu_sparse, mu_smooth, exomild)

        ## Compute MC-SURE and MSE
        x_tensor_opt = torch.tensor(xdisc_opt, dtype=torch.float32, device=device)

    # ------------------------------------------------------------
    # 7. Save results and visualize
    # ------------------------------------------------------------

    y_numpy = y.detach().cpu().numpy()
    x_gt_numpy = x_gt.detach().cpu().numpy()

    # Determine output path: allow overriding via CLI, else use a deterministic file name.
    if OUTPUT_PATH is not None:
        out_path = pathlib.Path(OUTPUT_PATH)
    else:
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True, parents=True)
        if MU_SMOOTH is not None and MU_SPARSE is not None:
            out_path = out_dir / f"results_hierarchic_mu_smooth{MU_SMOOTH}_mu_sparse{MU_SPARSE}.npz"
        else:
            out_path = out_dir / "results_hierarchic_ellipse1em5_grid_sure_40frames.npz"

    np.savez(out_path, y=y_numpy, x=xdisc_opt, x_gt=x_gt_numpy, mse=mse_total, sure=sure_total, mu_smooth=MU_SMOOTH, mu_sparse=MU_SPARSE)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()