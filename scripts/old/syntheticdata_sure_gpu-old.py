import gc
import sys, pathlib, os
import argparse

# Parse a few command-line options before Hydra consumes the rest.
# This allows launching multiple processes (one per GPU) with different parameters.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--mu-smooth", type=float, default=None, help="Regularization weight smoothness (overrides grid search)")
parser.add_argument("--mu-sparse", type=float, default=None, help="Regularization weight sparsity (overrides grid search)")
parser.add_argument("--flux", type=float, default=None)
parser.add_argument("--shape", type=str, default=None)
parser.add_argument("--out", type=str, default=None, help="Output NPZ path (overrides default naming)")
args, remaining = parser.parse_known_args()
# Remove parsed args so Hydra doesn't complain.
sys.argv = [sys.argv[0]] + remaining

MU_SMOOTH = args.mu_smooth
MU_SPARSE = args.mu_sparse
FLUX = args.flux
SHAPE = args.shape
OUTPUT_PATH = args.out

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
from models import get_model

from reconstruction.mdrex import MDREX

@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # 2. Set-up forward model and regularization
    # ------------------------------------------------------------

    ## Data-fidelity term
    cfg_model = cfg.model
    # cfg_model.repeats = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    cfg_model.repeats = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = None
    cfg_model.n_channels = 2
    # ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/1_asdi/2024-11-09_20-10-01/banger_ms_bs16_lr5e-4_unetnormal_aug_111_100_100_100_seed5/ckpt/ckpt_40000.pt"
    ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/exomild_H2/ckpt/ckpt_40000.pt"
    print(f"Loading checkpoint {ckpt_path}")
    # new_repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
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
    # 3. Ground truth loading
    # ------------------------------------------------------------

    mdrex = MDREX(y=y, rot=rot, psf=psf_crop, mask=mask, lbda=lbda, model_state=model_state, **cfg_model)

    def mahalanobis_mse(x_gt, x_tensor_opt):
        diffp = mdrex.to_patches.forward(mdrex.forward_model(x_gt) - mdrex.forward_model(x_tensor_opt), lbda)
        mse_total = diffp.pow(2).sum() 
        return mse_total.item()
    
    # Load disk
    flux = FLUX
    shape = SHAPE
    path_disk = ROOT / f"data/synthetic_disks/{shape}/"
    hdul = fits.open(path_disk / f"hid_fake_disk_image_{shape}_0degrees.fits")
    data = hdul[0].data

    # Ensure the byte order is native
    # Ensure the byte order is native
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().view(data.dtype.newbyteorder('='))
    datadisk = data[513-128:513+128, 513-128:513+128]
    with torch.no_grad():
        x_gt = flux*torch.tensor(datadisk, dtype=torch.float32, device=device) 
        x_gt = x_gt.unsqueeze(0).repeat(C, 1, 1)
        y = mdrex.forward_model(x_gt) + y 

    # ------------------------------------------------------------
    # 3. Scheme over regularization parameters
    # ------------------------------------------------------------
    
    # If the user specified parameters on the command line, run only that pair.
    if MU_SMOOTH is not None and MU_SPARSE is not None:
        mu_smooth = MU_SMOOTH
        mu_sparse = MU_SPARSE
        print(f"Running single combination: mu_smooth={mu_smooth}, mu_sparse={mu_sparse}")

        xdisc_0 = np.zeros((C, H, W))
        (xdisc_opt, fx, gx, status) = mdrex.run_bfgs(xdisc_0, y, mu_smooth, mu_sparse)

        ## Compute MC-SURE and MSE
        x_tensor_opt = torch.tensor(xdisc_opt, dtype=torch.float32, device=device)
        torch.manual_seed(42)
        sure_total, data_term, div_est = mdrex.compute_mc_sure(y, x_tensor_opt, xdisc_0, mu_sparse, mu_smooth)
        mse_total = mahalanobis_mse(x_gt, x_tensor_opt)
    else:
        s1 = 10; s2 = 10 # number of values for mu_smooth and mu_sparse
        n_smooth = 1; n_sparse = 1 # starting exponents for mu_smooth and mu_sparse
        MSE = np.zeros((s1,s2)); SURE = np.zeros((s1,s2)); DA = np.zeros((s1,s2)); DE = np.zeros((s1,s2))
        x_disk_store = np.empty((s1,s2), dtype=object)
        for k in range(s1):
            for j in range(s2):
                print(f"\nIterations {k+1,j+1} / {s1,s2} with mu_smooth=1e{(k+n_smooth)} and mu_sparse=1e{(j+n_sparse)}")
                xdisc_0 = np.zeros((C, H, W))
                mu_smooth = 10**(k+n_smooth)
                mu_sparse = 10**(j+n_sparse)
                (xdisc_opt, fx, gx, status) = mdrex.run_bfgs(xdisc_0, y, mu_smooth, mu_sparse)
                x_disk_store[k,j] = xdisc_opt
                
                x_tensor_opt = torch.tensor(xdisc_opt, dtype=torch.float32, device=device)
                sure_total, data_term, div_est = mdrex.compute_mc_sure(y, x_tensor_opt, xdisc_0, mu_sparse, mu_smooth)
                mse_total = mahalanobis_mse(x_gt, x_tensor_opt)

                # Store results
                MSE[k,j] = mse_total
                SURE[k,j] = sure_total
                DA[k,j] = data_term
                DE[k,j] = div_est
                x_disk_store[k, j] = xdisc_opt

                torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 7. Save results and visualize
    # ------------------------------------------------------------
    
    def flux_to_str(flux):
        s = f"{flux:.0e}"   # ex: "1e-06"
        mantissa, exp = s.split("e")
        return f"{mantissa}em{abs(int(exp))}"

    y_numpy = y.detach().cpu().numpy()
    x_gt_numpy = x_gt.detach().cpu().numpy()

    # Determine output path: allow overriding via CLI, else use a deterministic file name.
    if OUTPUT_PATH is not None:
        out_path = pathlib.Path(OUTPUT_PATH)
    else:
        out_dir = ROOT / f"results/grids_111111111111/grid_sure_{shape}_alpha{flux_to_str(flux)}/"
        out_dir.mkdir(exist_ok=True, parents=True)
        if MU_SMOOTH is not None and MU_SPARSE is not None:
            out_path = out_dir / f"musmooth{MU_SMOOTH}_musparse{MU_SPARSE}.npz"
        else:
            out_path = out_dir / "results.npz"

    np.savez(out_path, y=y_numpy, x=xdisc_opt, x_gt=x_gt_numpy, mse=mse_total, sure=sure_total, mu_smooth=MU_SMOOTH, mu_sparse=MU_SPARSE, data_term=data_term, div_est=div_est)
    print(f"Saved results to {out_path}")
    
    
    # plt.figure(1)
    # plt.subplot(2,4,1); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0^{(1)}$"); plt.colorbar()
    # plt.subplot(2,4,5); plt.imshow(np.squeeze(y.detach().cpu().numpy()[0,1,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0^{(2)}$"); plt.colorbar()
    # plt.subplot(2,4,2); plt.imshow(x_opt[0,:,:].detach().cpu().numpy()); plt.title(r"$\mathbf{x}^{(1)}$"); plt.colorbar()
    # plt.subplot(2,4,6); plt.imshow(x_opt[1,:,:].detach().cpu().numpy()); plt.title(r"$\mathbf{x}^{(2)}$"); plt.colorbar()
    # plt.subplot(2,4,3); plt.imshow(np.squeeze(im.detach().cpu().numpy()[0,0,0,:,:])); plt.title(r"$\mathbf{A_0 x}^{(1)}$"); plt.colorbar()
    # plt.subplot(2,4,7); plt.imshow(np.squeeze(im.detach().cpu().numpy()[0,1,0,:,:])); plt.title(r"$\mathbf{A_0 x}^{(2)}$"); plt.colorbar()
    # plt.subplot(2,4,4); plt.imshow(np.squeeze(diff.detach().cpu().numpy()[0,0,0,:,:])); plt.title(r"$(\mathbf{y}_0-\mathbf{A_0 x})^{(1)}$"); plt.colorbar()
    # plt.subplot(2,4,8); plt.imshow(np.squeeze(diff.detach().cpu().numpy()[0,1,0,:,:])); plt.title(r"$(\mathbf{y}_0-\mathbf{A_0 x})^{(2)}$"); plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("gridsearch.jpg", dpi=300)

if __name__ == "__main__":
    main()