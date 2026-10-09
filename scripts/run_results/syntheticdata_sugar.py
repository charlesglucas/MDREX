import gc
import sys, pathlib, os
import argparse

# MC-SURE minimized over (mu_smooth, mu_sparse) with its MC-SUGAR gradient (reconstruction/sugar.py), quasi-Newton
# (L-BFGS-B) on log10(mu). Same synthetic data as syntheticdata_sure_gpu.py (nuisance HIP 72192 + disk at one angle).
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--flux", type=float, required=True)
parser.add_argument("--shape", type=str, required=True)
parser.add_argument("--angle", type=int, default=0)
parser.add_argument("--mu0", type=float, nargs=2, default=[1e7, 1e6], help="initial (mu_smooth, mu_sparse)")
parser.add_argument("--maxfun", type=int, default=25, help="maximum number of MC-SURE evaluations")
parser.add_argument("--cg-maxiter", type=int, default=20)
parser.add_argument("--cg-rtol", type=float, default=1e-2)
parser.add_argument("--seed", type=int, default=42, help="seed of the Monte Carlo probe delta (fixed)")
parser.add_argument("--tag", type=str, default="gradS")
args, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
FLUX, SHAPE, ANGLE, TAG = args.flux, args.shape, args.angle, args.tag

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
    cfg_model.batch_size = 256 #None
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
        residual_gt = y - mdrex.forward_model(x_gt)
        residual_opt = y - mdrex.forward_model(x_tensor_opt)
        theta_gt = mdrex.forward_model(x_gt) + torch.mean(residual_gt, dim=2, keepdim=True)
        theta_opt = mdrex.forward_model(x_tensor_opt) + torch.mean(residual_opt, dim=2, keepdim=True)
        diff = theta_gt - theta_opt
        diffp = mdrex.to_patches.forward(diff, lbda)
        params = mdrex.fit_params_noweight(residual_opt)
        Cinv = params["C_inv"]

        bsp, _, patch_h, patch_w = diffp.shape
        diff_vec = diffp.view(bsp, C, T, patch_h * patch_w).unsqueeze(-1)
        Cx = Cinv @ diff_vec
        mahalanobis = (diff_vec.transpose(-1, -2) @ Cx).squeeze(-1).squeeze(-1)
        return mahalanobis.sum().item()

    # def mahalanobis_mse(x_gt, x_tensor_opt):
    #     diffp = mdrex.to_patches.forward(mdrex.forward_model(x_gt) - mdrex.forward_model(x_tensor_opt), lbda)
    #     mse_total = diffp.pow(2).sum() 
    #     return mse_total.item()
    
    # Load disk
    flux = FLUX
    shape = SHAPE
    path_disk = ROOT / f"data/synthetic_disks/{shape}/"
    hdul = fits.open(path_disk / f"hid_fake_disk_image_{shape}_{0 if ANGLE is None else ANGLE}degrees.fits")
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
    # 3. MC-SURE (fixed metric C_y^{-1}, as in Fig. 5) and its MC-SUGAR gradient
    # ------------------------------------------------------------
    sys.path.insert(0, str(ROOT / "scripts/plot_results"))
    import disk_rec_sure_fixedW as M
    from scipy.optimize import minimize
    from reconstruction.sugar import sugar_gradient

    fm, to_patches, centered = M.load_forward_model(device)
    with torch.no_grad():
        C_inv_y = M.shrinkage_cov_inv(centered(to_patches(y)))                 # metric of MC-SURE (from the data)
        C_inv_gt = M.shrinkage_cov_inv(centered(to_patches(y - fm(x_gt))))    # oracle metric of the MSE
        Ax_gt = fm(x_gt)
    N = y.numel()
    xi = (1e-1 * (y - y.median()).abs().median()).item()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    delta = torch.randn(y.shape, generator=gen, device=device)
    y_eps = y + xi * delta
    ct = lambda z: z - z.mean(dim=2, keepdim=True)                            # per-channel temporal centering
    div_const = torch.sum(delta * delta.mean(dim=2, keepdim=True)).item()    # (1/xi) delta^T mean_t(xi delta)

    def D_and_grad(x):   # data term of MC-SURE and its gradient w.r.t. x
        xt = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)
        r = centered(to_patches(y - fm(xt)))
        D = torch.einsum("ltf,lfg,ltg->", r, C_inv_y, r)
        (g,) = torch.autograd.grad(D, xt)
        return D.item(), g.detach().cpu().numpy().astype(np.float64)

    xt0 = torch.zeros((C, H, W), dtype=torch.float32, device=device, requires_grad=True)   # gradient of the linear
    (g_delta,) = torch.autograd.grad((2 / xi) * torch.sum(delta * ct(fm(xt0))), xt0)        # part of 2 div
    g_delta = g_delta.detach().cpu().numpy().astype(np.float64)

    def divergence(x, x_eps):
        with torch.no_grad():
            dA = fm(torch.tensor(x_eps - x, dtype=torch.float32, device=device))
            return (torch.sum(delta * ct(dA)).item() / xi) + div_const

    def reg_grads(x):
        out = []
        for R in (mdrex.l2_l1_edge_preserving_2d, mdrex.l2_l1_sparse_2d):
            xt = torch.tensor(x, dtype=torch.float64, device=device, requires_grad=True)
            (g,) = torch.autograd.grad(R(xt), xt)
            out.append(g.detach().cpu().numpy())
        return out

    def quality(x):
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            mse = M.mahalanobis(centered(to_patches(Ax_gt - fm(xt))), C_inv_gt)
        g, xm = x_gt.mean(0).cpu().numpy(), x.mean(0)
        supp = g > 0.04 * g.max()
        e = xm - g
        return mse, -20 * np.log10(np.linalg.norm(e) / np.linalg.norm(g)), \
            -20 * np.log10(np.linalg.norm(e[supp]) / np.linalg.norm(g[supp]))

    def flux_to_str(f):
        m, e = f"{f:.0e}".split("e")
        return f"{m}em{abs(int(e))}"
    out_dir = ROOT / "results" / "sugar" / f"{shape}_alpha{flux_to_str(flux)}_angle{ANGLE}_{TAG}"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"x": np.zeros((C, H, W)), "x_eps": np.zeros((C, H, W))}
    hist = []

    def sure_and_sugar(theta):   # theta = log10(mu_smooth, mu_sparse)
        mu_smooth, mu_sparse = 10.0 ** theta
        print(f"\n=== evaluation {len(hist) + 1}: mu_smooth={mu_smooth:.4e} mu_sparse={mu_sparse:.4e}", flush=True)
        x, fx, gx, st = mdrex.run_bfgs(state["x"], y, mu_sparse, mu_smooth)            # warm start
        x_eps, _, _, st_eps = mdrex.run_bfgs(state["x_eps"], y_eps, mu_sparse, mu_smooth)
        state["x"], state["x_eps"] = x, x_eps
        D, gD = D_and_grad(x)
        div = divergence(x, x_eps)
        sure = D - N + 2 * div
        grad_F = lambda z: mdrex.objective(z, y, mu_sparse, mu_smooth)[1]
        grad_F_eps = lambda z: mdrex.objective(z, y_eps, mu_sparse, mu_smooth)[1]
        g_mu, info = sugar_gradient(grad_F, grad_F_eps, x, x_eps, gD, g_delta, reg_grads,
                                    cg_maxiter=args.cg_maxiter, cg_rtol=args.cg_rtol, verbose=True)
        g_theta = g_mu * np.array([mu_smooth, mu_sparse]) * np.log(10)               # d SURE / d log10(mu)
        mse, psnr, psnr_supp = quality(x)
        rec = dict(mu_smooth=mu_smooth, mu_sparse=mu_sparse, sure=sure, data_term=D, div=div, grad_mu=g_mu,
                   grad_theta=g_theta, mse=mse, psnr=psnr, psnr_supp=psnr_supp, status=str(st), status_eps=str(st_eps), **info)
        hist.append(rec)
        print(f"[MC-SURE] {sure:.6e}  [MC-SUGAR d/dlog10mu] {g_theta}  [MSE] {mse:.4e}  [PSNR] {psnr:.2f} / {psnr_supp:.2f}  "
              f"[CG] {info}", flush=True)
        np.savez(out_dir / "trajectory.npz", **{k: np.array([h[k] for h in hist]) for k in hist[0]}, x_last=x)
        return sure, g_theta

    theta0 = np.log10(np.array(args.mu0, dtype=float))
    res = minimize(sure_and_sugar, theta0, jac=True, method="L-BFGS-B", bounds=[(4, 10), (3, 9)],
                   options=dict(maxfun=args.maxfun, maxiter=args.maxfun))
    print("\nL-BFGS-B:", res.message, " theta =", res.x, " mu =", 10.0 ** res.x, flush=True)
    np.savez(out_dir / "trajectory.npz", **{k: np.array([h[k] for h in hist]) for k in hist[0]}, x_last=state["x"],
             mu_opt=10.0 ** res.x, success=res.success, message=str(res.message))


if __name__ == "__main__":
    main()
