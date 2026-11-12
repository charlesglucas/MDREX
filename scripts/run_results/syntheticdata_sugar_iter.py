import gc
import sys, pathlib, os
import argparse

# Second version of syntheticdata_sugar.py (which is unchanged): MC-SUGAR by ITERATIVE differentiation of the
# reconstructions (reconstruction/sugar_iterative.py, vmlmb_diff: forward-mode tangents d x_k / d mu propagated along the
# iterations of VMLMB, L-BFGS metric frozen in the derivative) instead of the implicit differentiation with CG.
# Reconstructions: same objective as MDREX.objective on x >= 0, but with the gradient of mu_sparse sum(x) instead of the
# autograd gradient of mu_sparse sum|x|, which is 0 at x = 0 (instead of +mu_sparse): with it a pixel at 0 looks free to
# VMLMB, is pushed up and back, and VMLMB stalls (variables-change test with a projected gradient still ~1/10 of its
# initial norm) on a point that is not stationary (--abs-grad: autograd gradient of sum|x|, as in run_bfgs).
# --check: reconstructions with both gradients (convergence), MC-SUGAR (iterative; implicit CG for --check-cg) vs finite
# differences of MC-SURE computed with the same reconstructions.
# Results in results/sugar/<shape>_alpha<..>_angle<..>_<tag>[...]; default tag sugariter.
# MC-SURE minimized over (mu_smooth, mu_sparse) with its MC-SUGAR gradient, quasi-Newton
# with bounds (VMLMB, utils/optm.py, as for the reconstructions) on log10(mu). Same synthetic data as syntheticdata_sure_gpu.py (nuisance HIP 72192 + disk at one angle).
# --refined: refined MC-SURE (as on the grids, one iteration): metric re-estimated on the residual y - A x_hat(mu_pilot),
# mu_pilot = couple found by MC-SUGAR with the metric C^y (results/sugar/<...>_<tag>/trajectory.npz, mu_opt), start at
# mu_pilot; output folder suffixed _refined.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--flux", type=float, required=True)
parser.add_argument("--shape", type=str, required=True)
parser.add_argument("--angle", type=int, default=0)
parser.add_argument("--mu0", type=float, nargs=2, default=[1e7, 1e6], help="initial (mu_smooth, mu_sparse)")
parser.add_argument("--maxfun", type=int, default=25, help="maximum number of MC-SURE evaluations")
parser.add_argument("--cg-maxiter", type=int, default=20)
parser.add_argument("--cg-rtol", type=float, default=1e-2)
parser.add_argument("--hvp-step", type=float, default=1e-2, help="relative step of the finite-difference Hessian-vector products")
parser.add_argument("--seed", type=int, default=42, help="seed of the Monte Carlo probe delta (fixed)")
parser.add_argument("--tag", type=str, default="sugariter")
parser.add_argument("--ftol", type=float, default=1e-6,
                    help="relative tolerance on MC-SURE (VMLMB): about the reproducibility of MC-SURE at a given mu "
                         "(3-8 for |SURE| ~ 4e6, the reconstructions being stopped at ftol = 1e-8)")
parser.add_argument("--xtol", type=float, default=0.01,
                    help="absolute tolerance on the step in log10(mu) (VMLMB): 0.01 decade, i.e. 2%% on mu")
parser.add_argument("--gtol", type=float, default=1e-2,
                    help="tolerance on the projected gradient relative to its initial norm (VMLMB): about the "
                         "variability of MC-SUGAR at a given mu (1-2%%)")
parser.add_argument("--normalize", action="store_true",
                    help="MC-SURE and MC-SUGAR divided by the norm of MC-SUGAR (d/dlog10 mu) at the first evaluation, so that the "
                         "first L-BFGS-B step (identity initial metric) is about one decade instead of jumping to the bounds")
parser.add_argument("--refined", action="store_true", help="refined MC-SURE (metric from the residual of the C^y MC-SUGAR couple)")
parser.add_argument("--rec-ftol", type=float, default=1e-8,
                    help="relative tolerance of VMLMB on the objective for the reconstructions x_hat, x_hat_eps and the pilots "
                         "(default 1e-8; 0: converged reconstructions, as for the real data)")
parser.add_argument("--rec-xtol", type=float, default=1e-6,
                    help="relative tolerance of VMLMB on the variables for the reconstructions (default 1e-6)")
parser.add_argument("--outer", type=int, default=0,
                    help="fixed-point iterations on the covariance in the same job: after the minimization with the current "
                         "metric (C^y, or the refined one with --refined), the covariance is re-estimated on the residual of the "
                         "reconstruction at the couple found and MC-SURE is minimized again from this couple, at most --outer "
                         "times; stops when the couple moves by less than --outer-tol decade, or when mu_smooth reaches its lower "
                         "bound (drift toward weak regularizations: the previous iteration is kept). Folders <..>_refined<k>")
parser.add_argument("--outer-tol", type=float, default=0.1, help="tolerance (decade) of the fixed-point iterations")
parser.add_argument("--refined-iter", type=int, default=1,
                    help="iteration of the refinement (with --refined): k = 1 pilot = MC-SUGAR with C^y (folder <tag>), "
                         "k > 1 pilot = refined MC-SUGAR of iteration k-1 (folder <tag>_refined or <tag>_refined<k-1>); "
                         "output folder <tag>_refined (k = 1) or <tag>_refined<k>")
parser.add_argument("--max-step", type=float, default=float("inf"),
                    help="trust region (decade): a point farther than --max-step from the best couple evaluated so far is not "
                         "evaluated (no reconstruction) and gets a large MC-SURE, so that the line search of VMLMB backtracks "
                         "(the quasi-Newton steps of VMLMB are not bounded, only its first step)")
parser.add_argument("--gabs", type=float, default=0.0,
                    help="absolute tolerance on the projected MC-SUGAR gradient (d MC-SURE / d log10 mu, not normalized) at the "
                         "best couple: a stage is converged only below it (0: VMLMB tests only). In a narrow valley of MC-SURE, "
                         "VMLMB stops on its variables-change test after a few backtracked steps while the gradient is "
                         "still large; it is then restarted from the best couple (see --restarts)")
parser.add_argument("--srel", type=float, default=0.0,
                    help="stagnation test of a stage: stop when the best MC-SURE of the stage has decreased by less than "
                         "--srel (relative) over its last --sstall evaluations (0: off). In the flat valleys of MC-SURE "
                         "(high fluxes) the absolute test --gabs is never met and the stage otherwise runs to --maxfun")
parser.add_argument("--sstall", type=int, default=3, help="window (evaluations) of the stagnation test --srel")
parser.add_argument("--restarts", type=int, default=0,
                    help="maximum number of restarts of VMLMB from the best couple while the projected gradient exceeds --gabs "
                         "(within the --maxfun evaluations of the stage)")
parser.add_argument("--check", action="store_true",
                    help="diagnostic instead of the minimization: MC-SURE evaluated again at the same couple (--check-mu, "
                         "default --mu0) and MC-SUGAR compared with central finite differences (--check-steps, decade)")
parser.add_argument("--check-mu", type=float, nargs=2, default=None)
parser.add_argument("--check-steps", type=float, nargs="*", default=[0.05, 0.15])
parser.add_argument("--precond", action="store_true",
                    help="Jacobi preconditioner of the CG solves: exact diagonal of the Hessian of mu_smooth R_smooth "
                         "+ mean diagonal of the data term (reconstruction/sugar.py, jacobi_preconditioner)")
parser.add_argument("--check-cg", type=int, nargs="*", default=[],
                    help="--check: MC-SUGAR at the center (joint and split formulas) for these numbers of CG iterations")
parser.add_argument("--sugar-split", action="store_true",
                    help="MC-SUGAR with the split formula (3 solves, correlated errors of the 1/xi terms of d 2div / d mu)")
parser.add_argument("--check-repeats", type=int, default=2)
parser.add_argument("--abs-grad", action="store_true",
                    help="reconstructions with the autograd gradient of sum|x| (0 at x = 0), as run_bfgs (default: gradient of sum(x))")
parser.add_argument("--sugar-mode", choices=["implicit", "iterative"], default="implicit",
                    help="MC-SUGAR of the minimization: implicit (CG on H u = b, reconstruction/sugar.py, with the gradient of "
                         "sum(x): the reconstructions are then stationary and the CG converges in < 80 iterations) or "
                         "iterative (tangents of vmlmb_diff, frozen L-BFGS metric: inaccurate, check jobs of 2026-11-06)")
parser.add_argument("--central", action="store_true",
                    help="central finite differences for the Hessian-vector products of the tangents (default one-sided)")
parser.add_argument("--full-metric", action="store_true",
                    help="tangents of the L-BFGS memory propagated too (default: metric frozen in the derivative; unstable)")
parser.add_argument("--rec-verb", type=int, default=10, help="print every --rec-verb iterations of the reconstructions")
parser.add_argument("--grid-smooth", type=float, nargs="*", default=None,
                    help="grid mode (no minimization): log10 mu_smooth values; with --grid-sparse, reconstruction of y from x = 0 "
                         "at each couple and its MSE / PSNR -> <out>_grid/grid_<grid-name>.npz")
parser.add_argument("--grid-sparse", type=float, nargs="*", default=None, help="grid mode: log10 mu_sparse values")
parser.add_argument("--grid-name", type=str, default="0", help="grid mode: suffix of the output file (job splitting)")
parser.add_argument("--ckpt", type=str, default=None,
                    help="ExoMILD checkpoint (default exomild_H2, 6 terms); its config.yaml (two levels up) gives the terms it contains")
parser.add_argument("--proj", choices=["learned", "random"], default="learned",
                    help="projections D of the terms: learned (checkpoint, terms absent from it random) or all random")
parser.add_argument("--repeats", type=int, nargs=12, default=[1] * 12,
                    help="active terms of the data term, in the order p8 (s1, s2, s4), p16 (...), p32 (...), p64 (...) "
                         "(patch size p, symmetry s); default: the 12 terms")
parser.add_argument("--bounds", type=float, nargs=4, default=[4, 10, 3, 9],
                    help="bounds of log10(mu_smooth) (2 values) and log10(mu_sparse) (2 values)")
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
from reconstruction.load_exomild import remap_learned_projections

class StageStalled(Exception):
    """Raised by sure_and_sugar when the stagnation test (--srel) of the current stage is satisfied."""


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
    cfg_model.repeats = list(args.repeats)
    cfg_model.use_dataparallel = False
    cfg_model.batch_size = 256 #None
    cfg_model.n_channels = 2
    # ckpt_path = ROOT / "checkpoints_calib_exomild/checkpoints/1_asdi/2024-11-09_20-10-01/banger_ms_bs16_lr5e-4_unetnormal_aug_111_100_100_100_seed5/ckpt/ckpt_40000.pt"
    ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else ROOT / "checkpoints_calib_exomild/checkpoints/exomild_H2/ckpt/ckpt_40000.pt"
    print(f"Loading checkpoint {ckpt_path}")
    # new_repeats = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    new_repeats = list(args.repeats)
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
    # learned projections D of the checkpoint, moved to the rank of their term in this model (the D keys
    # removed above have the ranks of the training configuration, which differ from those of this one)
    model_state = remap_learned_projections(ckpt_path, cfg_model.repeats, model_state, learned=args.proj == "learned")

    # ------------------------------------------------------------
    # 3. Ground truth loading
    # ------------------------------------------------------------

    mdrex = MDREX(y=y, rot=rot, psf=psf_crop, mask=mask, lbda=lbda, model_state=model_state, **cfg_model)
    mdrex.sparse_abs = args.abs_grad

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
    from reconstruction.sugar import sugar_gradient, hvp_symmetry_test, edge_preserving_hess
    from reconstruction.sugar_iterative import vmlmb_diff, sugar_iterative

    fm, to_patches, centered = M.load_forward_model(device)
    with torch.no_grad():
        C_inv_y = M.shrinkage_cov_inv(centered(to_patches(y)))                 # metric of MC-SURE (from the data)
        C_inv_gt = M.shrinkage_cov_inv(centered(to_patches(y - fm(x_gt))))    # oracle metric of the MSE
        Ax_gt = fm(x_gt)
    N = y.numel()
    xi = (1e-1 * (y - y.median()).abs().median()).item()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)   # CPU: same probe on all the GPU models
    delta = torch.randn(y.shape, generator=gen).to(device)
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

    def objective(z, yy, mu_smooth, mu_sparse):
        # MDREX.objective, sparse term as mu_sparse sum(x) (same function on x >= 0, gradient mu_sparse at x = 0)
        # (since 2026-11-07 in MDREX itself: mdrex.sparse_abs = False; --abs-grad -> sparse_abs = True, former term)
        f, g = mdrex.objective(z, yy, mu_sparse, mu_smooth)
        return f, np.asarray(g, dtype=np.float64)

    def dgrad_mu(z):   # d grad F / d (mu_smooth, mu_sparse) at fixed x
        r_smooth, r_sparse = reg_grads(z)
        return np.array([r_smooth, r_sparse if args.abs_grad else np.ones_like(z)])

    def reconstruct(yy, mu_smooth, mu_sparse, tangents=True, label=""):
        # VMLMB from x = 0 (as run_bfgs: lower bound 0, mem 5, ftol / xtol of the reconstructions), with the tangents
        # d x / d (mu_smooth, mu_sparse) if tangents
        fg = lambda z: objective(z, yy, mu_smooth, mu_sparse)
        dg = dgrad_mu if tangents else (lambda z: np.zeros((0,) + z.shape))
        print(f"[rec{label}] mu_smooth={mu_smooth:.4e} mu_sparse={mu_sparse:.4e} tangents={tangents}", flush=True)
        x, f, g, st, X, info = vmlmb_diff(fg, np.zeros((C, H, W)), dg, lower=0.0, ftol=args.rec_ftol, xtol=args.rec_xtol,
                                          gtol=1e-5, hvp_step=args.hvp_step, central=args.central,
                                          freeze_metric=not args.full_metric, verb=args.rec_verb)
        free = optm.unblocked_variables(x, 0.0, None, g)
        g0 = fg(np.zeros((C, H, W)))[1]
        info.update(pg_ratio=float(np.linalg.norm(free * g) / np.linalg.norm(optm.unblocked_variables(np.zeros((C, H, W)), 0.0, None, g0) * g0)))
        print(f"[rec{label}] status {st}, {info}", flush=True)
        return x, st, X, info

    def reg_hess(mu_smooth):   # Hessian of the regularization for the Jacobi preconditioner (--precond), else None
        if not args.precond:
            return None
        return lambda x: edge_preserving_hess(x, mu_smooth, mdrex.l2_l1_edge_preserving_2d)

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
    theta0 = np.log10(np.array(args.mu0, dtype=float))
    if args.refined:
        # pilot: couple of the MC-SUGAR minimization with the metric C^y; the metric of the refined MC-SURE is the
        # inverse shrinkage covariance of the residual of its reconstruction (from x = 0), fixed during the minimization
        k = args.refined_iter
        pilot_dir = out_dir if k == 1 else out_dir.with_name(out_dir.name + ("_refined" if k == 2 else f"_refined{k - 1}"))
        traj = np.load(pilot_dir / "trajectory.npz", allow_pickle=True)
        if "mu_opt" in traj:
            mu_pilot = traj["mu_opt"]
        else:   # stage interrupted before its end: best couple evaluated so far
            i = int(np.argmin(traj["sure"]))
            mu_pilot = np.array([traj["mu_smooth"][i], traj["mu_sparse"][i]])
            print(f"Refined MC-SURE: {pilot_dir.name} interrupted, pilot = best of its {len(traj['sure'])} evaluations",
                  flush=True)
        print(f"Refined MC-SURE, iteration {k}: pilot from {pilot_dir.name}", flush=True)
        print(f"Refined MC-SURE: pilot mu = ({mu_pilot[0]:.4e}, {mu_pilot[1]:.4e})", flush=True)
        x_pilot = reconstruct(y, mu_pilot[0], mu_pilot[1], False, " pilot")[0]
        with torch.no_grad():
            C_inv_y = M.shrinkage_cov_inv(centered(to_patches(y - fm(torch.tensor(x_pilot, dtype=torch.float32, device=device)))))
        theta0 = np.log10(mu_pilot)
        out_dir = out_dir.with_name(out_dir.name + ("_refined" if k == 1 else f"_refined{k}"))
    state = {"x": np.zeros((C, H, W)), "x_eps": np.zeros((C, H, W)), "scale": 1.0}
    hist = []
    def sure_value(mu_smooth, mu_sparse, tangents=False):   # MC-SURE at a couple, reconstructions from x = 0
        x, st, X, ix = reconstruct(y, mu_smooth, mu_sparse, tangents, "")
        x_eps, st_eps, X_eps, ie = reconstruct(y_eps, mu_smooth, mu_sparse, tangents, " eps")
        D, gD = D_and_grad(x)
        div = divergence(x, x_eps)
        state["tangents"] = (X, X_eps, ix, ie)
        return x, x_eps, D, gD, div, D - N + 2 * div, st, st_eps

    def sure_and_sugar(theta):   # theta = log10(mu_smooth, mu_sparse)
        mu_smooth, mu_sparse = 10.0 ** theta
        for h in hist:   # couple already evaluated (restart of VMLMB from the best couple): no new reconstruction
            if np.array_equal(np.log10([h["mu_smooth"], h["mu_sparse"]]), theta):
                return h["sure"] * state["scale"], h["grad_theta"] * state["scale"]
        if hist and np.isfinite(args.max_step):
            best = min(hist, key=lambda h: h["sure"])
            dist = np.abs(theta - np.log10([best["mu_smooth"], best["mu_sparse"]])).max()
            if dist > args.max_step:   # outside the trust region: rejected without reconstruction
                print(f"\n=== rejected: mu_smooth={mu_smooth:.4e} mu_sparse={mu_sparse:.4e} is {dist:.2f} decade from the "
                      f"best couple (> --max-step {args.max_step})", flush=True)
                return (best["sure"] + abs(best["sure"])) * state["scale"], best["grad_theta"] * state["scale"]
        print(f"\n=== evaluation {len(hist) + 1}: mu_smooth={mu_smooth:.4e} mu_sparse={mu_sparse:.4e}", flush=True)
        # each mu: reconstructions from x = 0 (as for the grids), so that MC-SURE is a deterministic function of mu
        iterative = args.sugar_mode == "iterative"
        x, x_eps, D, gD, div, sure, st, st_eps = sure_value(mu_smooth, mu_sparse, tangents=iterative)
        state["x"], state["x_eps"] = x, x_eps
        X, X_eps, ix, ie = state["tangents"]
        if iterative:   # iterative differentiation of the two reconstructions
            g_mu, info = sugar_iterative(gD, g_delta, X, X_eps)
        else:   # implicit differentiation at the (stationary) reconstructions, gradient of sum(x)
            g_mu, info = sugar_gradient(lambda z: objective(z, y, mu_smooth, mu_sparse)[1],
                                        lambda z: objective(z, y_eps, mu_smooth, mu_sparse)[1], x, x_eps, gD, g_delta,
                                        lambda z: tuple(dgrad_mu(z)), cg_maxiter=args.cg_maxiter, cg_rtol=args.cg_rtol,
                                        verbose=False, hvp_step=args.hvp_step)
        info.update(evals=ix["evals"], evals_tan=ix["evals_tan"], evals_eps=ie["evals"], evals_tan_eps=ie["evals_tan"],
                    pg_ratio=ix["pg_ratio"], pg_ratio_eps=ie["pg_ratio"])
        g_theta = g_mu * np.array([mu_smooth, mu_sparse]) * np.log(10)               # d SURE / d log10(mu)
        mse, psnr, psnr_supp = quality(x)
        rec = dict(mu_smooth=mu_smooth, mu_sparse=mu_sparse, sure=sure, data_term=D, div=div, grad_mu=g_mu,
                   grad_theta=g_theta, mse=mse, psnr=psnr, psnr_supp=psnr_supp, status=str(st), status_eps=str(st_eps), **info)
        if args.normalize and not hist:   # scale fixed at the first evaluation (same function for all the iterations)
            state["scale"] = 1.0 / max(np.linalg.norm(g_theta), 1e-300)
            print(f"[normalize] MC-SURE and MC-SUGAR scaled by {state['scale']:.3e}", flush=True)
        rec["scale"] = state["scale"]
        hist.append(rec)
        print(f"[MC-SURE] {sure:.6e}  [MC-SUGAR d/dlog10mu] {g_theta}  [MSE] {mse:.4e}  [PSNR] {psnr:.2f} / {psnr_supp:.2f}  "
              f"[SUGAR iter] {info}", flush=True)
        np.savez(out_dir / "trajectory.npz", **{k: np.array([h[k] for h in hist]) for k in hist[0]}, x_last=x)
        hs = [h["sure"] for h in hist[state.get("n0", 0):]]
        if args.srel > 0 and len(hs) > args.sstall:   # stagnation test (--srel, --sstall)
            before, after = min(hs[:-args.sstall]), min(hs)
            if before - after <= args.srel * abs(before):
                raise StageStalled(f"MC-SURE stalled: relative decrease {(before - after) / abs(before):.1e} <= srel "
                                   f"{args.srel} over the last {args.sstall} evaluations")
        return sure * state["scale"], g_theta * state["scale"]

    b = args.bounds
    lower, upper = np.array([b[0], b[2]]), np.array([b[1], b[3]])

    def minimize(theta_start):
        # VMLMB (same quasi-Newton method with bounds as for the reconstructions); its first step is limited to
        # dxabs = 1 in log10(mu), i.e. one decade. A line-search instance of its own is required: the default `lnsrch`
        # argument of optm.vmlmb is a single LineSearch object shared by all the calls, so that the reconstructions run
        # inside sure_and_sugar (run_bfgs -> optm.vmlmb) would overwrite the state of the outer line search (initial value,
        # step), which then never converges and keeps evaluating the same point.
        # Restarts (--restarts, --gabs): from the best couple evaluated so far, with a fresh quasi-Newton memory, while
        # the projected gradient at this couple exceeds --gabs and evaluations remain.
        theta, n0 = np.clip(theta_start, lower, upper), len(hist)
        state["n0"] = n0
        for r in range(args.restarts + 1):
            stalled = None
            try:
                theta, fth, gth, status = optm.vmlmb(sure_and_sugar, theta, lower=lower, upper=upper,
                                                     maxeval=args.maxfun - (len(hist) - n0) + (1 if r > 0 else 0),
                                                     lnsrch=optm.LineSearch(), ftol=args.ftol,
                                                     xtol=(args.xtol, 0.0), gtol=(0.0, args.gtol), verb=1, throwerrors=False)
                message = optm.reason(status)
            except StageStalled as e:   # VMLMB interrupted by the stagnation test
                stalled, status = str(e), optm.FTEST_SATISFIED
                message = stalled
            print("\nVMLMB:", message, " theta =", theta, " mu =", 10.0 ** theta, flush=True)
            best = min(hist[n0:], key=lambda h: h["sure"])
            theta = np.log10([best["mu_smooth"], best["mu_sparse"]])
            g = np.array(best["grad_theta"], dtype=float)
            g[((theta <= lower + 1e-9) & (g > 0)) | ((theta >= upper - 1e-9) & (g < 0))] = 0.0   # projected gradient
            print(f"[stage] best couple mu = {10.0 ** theta}, MC-SURE {best['sure']:.6e}, |projected MC-SUGAR| "
                  f"{np.linalg.norm(g):.1f} (--gabs {args.gabs}), {len(hist) - n0}/{args.maxfun} evaluations", flush=True)
            if args.gabs <= 0 or np.linalg.norm(g) <= args.gabs:
                message = f"{message}; projected gradient {np.linalg.norm(g):.1f} <= gabs {args.gabs}" if args.gabs > 0 else message
                status = optm.GTEST_SATISFIED if args.gabs > 0 else status
                break
            if stalled:
                print(f"[stage] {stalled}: stage stopped", flush=True)
                break
            if len(hist) - n0 >= args.maxfun or r == args.restarts:
                message = f"not converged: projected gradient {np.linalg.norm(g):.1f} > gabs {args.gabs}"
                status = optm.TOO_MANY_EVALUATIONS
                break
            print(f"##### restart {r + 1} of VMLMB from the best couple", flush=True)
        np.savez(out_dir / "trajectory.npz", **{k: np.array([h[k] for h in hist]) for k in hist[0]}, x_last=state["x"],
                 mu_opt=10.0 ** theta, success=status in (optm.FTEST_SATISFIED, optm.GTEST_SATISFIED, optm.XTEST_SATISFIED),
                 message=str(message), optimizer="vmlmb")
        return theta

    if args.grid_smooth:
        # MSE / PSNR of the reconstructions (gradient of sum(x) unless --abs-grad) on a grid of couples, to locate the
        # MSE optimum of the corrected problem and compare it with the couples of the MC-SURE minimization
        out_dir = out_dir.with_name(out_dir.name + "_grid")
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for ts in args.grid_smooth:
            for tp in args.grid_sparse:
                x, st, _, info = reconstruct(y, 10.0 ** ts, 10.0 ** tp, False, "")
                mse, psnr, psnr_supp = quality(x)
                rows.append([ts, tp, mse, psnr, psnr_supp, info["pg_ratio"], info["evals"]])
                np.savez(out_dir / f"x_{ts:.3f}_{tp:.3f}.npz", x=np.asarray(x, dtype=np.float32))   # for post-processing
                print(f"[grid] log10 mu = ({ts:.3f}, {tp:.3f})  [MSE] {mse:.4e}  [PSNR] {psnr:.2f} / {psnr_supp:.2f}  "
                      f"(status {st}, pg_ratio {info['pg_ratio']:.2e})", flush=True)
                np.savez(out_dir / f"grid_{args.grid_name}.npz", rows=np.array(rows, dtype=float),
                         columns=np.array(["log10_mu_smooth", "log10_mu_sparse", "mse", "psnr", "psnr_supp", "pg_ratio",
                                           "evals"]))
        best = min(rows, key=lambda r: r[2])
        print(f"[grid] min MSE {best[2]:.4e} at mu = ({10 ** best[0]:.4e}, {10 ** best[1]:.4e})", flush=True)
        sys.exit(0)

    if args.check:
        # diagnostic, no minimization: reproducibility of MC-SURE at a couple (same mu evaluated again, then mu x (1+1e-6))
        # and MC-SUGAR compared with central finite differences of MC-SURE along each log10(mu) axis
        out_dir = out_dir.with_name(out_dir.name + "_check")
        out_dir.mkdir(parents=True, exist_ok=True)
        tc = np.log10(np.array(args.check_mu if args.check_mu else 10.0 ** theta0, dtype=float))
        if args.check_cg:
            # MC-SUGAR at the center, joint (u = H^{-1}(grad D - g_delta)) and split (3 solves) formulas, for several
            # numbers of CG iterations: the terms of the split formula are printed in d / d log10(mu)
            mu = 10.0 ** tc
            x, x_eps, D, gD, div, s, st, st_eps = sure_value(*mu)
            print(f"[check] center: log10 mu = {tc}, D {D:.2f}, 2 div {2 * div:.2f}, MC-SURE {s:.2f}", flush=True)
            grad_F = lambda z: objective(z, y, mu[0], mu[1])[1]
            grad_F_eps = lambda z: objective(z, y_eps, mu[0], mu[1])[1]
            reg_grads_cg = (lambda z: tuple(dgrad_mu(z)))
            jac = mu * np.log(10)
            res = []
            ks = sorted(args.check_cg)
            for split in (False,):   # one run of CG per linear system, gradient at each number of iterations
                path = sugar_gradient(grad_F, grad_F_eps, x, x_eps, gD, g_delta, reg_grads_cg, cg_maxiter=ks[-1],
                                      cg_rtol=args.cg_rtol, verbose=False, hvp_step=args.hvp_step, split=split,
                                      reg_hess=reg_hess(mu[0]), snapshots=ks)
                for k, g, info in path:
                    line = f"[check] CG {k}{' precond' if args.precond else ''}, {'split' if split else 'joint'}: MC-SUGAR {g * jac}"
                    if split:
                        line += (f", d D {info['dD'] * jac}, d 2div {info['d2div'] * jac} = {info['d2div_a'] * jac} - "
                                 f"{info['d2div_b'] * jac}")
                    if "precond_c" in info:
                        line += f", precond c {info['precond_c']:.3e}, max/min diag {info['precond_ratio']:.3e}"
                    print(line + f", CG residuals { {c: info[c] for c in info if c.startswith('cg_res')} }", flush=True)
                    res.append([k, split, *(g * jac)])
                np.savez(out_dir / f"check_cg{'_precond' if args.precond else ''}.npz", center=tc, rows=np.array(res, dtype=float))
        if not args.check_steps and not args.check_repeats:
            sys.exit(0)
        f0, g0 = sure_and_sugar(tc)
        f0, g0 = f0 / state["scale"], g0 / state["scale"]
        print(f"[check] center: MC-SURE {f0:.2f}, MC-SUGAR (iterative) d/dlog10 mu {g0}, {hist[-1]['dD']} + {hist[-1]['d2div']} "
              f"(d D + d 2div, d/dmu)", flush=True)
        rows = []
        def ev(theta, label):
            mu = 10.0 ** theta
            _, _, D, _, div, s, st, st_eps = sure_value(*mu)
            rows.append((label, *theta, D, 2 * div, s))
            print(f"[check] {label}: log10 mu = {theta}, D {D:.2f}, 2 div {2 * div:.2f}, MC-SURE {s:.2f} "
                  f"(status {st} / {st_eps})", flush=True)
            np.savez(out_dir / "check.npz", center=tc, sure0=f0, sugar=g0,
                     rows=np.array([r[1:] for r in rows]), labels=np.array([r[0] for r in rows]))
            return s
        reps = [f0] + [ev(tc, f"repeat {i + 1}") for i in range(args.check_repeats)]
        reps.append(ev(tc + np.log10(1 + 1e-6), "mu x (1+1e-6)"))
        print(f"[check] MC-SURE at the center: {np.array(reps)}, spread {np.ptp(reps):.2f}", flush=True)
        for h in args.check_steps:
            for a, name in enumerate(("mu_smooth", "mu_sparse")):
                e = np.zeros(2); e[a] = h
                fp, fm_ = ev(tc + e, f"+{h} {name}"), ev(tc - e, f"-{h} {name}")
                print(f"[check] h = {h} decade, {name}: finite difference {(fp - fm_) / (2 * h):.1f}, MC-SUGAR {g0[a]:.1f}, "
                      f"noise of the finite difference ~ {np.ptp(reps) / (2 * h):.1f}", flush=True)
        sys.exit(0)

    out_dir.mkdir(parents=True, exist_ok=True)
    theta = minimize(theta0)
    base = out_dir.name.split("_refined")[0]
    k0 = args.refined_iter if args.refined else 0
    path_theta = [theta]
    n_outer = args.outer
    if args.refined:   # continuation of the fixed point (MODE=refine): move of this stage from its pilot
        step = np.abs(theta - theta0).max()
        print(f"##### fixed-point iteration {k0}: couple moved by {step:.3f} decade", flush=True)
        if step < args.outer_tol:
            print(f"##### fixed point reached (< {args.outer_tol} decade)", flush=True)
            n_outer = 0
    for k in range(k0 + 1, k0 + 1 + n_outer):
        # covariance re-estimated on the residual of the reconstruction at the couple found, then MC-SURE minimized again
        mu = 10.0 ** theta
        print(f"\n##### fixed-point iteration {k}: pilot mu = ({mu[0]:.4e}, {mu[1]:.4e})", flush=True)
        x_pilot = reconstruct(y, mu[0], mu[1], False, " pilot")[0]
        with torch.no_grad():
            C_inv_y = M.shrinkage_cov_inv(centered(to_patches(y - fm(torch.tensor(x_pilot, dtype=torch.float32, device=device)))))
        out_dir = out_dir.with_name(base + ("_refined" if k == 1 else f"_refined{k}"))
        out_dir.mkdir(parents=True, exist_ok=True)
        state = {"x": np.zeros((C, H, W)), "x_eps": np.zeros((C, H, W)), "scale": 1.0}
        hist = []
        theta_new = minimize(theta)
        step = np.abs(theta_new - theta).max()
        print(f"##### fixed-point iteration {k}: couple moved by {step:.3f} decade", flush=True)
        if theta_new[0] <= lower[0] + 1e-3:
            print("##### mu_smooth at its lower bound (drift): previous iteration kept", flush=True)
            break
        theta = theta_new
        path_theta.append(theta)
        if step < args.outer_tol:
            print(f"##### fixed point reached (< {args.outer_tol} decade)", flush=True)
            break
    np.savez(out_dir.with_name(base + "_fixedpoint.npz"), mu_path=10.0 ** np.array(path_theta), mu_opt=10.0 ** theta)
    print("Final couple:", 10.0 ** theta, flush=True)


if __name__ == "__main__":
    main()
