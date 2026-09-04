import sys, pathlib, os

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
from models.exomild.exomild import ExoMILD
from utils.rotation import BatchRotationOperator
from models import get_model

@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Load data
    #path_folder = ("/scratch/vasher/tbodrito/exo/data/real_data/HR_4796/2015-02-03")
    # path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HR_4796/2015-02-03/IRDIS/data/"
    # path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/PDS_70/2018-02-24/IRDIS/data/"
    # path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/RY_lup/2016-04-16/IRDIS/data/"
    # path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/SAO_206462/2015-05-15/IRDIS/data/"
    # path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/RX_J161533255/2019-05-18/IRDIS/data/"
    path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/AB_AURIGAE/2020-01-18/IRDIS/data/"
    # path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HD_106906/2016-03-28/IRDIS/h2_h3/data/"
    # path_folder = ROOT / "data/real_data/DISKS_IRDIS_CHARLES/HD_202917/2017-05-16/data/"
    # path_folder = ("/scratch/vasher/tbodrito/exo/data/real_data/HIP_60074/2015-04-08")
    inputs = load_folder(path_folder=path_folder, use_centered=False, channel_sortframes=0, channel_idx=None,)
    y = inputs["y"].astype(np.float32) # (C, T, H, W)
    C, T, H, W = y.shape
    lbda = inputs["lbdas"].astype(np.float32)
    print(f"{y.shape=}")
    rot = inputs["rot"].astype(np.float32)
    print(f"{rot.shape=}")
    psf = inputs["psf"].astype(np.float32)
    print(f"{psf.shape=}")
    
    # Convert to torch tensors
    y = torch.tensor(y, device=device).unsqueeze(0) # (b, C, T, H, W)
    lbda = torch.tensor(lbda, device=device).unsqueeze(0) # (b, C)
    rot = torch.tensor(rot, device=device) #.unsqueeze(0)
    rot = rot.expand(C, -1)  # (C, T)
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
    torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # 2. Set-up forward model and regularization
    # ------------------------------------------------------------

    # Initialize rotation operator
    batch_rotation = BatchRotationOperator(device=device, in_size=H, out_size=H, mode="bicubic", zero_init=True,)

    ## Forward model
    def forward(x_disc):
        im = x_disc.unsqueeze(0)   # (1, C, H, W)
        im = im.expand(T, -1, -1, -1)        # (T, C, H, W)
        im = im.permute(1, 0, 2, 3)          # (C, T, H, W)
        im = batch_rotation.forward(x=im, rot=rot.expand(C,-1))  # (C, T, H, W) .expand(C,-1)
        im = im.permute(1, 0, 2, 3)   
        im = F.conv2d(im, weight=psf_crop, padding="same", groups=C)  # (T, C, H, W)
        im = im.permute(1, 0, 2, 3) 
        im = im * mask.unsqueeze(1)  # mask: (C, 1, H, W)
        return im.unsqueeze(0) # (1, C, T, H, W)

    ## Data-fidelity term
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

    ## Regularization term
    def l2_l1_edge_preserving_2d(x, epsilon=10**(-6)): 
        # x : (C, H, W) 
        dx = x[:, :-1, 1:] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        dy = x[:, 1:, :-1] - x[:, :-1, :-1]  # (C, H-1, W-1) 
        grad_sq = dx**2 + dy**2  # (C, H-1, W-1) 
        grad_sq = grad_sq.mean(dim=0)  # (H-1, W-1) 
        return torch.sum(torch.sqrt(grad_sq + epsilon**2))
    
    def l2_l1_sparse_2d(x):
        return torch.sum(torch.abs(x))
    
    
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
        xdisc_opt, fx, gx, status = optm.vmlmb(fg, x_disc_0, verb=1, lower=0, maxiter=100000, observer=None)
        return xdisc_opt, fx, gx, status
    
    # ------------------------------------------------------------
    # 5. Scheme over regularization parameters
    # ------------------------------------------------------------
    
    ## Optimization scheme
    exomild = ExoMILD(**cfg_model).to(device)
    exomild.load_state_dict(model_state)
    s1 = 1; s2 = 5 # number of values for mu_smooth and mu_sparse
    n_smooth = 7; n_sparse = 6 # starting exponents for mu_smooth and mu_sparse
    x_disk_store = np.empty((s1,s2), dtype=object)    
    for k in range(s1):
        for j in range(s2):
            xdisc_0 = np.zeros((C, H, W))
            (xdisc_opt, fx, gx, status) = run_bfgs(xdisc_0, y, 10**(k+n_smooth), 10**(j+n_sparse), exomild)
            x_disk_store[k,j] = xdisc_opt
            
    y_numpy = y.detach().cpu().numpy()
    np.savez(ROOT / f"results/results_realdata.npz", y=y_numpy, x=x_disk_store, n_smooth=n_smooth, n_sparse=n_sparse)


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