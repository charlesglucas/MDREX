"""Noise model diagnostic: are the two IRDIS channels independent samples, as assumed by MD-REX?

MD-REX (ExoMILD terms, dim_samples "t l") and its MC-SURE metric treat the (frame, channel) pairs of a patch as
independent samples of one spatial covariance, after wavelength alignment and per-channel temporal centering. REXPACO
ASDI (Flasseur et al. 2024) also models the covariance between channels (C_spat x C_spec). On the nuisance alone (data
without injected disk), this script measures, per 8 x 8 patch:
  - rho_raw: correlation between the two channels of the centered residual (same pixel, same frame);
  - rho_white: the same after whitening by the MD-REX shrinkage covariance (both channels pooled, as in MC-SURE): if
    the model held, the whitened residual would be N(0, I) with rho_white = 0;
  - std / excess kurtosis of the whitened residual (1 / 0 for the model);
  - 1 + rho_white: factor by which the information on a signal common to both channels (grey disk) is overcounted
    when the channels are taken as independent.
Results by separation (pixels) printed and saved to results/diagnostics/noise_channel_corr_<star>.npz.
  python scripts/run_results/noise_channel_corr.py [--star HIP_72192/2015-06-11]
"""
import argparse, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/plot_results"))
import numpy as np, torch
from inference.inference import load_folder
import disk_rec_sure_fixedW as M

p = argparse.ArgumentParser()
p.add_argument("--star", default="HIP_72192/2015-06-11")
args = p.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inp = load_folder(path_folder=ROOT / "data/nuisances" / args.star / "IRDIS/data", use_centered=False,
                  channel_sortframes=0, channel_idx=None)
y = torch.tensor(inp["y"].astype(np.float32), device=dev).unsqueeze(0)          # (1, C, T, H, W)
C, T, H = y.shape[1], y.shape[2], y.shape[3]
lbda = torch.tensor(inp["lbdas"].astype(np.float32), device=dev).unsqueeze(0)
rot = torch.tensor(inp["rot"].astype(np.float32), device=dev).expand(C, -1)
psf = torch.zeros((C, 1, 13, 13), device=dev); mask = torch.ones((C, H, H), device=dev)   # unused by to_patches
_, to_patches, centered = M.load_forward_model(dev, rot=rot, psf_crop=psf, mask=mask, lbda=lbda)
print(f"[data] {args.star}: lambda = {lbda.flatten().tolist()}, C = {C}, T = {T}", flush=True)

with torch.no_grad():
    pc = centered(to_patches(y))                       # (L, C*T, 64), channel-major samples
    L = pc.shape[0]
    Cinv = M.shrinkage_cov_inv(pc)                     # MD-REX metric (channels pooled)
    Lc = torch.linalg.cholesky(Cinv)                   # Cinv = Lc Lc^T -> whitened w = Lc^T r
    w = torch.einsum("lfg,ltf->ltg", Lc, pc)           # (L, C*T, 64)
    out = {}
    for name, z in (("raw", pc), ("white", w)):
        z = z.view(L, C, T, -1).double()
        a, b = z[:, 0], z[:, 1]
        out["rho_" + name] = ((a * b).sum((1, 2)) / torch.sqrt((a * a).sum((1, 2)) * (b * b).sum((1, 2)))).cpu().numpy()
    wz = w.double()
    out["std_white"] = wz.std(dim=(1, 2)).cpu().numpy()
    out["kurt_white"] = (((wz - wz.mean((1, 2), keepdim=True)) ** 4).mean((1, 2)) / wz.var((1, 2)) ** 2 - 3).cpu().numpy()
n = int(round(np.sqrt(L))); ps = H // n
ii, jj = np.divmod(np.arange(L), n)
sep = np.hypot(ps * ii + (ps - 1) / 2 - H / 2, ps * jj + (ps - 1) / 2 - H / 2)
out["sep"] = sep
bins = [0, 20, 40, 60, 80, 100, 130, 200]
print(f"{'sep (px)':>10} {'patches':>8} {'rho raw':>8} {'rho white':>10} {'1+rho':>6} {'std w':>6} {'kurt w':>7}")
for lo, hi in zip(bins[:-1], bins[1:]):
    s = (sep >= lo) & (sep < hi)
    if s.any():
        rw = np.median(out["rho_white"][s])
        print(f"{lo:>4}-{hi:<5} {s.sum():>8} {np.median(out['rho_raw'][s]):>8.3f} {rw:>10.3f} {1 + rw:>6.2f} "
              f"{np.median(out['std_white'][s]):>6.3f} {np.median(out['kurt_white'][s]):>7.2f}")
s = sep < 130
print(f"all (< 130 px): rho raw {np.median(out['rho_raw'][s]):.3f}, rho white {np.median(out['rho_white'][s]):.3f}", flush=True)
od = ROOT / "results/diagnostics"; od.mkdir(parents=True, exist_ok=True)
np.savez(od / f"noise_channel_corr_{args.star.split('/')[0]}.npz", **out)
