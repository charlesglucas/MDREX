"""Temporal correlation of the nuisance after whitening by the MD-REX patch covariance.
C (one 64 x 64 matrix per 8 x 8 patch, channels pooled) models the correlation between the pixels of a patch at a
given frame; frames are assumed independent (as in REXPACO, Cov = 0 for t1 != t2). This script measures, on the
nuisance alone (HIP 72192, K12, no disk), the correlation between whitened frames t and t + k of the same patch and
channel, by separation, and the effective number of independent frames T_eff = T / (1 + 2 sum_k rho_k).
  python scripts/run_results/noise_temporal_corr.py
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/plot_results"))
import numpy as np, torch
from inference.inference import load_folder
import disk_rec_sure_fixedW as M
torch.backends.mkldnn.enabled = False
dev = torch.device("cpu")
fm, to_patches, centered = M.load_forward_model(dev)
inp = load_folder(path_folder=ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data", use_centered=False,
                  channel_sortframes=0, channel_idx=None)
y0 = torch.tensor(inp["y"].astype(np.float32)).unsqueeze(0)
C, T, H = y0.shape[1], y0.shape[2], y0.shape[3]
with torch.no_grad():
    pc = centered(to_patches(y0)).double()                       # (L, C*T, 64)
    L = pc.shape[0]
    Lc = torch.linalg.cholesky(M.shrinkage_cov_inv(pc.float()).double())
    w = torch.einsum("lfg,ltf->ltg", Lc, pc).view(L, C, T, -1)   # whitened, (L, C, T, 64)
    raw = pc.view(L, C, T, -1)
    lags = [1, 2, 4, 8, 16, 32]
    def acf(z):
        z = z - z.mean(2, keepdim=True)
        den = (z * z).sum((1, 2, 3))
        return {k: ((z[:, :, k:] * z[:, :, :-k]).sum((1, 2, 3)) / den * T / (T - k)).numpy() for k in range(1, T // 2)}
    A_w, A_r = acf(w), acf(raw)
n = int(round(np.sqrt(L))); ps = H // n
ii, jj = np.divmod(np.arange(L), n)
sep = np.hypot(ps * ii + (ps - 1) / 2 - H / 2, ps * jj + (ps - 1) / 2 - H / 2)
print("separation | correlation of whitened frames t, t+k for k = " + " ".join(map(str, lags)) + " | raw k=1 | T_eff (whitened)")
for lo, hi in zip([0, 20, 40, 60, 80, 100, 130], [20, 40, 60, 80, 100, 130, 200]):
    s = (sep >= lo) & (sep < hi)
    rho = np.array([np.median(A_w[k][s]) for k in range(1, T // 2)])
    teff = T / (1 + 2 * np.sum(np.clip(rho[:16], 0, None)))
    print(f"{lo:>4}-{hi:<4} px | " + " ".join(f"{np.median(A_w[k][s]):5.2f}" for k in lags) +
          f" | {np.median(A_r[1][s]):5.2f} | {teff:5.1f} / {T}", flush=True)
