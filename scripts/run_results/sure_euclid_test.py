"""SURE of the Euclidean data-space MSE E||Q A (x_hat - x)||^2 (no metric C^-1), from the GSURE perturbations.

z = Q y = Q A x + w, Q = centered(to_patches(.)), w = Q n, n ~ N(0, Sigma) (Sigma = S_c x I_T, the probe delta of
--gsure perturb: random mixtures of the centered frames, seed + 1). For a metric P (identity or C^-1):
  SURE_P = ||z - Q A x_hat||_P^2 - tr(P Q Sigma Q^T) + (2 / eps) (Q delta)^T P Q A (x_hat(y + eps delta) - x_hat(y)),
an unbiased estimate (up to O(eps) and the Monte Carlo error) of E||Q A (x_hat - x)||_P^2 (the trace is constant in mu
and dropped). Compared on the grid with the oracle ||Q A (x_hat - x)||_P^2 and with the PSNR.
  python scripts/run_results/sure_euclid_test.py spiral 5e-6
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/plot_results"))
import numpy as np, torch
from astropy.io import fits
from inference.inference import load_folder
import disk_rec_sure_fixedW as M

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.mkldnn.enabled = False   # CPU conv2d of the front-end fails with oneDNN
shape, flux = sys.argv[1], float(sys.argv[2]); ft = f"{flux:.0e}".replace("e-0", "em")
angle = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # injection angle of the disk (another noise realization)
seed, eps = 42, 0.1
fm, to_patches, centered = M.load_forward_model(dev)    # HIP 72192, K12
inp = load_folder(path_folder=ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data", use_centered=False,
                  channel_sortframes=0, channel_idx=None)
y0 = torch.tensor(inp["y"].astype(np.float32), device=dev).unsqueeze(0)
C, T = y0.shape[1], y0.shape[2]
g = flux * fits.getdata(ROOT / f"data/synthetic_disks/{shape}/hid_fake_disk_image_{shape}_{angle}degrees.fits").astype(np.float32)[513-128:513+128, 513-128:513+128]
xg = torch.tensor(g, device=dev).unsqueeze(0).repeat(C, 1, 1)
Q = lambda v: centered(to_patches(v)).double()
with torch.no_grad():
    y = fm(xg) + y0
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)               # same probe as --gsure
    r0 = (y - y.mean(dim=2, keepdim=True))[0]
    mix = (torch.randn((C, T, T), generator=gen, dtype=torch.float32) / np.sqrt(T)).to(dev)
    dlt = torch.einsum("cts,cshw->cthw", mix, r0).unsqueeze(0)
    z, Qd, QAx = Q(y), Q(dlt), Q(fm(xg))
    w = Q(y0)                                                               # true noise (synthetic data)
    Cinv = M.shrinkage_cov_inv(centered(to_patches(y))).double()           # metric C_y^-1 of MC-SURE
P = {"euclid": lambda u: u, "mahal": lambda u: torch.einsum("lfg,ltg->ltf", Cinv, u)}
# intermediate metrics (C + nu I)^-1, nu = kappa * tr(C_l) / 64: from Mahalanobis (kappa -> 0) to Euclidean (kappa -> inf)
with torch.no_grad():
    lam, U = torch.linalg.eigh(torch.linalg.inv(Cinv))
for kap in [0.01, 0.1, 1.0, 10.0]:
    nu = kap * lam.mean(-1, keepdim=True)
    Pm = (U / (lam + nu).unsqueeze(-2)) @ U.transpose(-1, -2) * (lam.mean(-1) + nu.squeeze(-1)).view(-1, 1, 1)
    P[f"k{kap:g}"] = (lambda M_: (lambda u: torch.einsum("lfg,ltg->ltf", M_, u)))(Pm)

G = ROOT / f"results/sugar/{shape}_alpha{ft}_angle{angle}_K12f_grid"
GS = ROOT / f"results/sugar/{shape}_alpha{ft}_angle{angle}_K12f_gsure"
rows = np.load(G / "grid_0.npz")["rows"]
res = {k: [] for k in P}
cpl = []
for r in rows:
    f, fe = G / f"x_{r[0]:.3f}_{r[1]:.3f}.npz", GS / f"xeps_{r[0]:.3f}_{r[1]:.3f}.npz"
    if not fe.exists():
        continue
    with torch.no_grad():
        QAxh = Q(fm(torch.tensor(np.load(f)["x"], device=dev)))
        QAxe = Q(fm(torch.tensor(np.load(fe)["x"], device=dev)))
        for k, Pk in P.items():
            res_ = z - QAxh
            sure = float((res_ * Pk(res_)).sum() + (2 / eps) * (Qd * Pk(QAxe - QAxh)).sum())
            e = QAxh - QAx
            # mu-dependent part of SURE - oracle = 2 div_MC - 2 w^T P Q A x_hat: Monte Carlo divergence vs its target
            res[k].append((sure, float((e * Pk(e)).sum()), float((2 / eps) * (Qd * Pk(QAxe - QAxh)).sum()),
                           float(2 * (w * Pk(QAxh)).sum()),
                           float((QAxh * Pk(QAx)).sum() / (QAx * Pk(QAx)).sum()), float(2 * (w * Pk(QAx)).sum())))
    cpl.append((r[0], r[1], r[3]))
cpl = np.array(cpl)
print(f"== {shape} {flux} angle {angle}: {len(cpl)} couples, max PSNR at ({cpl[np.argmax(cpl[:, 2]), 0]:.2f}, "
      f"{cpl[np.argmax(cpl[:, 2]), 1]:.2f}) {cpl[:, 2].max():.2f} dB")
for k in P:
    a = np.array(res[k]); io, isu = np.argmin(a[:, 1]), np.argmin(a[:, 0])
    off = a[:, 0] - a[:, 1]
    print(f"  {k:6s}: min oracle at ({cpl[io, 0]:.2f}, {cpl[io, 1]:.2f}) PSNR {cpl[io, 2]:.2f} | min SURE at "
          f"({cpl[isu, 0]:.2f}, {cpl[isu, 1]:.2f}) PSNR {cpl[isu, 2]:.2f} | SURE - oracle: spread {np.ptp(off):.3e} "
          f"vs oracle spread {np.ptp(a[:, 1]):.3e}; Spearman(SURE, oracle) "
          f"{np.corrcoef(np.argsort(np.argsort(a[:, 0])), np.argsort(np.argsort(a[:, 1])))[0, 1]:.2f}", flush=True)
    print(f"          2 div_MC spread {np.ptp(a[:, 2]):.3e}, 2 w^T P QAx_hat spread {np.ptp(a[:, 3]):.3e}, their difference "
          f"spread {np.ptp(a[:, 2] - a[:, 3]):.3e}, correlation {np.corrcoef(a[:, 2], a[:, 3])[0, 1]:.2f}", flush=True)
    pred = a[:, 4] * a[:, 5]   # c(mu) * 2 w^T P Q A x: part of the noise term carried by the reconstructed disk itself
    print(f"          c(mu) = <QAx_hat, QAx>_P / |QAx|_P^2 in [{a[:, 4].min():.2f}, {a[:, 4].max():.2f}]; 2 w^T P QAx = "
          f"{a[0, 5]:.3e}; c(mu) * 2 w^T P QAx: spread {np.ptp(pred):.3e}, share of the variance of the noise term "
          f"{1 - np.var(a[:, 3] - pred) / np.var(a[:, 3]):.2f}", flush=True)
