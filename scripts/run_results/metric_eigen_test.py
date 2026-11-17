"""Why does the data-space MSE (Mahalanobis, metric C^-1 of the nuisance) prefer a stronger regularization than the PSNR?
Hypothesis: the flux deficit of the strongly regularized reconstructions lies along the high-variance (speckle-like)
eigenvectors of the patch covariances C_l, where it costs little in the metric C^-1.

On a K12f grid (x_hat of results/sugar/<case>_K12f_grid, angle 0, HIP 72192), for every couple:
  d = centered(to_patches(A (x_hat - x_gt)))  (data-space error, as in the MSE of the grids),
  C_l = shrinkage covariance of the nuisance (y - A x_gt, as C_inv_gt of syntheticdata_sugar_iter.py) = U diag(lam) U^T,
  projections p = U^T d, sorted by decreasing variance lam within each patch.
Printed: argmin over the grid of (a) the Mahalanobis MSE sum p^2 / lam, (b) the Euclidean data-space MSE sum p^2,
(c) the object-space MSE ||mean_c x_hat - x_gt||^2 (PSNR); and for the MSE-min and PSNR-max couples, the shares of the
Euclidean error and of the Mahalanobis cost in the eigen-directions of highest variance (top 1, 4, 16 of 64).
  python scripts/run_results/metric_eigen_test.py spiral 5e-6 [spiral 1e-5 medium_ellipse 5e-6 ...]
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/plot_results"))
import numpy as np, torch
from astropy.io import fits
from inference.inference import load_folder
import disk_rec_sure_fixedW as M

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fm, to_patches, centered = M.load_forward_model(dev)    # HIP 72192, K12
inp = load_folder(path_folder=ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data", use_centered=False,
                  channel_sortframes=0, channel_idx=None)
y0 = torch.tensor(inp["y"].astype(np.float32), device=dev).unsqueeze(0)

with torch.no_grad():
    pc = centered(to_patches(y0)).double()                       # nuisance only (= y - A x_gt)
    L, T, f = pc.shape
    S = pc.transpose(1, 2) @ pc / T
    dS = torch.diagonal(S, dim1=-2, dim2=-1)
    tr_S2, tr2_S, tr_SS = (dS ** 2).sum(-1), dS.sum(-1) ** 2, (S * S).sum((-2, -1))
    rho = torch.clip((tr_SS + tr2_S - 2 * tr_S2) / ((T + 1) * (tr_SS - tr_S2)), 0, 1).view(L, 1, 1)
    Ch = (1 - rho) * S + rho * torch.diag_embed(dS)              # as M.shrinkage_cov_inv
    lam, U = torch.linalg.eigh(Ch)                               # ascending
    lam, U = lam.flip(-1), U.flip(-1)                            # decreasing variance

def spectrum(e):   # e: (C, H, W) object-space error -> (Euclidean, Mahalanobis) energy per eigen rank (64,)
    with torch.no_grad():
        d = centered(to_patches(fm(torch.tensor(e, dtype=torch.float32, device=dev)))).double()
        p2 = (torch.einsum("lfk,ltf->ltk", U, d) ** 2).sum(1)    # (L, 64)
        return p2.sum(0).cpu().numpy(), (p2 / lam).sum(0).cpu().numpy()

args = sys.argv[1:]
for shape, flux in zip(args[::2], args[1::2]):
    fl = float(flux); ft = f"{fl:.0e}".replace("e-0", "em")
    g = fl * fits.getdata(ROOT / f"data/synthetic_disks/{shape}/hid_fake_disk_image_{shape}_0degrees.fits").astype(np.float64)[513-128:513+128, 513-128:513+128]
    G = ROOT / f"results/sugar/{shape}_alpha{ft}_angle0_K12f_grid"
    rows = np.load(G / "grid_0.npz")["rows"]
    res = []
    for r in rows:
        x = np.load(G / f"x_{r[0]:.3f}_{r[1]:.3f}.npz")["x"].astype(np.float64)
        eu, ma = spectrum(x - g[None])
        res.append((r[0], r[1], ma.sum(), eu.sum(), np.sum((x.mean(0) - g) ** 2), r[3], eu, ma))
    lab = lambda i: f"({res[i][0]:.2f}, {res[i][1]:.2f}) PSNR {res[i][5]:.2f}"
    i_ma, i_eu, i_x = (int(np.argmin([q[k] for q in res])) for k in (2, 3, 4))
    print(f"\n== {shape} {flux} ({len(res)} couples): min Mahalanobis {lab(i_ma)} | min Euclidean data-space {lab(i_eu)} | "
          f"min object-space (max PSNR) {lab(i_x)}", flush=True)
    eu_s, ma_s = spectrum(g[None].repeat(2, 0))
    print(f"   disk A x itself: Euclidean share in top 1/4/16 directions {eu_s[:1].sum()/eu_s.sum():.2f} / "
          f"{eu_s[:4].sum()/eu_s.sum():.2f} / {eu_s[:16].sum()/eu_s.sum():.2f}")
    for name, i in (("MSE-min ", i_ma), ("PSNR-max", i_x)):
        eu, ma = res[i][6], res[i][7]
        print(f"   {name} {lab(i)}: Euclid {eu.sum():.3e} (top 1/4/16: {eu[:1].sum()/eu.sum():.2f} / {eu[:4].sum()/eu.sum():.2f} / "
              f"{eu[:16].sum()/eu.sum():.2f}), Mahalanobis {ma.sum():.4e} (top 1/4/16: {ma[:1].sum()/ma.sum():.2f} / "
              f"{ma[:4].sum()/ma.sum():.2f} / {ma[:16].sum()/ma.sum():.2f}; low-variance 48: {ma[16:].sum():.4e})")
