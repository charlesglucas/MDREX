"""Does the patch size of the data-space metric change the couple chosen by the MSE?
Oracle Mahalanobis MSE sum_l sum_t d^T C_l^-1 d, d = centered(patches_p(A (x_hat - x_gt))), C_l = shrinkage covariance
of the nuisance (as C_inv_gt of syntheticdata_sugar_iter.py), for patch sizes p = 8 (MC-SURE, the grids), 13 (REXPACO), 16, 32,
on the K12f grids (angle 0, HIP 72192). Argmin compared with the PSNR maximum. Where the GSURE perturbations exist
(<case>_K12f_gsure/xeps_*.npz, probe delta of --gsure perturb), also MC-SURE with the metric C_y^-1 of patch size p:
  SURE_p = ||z - Q_p A x_hat||^2_{C_y^-1} + (2 / eps) (Q_p delta)^T C_y^-1 Q_p A (x_hat(y + eps delta) - x_hat(y)) + const.
With p = 13, the 19 x 19 patches cover the central 247 x 247 pixels.
  python scripts/run_results/metric_patch_test.py spiral 5e-6 [medium_ellipse 5e-6 ...]
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/plot_results"))
import numpy as np, torch, torch.nn.functional as F
from astropy.io import fits
from inference.inference import load_folder
import disk_rec_sure_fixedW as M
torch.backends.mkldnn.enabled = False
fm, _, _ = M.load_forward_model(torch.device("cpu"))
inp = load_folder(path_folder=ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data", use_centered=False,
                  channel_sortframes=0, channel_idx=None)
y0 = torch.tensor(inp["y"].astype(np.float32)).unsqueeze(0)
C, T, H = y0.shape[1], y0.shape[2], y0.shape[3]
lbda = torch.tensor([[2.110, 2.251]])

def patches(x, p):   # (1, C, T, H, W) -> centered (L, C*T, p*p), wavelength-aligned, as M.load_forward_model
    coeff = (lbda / lbda.max()).view(C, 1, 1, 1)
    yyxx = torch.tensor(2 * np.mgrid[:H, :H] / (H - 1) - 1, dtype=torch.float32)[None].permute(0, 2, 3, 1)
    off = 1 / (H - 1)
    x = F.grid_sample(x[0], grid=off + (yyxx - off) * coeff, align_corners=True, mode="bicubic", padding_mode="border")
    x = x.permute(0, 1, 3, 2).reshape(1, C * T, H, H)
    q = F.unfold(x, kernel_size=p, stride=p).permute(0, 2, 1).reshape(-1, C, T, p * p)
    return (q - q.mean(2, keepdim=True)).reshape(-1, C * T, p * p).double()

def cov_inv(pc):     # shrinkage covariance (M.shrinkage_cov_inv), float64
    L, Tn, f = pc.shape
    S = pc.transpose(1, 2) @ pc / Tn
    d = torch.diagonal(S, dim1=-2, dim2=-1)
    tr_S2, tr2_S, tr_SS = (d ** 2).sum(-1), d.sum(-1) ** 2, (S * S).sum((-2, -1))
    rho = torch.clip((tr_SS + tr2_S - 2 * tr_S2) / ((Tn + 1) * (tr_SS - tr_S2)), 0, 1).view(L, 1, 1)
    return torch.linalg.inv((1 - rho) * S + rho * torch.diag_embed(d)), float(rho.mean())

PS = [int(v) for v in __import__('os').environ.get('PS', '8 13 16 32').split()]
with torch.no_grad():
    CI = {p: cov_inv(patches(y0, p)) for p in PS}
for p in PS:
    print(f"patch {p}: mean shrinkage rho {CI[p][1]:.3f}", flush=True)
args = sys.argv[1:]
for shape, flux in zip(args[::2], args[1::2]):
    fl = float(flux); ft = f"{fl:.0e}".replace("e-0", "em")
    g = fl * fits.getdata(ROOT / f"data/synthetic_disks/{shape}/hid_fake_disk_image_{shape}_0degrees.fits").astype(np.float32)[513-128:513+128, 513-128:513+128]
    G = ROOT / f"results/sugar/{shape}_alpha{ft}_angle0_K12f_grid"
    rows = np.load(G / "grid_0.npz")["rows"]
    res = []
    for r in rows:
        x = np.load(G / f"x_{r[0]:.3f}_{r[1]:.3f}.npz")["x"].astype(np.float32)
        with torch.no_grad():
            e = fm(torch.tensor(x - g[None]))
            res.append([float(torch.einsum("ltf,lfg,ltg->", patches(e, p), CI[p][0], patches(e, p))) for p in PS])
    res = np.array(res); ip = int(np.argmax(rows[:, 3]))
    GS = ROOT / f"results/sugar/{shape}_alpha{ft}_angle0_K12f_gsure"
    sure = None
    if all((GS / f"xeps_{r[0]:.3f}_{r[1]:.3f}.npz").exists() for r in rows):
        with torch.no_grad():
            xg = torch.tensor(g).unsqueeze(0).repeat(C, 1, 1)
            y = fm(xg) + y0
            gen = torch.Generator(device="cpu").manual_seed(43)          # probe of --gsure (seed + 1)
            r0 = (y - y.mean(dim=2, keepdim=True))[0]
            mix = torch.randn((C, T, T), generator=gen, dtype=torch.float32) / np.sqrt(T)
            dlt = torch.einsum("cts,cshw->cthw", mix, r0).unsqueeze(0)
            sure, orc_y = [], []
            Cy = {p: cov_inv(patches(y, p))[0] for p in PS}
            zq = {p: patches(y, p) for p in PS}; dq = {p: patches(dlt, p) for p in PS}
            for r in rows:
                xh = torch.tensor(np.load(G / f"x_{r[0]:.3f}_{r[1]:.3f}.npz")["x"].astype(np.float32))
                xe = torch.tensor(np.load(GS / f"xeps_{r[0]:.3f}_{r[1]:.3f}.npz")["x"].astype(np.float32))
                ah, ae = fm(xh), fm(xe)
                v, o = [], []
                eq = fm(xh - xg)
                for p in PS:
                    rq, aq = zq[p] - patches(ah, p), patches(ae, p) - patches(ah, p)
                    v.append(float(torch.einsum("ltf,lfg,ltg->", rq, Cy[p], rq) +
                                   (2 / 0.1) * torch.einsum("ltf,lfg,ltg->", dq[p], Cy[p], aq)))
                    e_ = patches(eq, p)
                    o.append(float(torch.einsum("ltf,lfg,ltg->", e_, Cy[p], e_)))   # oracle MSE in the SAME metric C_y
                sure.append(v); orc_y.append(o)
        sure, orc_y = np.array(sure), np.array(orc_y)
    print(f"== {shape} {flux}: max PSNR at ({rows[ip, 0]:.2f}, {rows[ip, 1]:.2f}) {rows[ip, 3]:.2f} dB", flush=True)
    for k, p in enumerate(PS):
        i = int(np.argmin(res[:, k]))
        print(f"   patch {p:2d}: min Mahalanobis MSE at ({rows[i, 0]:.2f}, {rows[i, 1]:.2f}) PSNR {rows[i, 3]:.2f} dB; "
              f"MSE(PSNR max) / MSE(min) = {res[ip, k] / res[i, k]:.3f}"
              + ("" if sure is None else f" | min MC-SURE at ({rows[int(np.argmin(sure[:, k])), 0]:.2f}, "
                 f"{rows[int(np.argmin(sure[:, k])), 1]:.2f}) PSNR {rows[int(np.argmin(sure[:, k])), 3]:.2f} dB, Spearman(SURE, MSE) "
                 f"{np.corrcoef(np.argsort(np.argsort(sure[:, k])), np.argsort(np.argsort(res[:, k])))[0, 1]:.2f}; same metric C_y: min "
                 f"oracle at ({rows[int(np.argmin(orc_y[:, k])), 0]:.2f}, {rows[int(np.argmin(orc_y[:, k])), 1]:.2f}) PSNR "
                 f"{rows[int(np.argmin(orc_y[:, k])), 3]:.2f} dB, Spearman(SURE, oracle C_y) "
                 f"{np.corrcoef(np.argsort(np.argsort(sure[:, k])), np.argsort(np.argsort(orc_y[:, k])))[0, 1]:.2f}"), flush=True)
