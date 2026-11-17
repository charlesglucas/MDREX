"""Global amplitude debiasing of the grid reconstructions: x_s = s x_hat, with s fitted on the data,
  s = <Q A x_hat, Q y>_P / <Q A x_hat, Q A x_hat>_P   (P = C_y^-1, the metric of MC-SURE, or identity),
i.e. the least-squares scale of the reconstructed disk in the data space (Q: patches + temporal centering, so the
static nuisance and the axisymmetric part drop out). PSNR (whole image, support at 50 % / 4 % of the maximum) of x_hat
and of s x_hat on a K12f grid (angle 0, HIP 72192).
  python scripts/run_results/amplitude_debias_test.py spiral 5e-6 [medium_ellipse 5e-6 ...]
"""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/plot_results"))
import numpy as np, torch
from astropy.io import fits
from inference.inference import load_folder
import disk_rec_sure_fixedW as M
torch.backends.mkldnn.enabled = False
dev = torch.device("cpu")
fm, to_patches, centered = M.load_forward_model(dev)
inp = load_folder(path_folder=ROOT / "data/nuisances/HIP_72192/2015-06-11/IRDIS/data", use_centered=False,
                  channel_sortframes=0, channel_idx=None)
y0 = torch.tensor(inp["y"].astype(np.float32)).unsqueeze(0)
C = y0.shape[1]
Q = lambda v: centered(to_patches(v)).double()
args = sys.argv[1:]
for shape, flux in zip(args[::2], args[1::2]):
    fl = float(flux); ft = f"{fl:.0e}".replace("e-0", "em")
    g = fl * fits.getdata(ROOT / f"data/synthetic_disks/{shape}/hid_fake_disk_image_{shape}_0degrees.fits").astype(np.float32)[513-128:513+128, 513-128:513+128]
    with torch.no_grad():
        y = fm(torch.tensor(g).unsqueeze(0).repeat(C, 1, 1)) + y0
        z = Q(y)
        Cinv = M.shrinkage_cov_inv(centered(to_patches(y))).double()
    P = {"C^-1": lambda u: torch.einsum("lfg,ltg->ltf", Cinv, u), "I": lambda u: u}
    gd = g.astype(np.float64)
    ps = lambda x, s=None: -20 * np.log10(np.linalg.norm((x - gd) if s is None else (x - gd)[s]) /
                                          np.linalg.norm(gd if s is None else gd[s]))
    s50, s4 = gd > 0.5 * gd.max(), gd > 0.04 * gd.max()
    G = ROOT / f"results/sugar/{shape}_alpha{ft}_angle0_K12f_grid"
    rows = np.load(G / "grid_0.npz")["rows"]
    print(f"\n== {shape} {flux}: couple | mse (data) | PSNR whole / 50 % / 4 % | flux 4 % | s (C^-1), s (I) | "
          f"PSNR of s x_hat (C^-1): whole / 50 % | (I): whole / 50 %", flush=True)
    out = []
    for r in rows:
        x = np.load(G / f"x_{r[0]:.3f}_{r[1]:.3f}.npz")["x"].astype(np.float64)
        with torch.no_grad():
            qa = Q(fm(torch.tensor(x, dtype=torch.float32)))
            sc = {k: float((qa * Pk(z)).sum() / (qa * Pk(qa)).sum()) for k, Pk in P.items()}
        xm = x.mean(0)
        out.append((r[0], r[1], r[2], ps(xm), ps(xm, s50), sc["C^-1"], ps(sc["C^-1"] * xm), ps(sc["C^-1"] * xm, s50),
                    sc["I"], ps(sc["I"] * xm), ps(sc["I"] * xm, s50)))
        o = out[-1]
        print(f"  ({r[0]:.2f}, {r[1]:.2f}) | {r[2]:.3e} | {o[3]:5.2f} / {o[4]:5.2f} / {ps(xm, s4):5.2f} | "
              f"{xm[s4].sum() / gd[s4].sum():.2f} | {o[5]:.2f}, {o[8]:.2f} | {o[6]:5.2f} / {o[7]:5.2f} | {o[9]:5.2f} / {o[10]:5.2f}", flush=True)
    o = np.array(out)
    for name, col in (("min data MSE", np.argmin(o[:, 2])), ("max PSNR", np.argmax(o[:, 3])),
                      ("max PSNR of s x_hat (C^-1)", np.argmax(o[:, 6])), ("max PSNR of s x_hat (I)", np.argmax(o[:, 9]))):
        print(f"  {name}: ({o[col, 0]:.2f}, {o[col, 1]:.2f})", flush=True)
