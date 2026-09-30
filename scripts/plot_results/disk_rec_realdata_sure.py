import sys, pathlib, re
import argparse

# Selection of (mu_smooth, mu_sparse) on real data with the MC-SURE and a pilot covariance.
# Reads results/realdata_sure/<result>/musmooth<mu>_musparse<mu>.npz written by run_results/realdata_sure.py
# (x(y), divergence div2) and y.npz, then:
#   1. MC-SURE with the covariance C^y estimated on y, fixed over the grid        -> mu_0
#   2. pilot covariance estimated on the residual y - A x(mu_0) (disk removed)
#   3. MC-SURE with the pilot covariance, fixed over the grid                     -> selected couple
# SURE = data term - N + div2, data term = sum_n sum_t || P_n (y - Theta) ||^2_{C_n^-1},
# Theta = A x + mean_t(y - A x) per channel (same definitions as disk_rec_sure_fixedW.py).
parser = argparse.ArgumentParser()
parser.add_argument("--result", type=str, default="RY_lup-2016-04-16", help="folder in results/realdata_sure")
parser.add_argument("--data", type=str, default=None, help="real data folder, None = deduced from --result")
parser.add_argument("--band", type=str, default=None, help="coronagraph band (h2_h3 or k1_k2), None = read from the results")
parser.add_argument("--results-root", type=str, default=None, help="default: results/realdata_sure")
parser.add_argument("--extra-probes", type=str, nargs="*", default=[],
                    help="other folders of results/realdata_sure (other seeds): the divergence is averaged over the probes")
args = parser.parse_args()

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
ROOT = pathlib.Path(__file__).resolve().parents[2]

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from utils.rotation import BatchRotationOperator

argv, sys.argv = sys.argv, sys.argv[:1]   # the imported plot scripts parse the command line at import
import disk_rec_realdata_grid as rgrid       # data_root, find_data, get_psf, fmt_mu, tex_mu, get_title
from disk_rec_sure_fixedW import shrinkage_cov_inv, mahalanobis
sys.argv = argv

torch.set_grad_enabled(False)


def load_results(folder):
    pattern = re.compile(r"musmooth(.+?)_musparse(.+)\.npz$")
    entries = {}
    for f in folder.glob("musmooth*_musparse*.npz"):
        if pattern.search(f.name):
            z = np.load(f, allow_pickle=True)
            entries[(float(z["mu_smooth"]), float(z["mu_sparse"]))] = z
    if not entries:
        raise FileNotFoundError(f"No result in {folder}")
    return entries


def forward_operators(data, band, wavelengths, T, C, H):
    """Forward model A (C, H, W) -> (1, C, T, H, W) and wavelength-aligned 8x8 patches, as in reconstruction/mdrex.py."""
    rot = np.asarray(fits.getdata(rgrid.data_root() / data
                                  / "ird_convert_recenter_dc5-IRD_SCIENCE_PARA_ROTATION_CUBE-rotnth.fits"),
                     dtype=np.float32).reshape(-1)
    assert rot.size == T, f"{rot.size} angles for {T} frames"
    rot = torch.tensor(rot).expand(C, -1)
    psf = rgrid.get_psf(data)
    m = fits.getdata(ROOT / f"data/coronograph/sphere_irdis_{band}_coronagraph_transmission_map.fits").astype(np.float32)
    d = (m.shape[1] - H) // 2
    mask = torch.tensor(m[:, d:d+H, d:d+H])
    op = BatchRotationOperator(device="cpu", in_size=H, out_size=H, mode="bicubic", zero_init=True)
    lbda = torch.tensor(np.asarray(wavelengths, dtype=np.float32)).view(1, C)

    def A(x):
        im = x.unsqueeze(0).expand(T, -1, -1, -1).permute(1, 0, 2, 3)
        im = op.forward(x=im, rot=rot).permute(1, 0, 2, 3)
        im = F.conv2d(im, weight=psf, padding="same", groups=C).permute(1, 0, 2, 3)
        return (im * mask.unsqueeze(1)).unsqueeze(0)

    def patches(r):  # (1, C, T, H, W) -> centered (per channel, over time) patches (L, C*T, 64)
        r = r - r.mean(dim=2, keepdim=True)
        coeff = (lbda / lbda.max()).view(C, 1, 1, 1)
        yyxx = torch.tensor(2 * np.mgrid[:H, :H] / (H - 1) - 1, dtype=torch.float32)[None].permute(0, 2, 3, 1)
        off = 1 / (H - 1)
        z = F.grid_sample(r[0], grid=off + (yyxx - off) * coeff, align_corners=True, mode="bicubic",
                          padding_mode="border")
        z = z.permute(0, 1, 3, 2).reshape(1, C * T, H, H)
        return F.unfold(z, kernel_size=8, stride=8).permute(0, 2, 1).reshape(-1, C * T, 64)

    return A, patches


def main():
    folder = pathlib.Path(args.results_root) if args.results_root else ROOT / "results" / "realdata_sure"
    folder = folder / args.result
    entries = load_results(folder)
    y = torch.tensor(np.load(folder / "y.npz")["y"])
    _, C, T, H, W = y.shape
    N = y.numel()
    first = next(iter(entries.values()))
    band = args.band or str(first["band"])
    data = args.data or rgrid.find_data(args.result)
    print(f"{args.result}: {len(entries)} couples, data {data}, band {band}")
    A, patches = forward_operators(data, band, first["wavelengths"], T, C, H)

    keys = sorted(entries)
    residual_patches = {}
    for k in keys:
        residual_patches[k] = patches(y - A(torch.tensor(entries[k]["x"], dtype=torch.float32)))
    div2 = {k: [float(entries[k]["div2"])] for k in keys}
    for extra in args.extra_probes:  # other Monte Carlo probes (seeds): average of the divergences
        other = load_results(folder.parent / extra)
        for k in keys:
            if k in other:
                div2[k].append(float(other[k]["div2"]))
    if args.extra_probes:
        print("2 * divergence per probe (mean, std):")
        for k in keys:
            print(f"  ({k[0]:.0e}, {k[1]:.0e}) " + "  ".join(f"{v:.4e}" for v in div2[k])
                  + f"   mean {np.mean(div2[k]):.4e}  std {np.std(div2[k]):.2e}")
    div2 = {k: float(np.mean(v)) for k, v in div2.items()}

    def sure(C_inv):
        return {k: mahalanobis(residual_patches[k], C_inv) - N + div2[k] for k in keys}

    # 1. covariance C^y estimated on y
    sure_y = sure(shrinkage_cov_inv(patches(y)))
    mu0 = min(sure_y, key=sure_y.get)
    # 2.-3. pilot covariance estimated on the residual of the reconstruction mu_0
    sure_p = sure(shrinkage_cov_inv(residual_patches[mu0]))
    best = min(sure_p, key=sure_p.get)
    print(f"MC-SURE with C^y:     argmin mu_0 = ({mu0[0]:.0e}, {mu0[1]:.0e})")
    print(f"MC-SURE with pilot C: argmin      = ({best[0]:.0e}, {best[1]:.0e})")
    print(f"{'couple':>16s} {'2 div':>12s} {'SURE C^y - min':>16s} {'SURE pilot - min':>18s}")
    my, mp = min(sure_y.values()), min(sure_p.values())
    for k in keys:
        print(f"  ({k[0]:.0e}, {k[1]:.0e}) {div2[k]:12.4e} {sure_y[k] - my:16.4e} {sure_p[k] - mp:18.4e}")

    # grids of SURE - min (+1 for the log scale), argmin marked
    sms = sorted({k[0] for k in keys}); sps = sorted({k[1] for k in keys})
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, S, am, title in [(axs[0], sure_y, mu0, r"MC-SURE, $\widehat{\mathbf{C}}^{\mathbf{y}}$"),
                             (axs[1], sure_p, best, r"MC-SURE, pilot covariance")]:
        G = np.full((len(sms), len(sps)), np.nan)
        m = min(S.values())
        for k, v in S.items():
            G[sms.index(k[0]), sps.index(k[1])] = v - m + 1
        im = ax.imshow(G, cmap="RdBu_r", norm=LogNorm())
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.plot(sps.index(am[1]), sms.index(am[0]), "m+", ms=14, mew=2)
        ax.set_xticks(range(len(sps))); ax.set_xticklabels([f"${rgrid.tex_mu(v)}$" for v in sps])
        ax.set_yticks(range(len(sms))); ax.set_yticklabels([f"${rgrid.tex_mu(v)}$" for v in sms])
        ax.set_xlabel(r"$\mu_{\rm sparse}$"); ax.set_ylabel(r"$\mu_{\rm smooth}$"); ax.set_title(title)
    fig.suptitle(rgrid.get_title(args.result))
    fig.tight_layout()
    out = ROOT / "figures" / f"{args.result}_sure_grid.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    print(f"RGB figure of the selected couple: python scripts/plot_results/disk_rec_realdata_grid.py "
          f"--result {args.result} --mu-smooth {rgrid.fmt_mu(best[0])} --mu-sparse {rgrid.fmt_mu(best[1])}")


if __name__ == "__main__":
    main()
