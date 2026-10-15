"""Compact figure of the hyperparameter grids for one shape: rows MSE / MC-SURE / refined MC-SURE (row labels in grey
boxes on the left), columns = contrasts alpha (grey boxes on top); y ticks / label on the left column only, x ticks /
label on the last row only. Data from disk_rec_sure_fixedW.py (figures/grid_data_<shape>_<flux>_<tag>.npz) and
MC-SUGAR results (results/sugar/<shape>_alpha<flux>_angle0_gradS[_refined]/trajectory.npz, if finished).
Markers: MSE minimum (red x), MC-SURE minimum (purple +), MC-SUGAR (orange X), refined MC-SURE minimum (green circle),
refined MC-SUGAR (orange diamond): all on the MSE row, MSE / MC-SURE / MC-SUGAR on the MC-SURE row, MSE / refined on
the refined row.
Usage: python plot_grids_combined.py [shape] [tag]   (default medium_ellipse angle0_gradS)"""
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHAPE = sys.argv[1] if len(sys.argv) > 1 else "medium_ellipse"
TAG = sys.argv[2] if len(sys.argv) > 2 else "angle0_gradS"
FLUXES = [("1em6", r"$\alpha = 10^{-6}$"), ("5em6", r"$\alpha = 5\cdot 10^{-6}$"), ("1em5", r"$\alpha = 10^{-5}$")]
ROWS = [("mse", "MSE", {"sure", "sugar", "ref", "ref_sugar"}),
        ("sure", "MC-SURE", {"sure", "sugar"}),
        ("sure_ref", "Refined MC-SURE", {"ref", "ref_sugar"})]
BOX = dict(boxstyle="round,pad=0.35", facecolor="0.85", edgecolor="0.5")


def mu_label(v):  # 1e7 -> 10^7, 3e7 -> 3·10^7
    m, e = f"{v:.0e}".split("e")
    return f"$10^{{{int(e)}}}$" if m == "1" else f"${m}\\cdot 10^{{{int(e)}}}$"


def sci_colorbar(cbar, nbins=4, decimals=2):
    """Scientific notation with the common factor on top, at most nbins + 1 ticks, distinct labels."""
    vmin, vmax = cbar.mappable.get_clim()
    e = int(np.floor(np.log10(max(abs(vmin), abs(vmax)))))
    cbar.locator = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10])
    ticks = [t for t in cbar.locator.tick_values(vmin, vmax) if vmin <= t <= vmax]
    while len({f"{t / 10**e:.{decimals}f}" for t in ticks}) < len(ticks):
        decimals += 1
    cbar.formatter = FuncFormatter(lambda v, pos: f"{v / 10**e:.{decimals}f}".replace("-", "−"))
    cbar.update_ticks()
    cbar.ax.text(0.0, 1.02, rf"$\times 10^{{{e}}}$", transform=cbar.ax.transAxes, ha="left", va="bottom", fontsize=9)
    cbar.ax.tick_params(labelsize=9)


def sugar_mu(flux, refined):
    f = ROOT / "results" / "sugar" / f"{SHAPE}_alpha{flux}_angle0_gradS{'_refined' if refined else ''}" / "trajectory.npz"
    if not f.exists():
        return None
    tr = np.load(f, allow_pickle=True)
    return tr["mu_opt"] if "mu_opt" in tr.files else None


fig, axs = plt.subplots(3, 3, figsize=(10, 8.4), constrained_layout=True)
for c, (flux, alpha_label) in enumerate(FLUXES):
    d = np.load(ROOT / "figures" / f"grid_data_{SHAPE}_{flux}_{TAG}.npz")
    ms, mp = list(d["mu_smooth"]), list(d["mu_sparse"])
    to_j = lambda v: np.interp(np.log10(v), np.log10(mp), np.arange(len(mp)))
    to_k = lambda v: np.interp(np.log10(v), np.log10(ms), np.arange(len(ms)))
    sug, sug_ref = sugar_mu(flux, False), sugar_mu(flux, True)
    for r, (key, title, shown) in enumerate(ROWS):
        ax = axs[r, c]
        im = ax.imshow(d[key], cmap="RdBu_r")
        sci_colorbar(fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, aspect=15))
        ax.set_xticks(np.arange(len(mp))); ax.set_yticks(np.arange(len(ms)))
        if r == len(ROWS) - 1:
            ax.set_xticklabels([mu_label(v) for v in mp], fontsize=10, rotation=45, ha="right", rotation_mode="anchor")
            ax.set_xlabel(r"$\mu_{\rm sparse}$", fontsize=12)
        else:
            ax.set_xticklabels([])
        if c == 0:
            ax.set_yticklabels([mu_label(v) for v in ms], fontsize=10)
            ax.set_ylabel(r"$\mu_{\rm smooth}$", fontsize=12)
            ax.annotate(title, xy=(-0.55, 0.5), xycoords="axes fraction", rotation=90, ha="center", va="center",
                        fontsize=14, bbox=BOX)
        else:
            ax.set_yticklabels([])
        if r == 0:
            ax.annotate(alpha_label, xy=(0.5, 1.12), xycoords="axes fraction", ha="center", va="center", fontsize=14,
                        bbox=BOX)
        ax.plot(d["i_mse"][1], d["i_mse"][0], "rx", markersize=11, markeredgewidth=2)
        if "sure" in shown:
            ax.plot(d["i_sure"][1], d["i_sure"][0], "m+", markersize=11, markeredgewidth=2)
        if "ref" in shown and d["i_ref"][0] >= 0:
            ax.plot(d["i_ref"][1], d["i_ref"][0], "o", markersize=12, markerfacecolor="none",
                    markeredgecolor="limegreen", markeredgewidth=2.5)
        if "sugar" in shown and sug is not None:
            ax.plot(to_j(sug[1]), to_k(sug[0]), "X", color="orange", markersize=11, markeredgecolor="k", markeredgewidth=1)
        if "ref_sugar" in shown and sug_ref is not None:
            ax.plot(to_j(sug_ref[1]), to_k(sug_ref[0]), "D", color="orange", markersize=8, markeredgecolor="k",
                    markeredgewidth=1)
fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.0, wspace=0.0)
out = ROOT / "figures" / f"grids_{SHAPE}_combined_{TAG}.pdf"
fig.savefig(out)
print(out)
