"""Compact comparison figures REXPACO / MD-REX: the ground truth appears once on the left of each block, all the images
are divided by the contrast alpha (common colour scale "x alpha"), the contrasts are given in grey boxes on top of the
columns. Each block: row 1 = ground truth + reconstructions (REXPACO, MD-REX) for the 3 contrasts, row 2 = relative
errors |x_hat - x| / alpha. Data from disk_rec_comp.py (figures/comp_data_refinedSURE_gradS_mean.npz).
Figures:
  figures/comp_compact_medium_ellipse.pdf : ellipse, blocks x_hat and H x_hat (paper, Sect. 3)
  figures/comp_compact_spiral_circle.pdf  : spiral and circle, x_hat (appendix)
  figures/comp_compact_conv_spiral_circle.pdf : spiral and circle, H x_hat (appendix)"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = np.load(ROOT / "figures" / "comp_data_refinedSURE_gradS_mean.npz")
FLUXES = list(D["fluxes"])
ALPHA_LABELS = [r"$\alpha = 1\cdot 10^{-6}$", r"$\alpha = 5\cdot 10^{-6}$", r"$\alpha = 1\cdot 10^{-5}$"]
SHAPES = {"medium_ellipse": 0, "spiral": 1, "circle": 2}
BOX = dict(boxstyle="round,pad=0,rounding_size=0.006", facecolor="0.85", edgecolor="0.5", clip_on=False)
VMAX = 2.0   # reconstructions / alpha
FS = 9


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("black"); sp.set_linewidth(1.0)


def box(fig, rect, text, rotation=0, fontsize=10):
    """Grey rounded box rect = (x, y, w, h) in figure fraction."""
    x, y, w, h = rect
    W, H = fig.get_size_inches()
    fig.patches.append(FancyBboxPatch((x, y), w, h, transform=fig.transFigure, mutation_aspect=W / H, **BOX))
    fig.text(x + w / 2, y + h / 2, text, rotation=rotation, ha="center", va="center", fontsize=fontsize)


def compact(blocks, out):
    """blocks: list of (block label, shape, conv). Layout in inches, image side s."""
    s, g_pair, g_alpha, g_gt = 0.95, 0.04, 0.22, 0.3      # image side, gaps inside a pair / between alphas / after GT
    left, lab_w, cb_gap, cb_w, right = 0.12, 0.22, 0.08, 0.08, 0.42
    top_box, title_h, g_rows, g_blocks, bottom = 0.3, 0.17, 0.12, 0.25, 0.05
    W = left + lab_w + 0.22 + s + g_gt + 3 * (2 * s + g_pair) + 2 * g_alpha + cb_gap + cb_w + right
    block_h = title_h + s + g_rows + s
    H = top_box + len(blocks) * block_h + (len(blocks) - 1) * g_blocks + bottom
    fig = plt.figure(figsize=(W, H))
    fx, fy = (lambda v: v / W), (lambda v: v / H)
    x_gt = left + lab_w + 0.22
    x_cols = []
    x = x_gt + s + g_gt
    for a in range(3):
        x_cols.append((x, x + s + g_pair)); x += 2 * s + g_pair + g_alpha
    x_cb = x - g_alpha + cb_gap
    # contrast boxes on top
    for a in range(3):
        x0 = x_cols[a][0]
        box(fig, (fx(x0 + 0.03), fy(H - top_box + 0.03), fx(2 * s + g_pair - 0.06), fy(top_box - 0.08)), ALPHA_LABELS[a])
    y = H - top_box
    for (label, shape, conv) in blocks:
        si = SHAPES[shape]
        gt = (D["gt_conv"] if conv else D["gt"])[0, si] / FLUXES[0]
        rex, md, gtf = (D["rex_conv"], D["md_conv"], D["gt_conv"]) if conv else (D["rex"], D["md"], D["gt"])
        y_rec = y - title_h - s
        y_err = y_rec - g_rows - s
        # block label (left, rotated grey box)
        box(fig, (fx(left), fy(y_err + 0.04), fx(lab_w), fy(2 * s + g_rows - 0.08)), label, rotation=90)
        # ground truth, once
        ax = fig.add_axes([fx(x_gt), fy(y_rec), fx(s), fy(s)])
        im = ax.imshow(gt, cmap="hot", vmin=0, vmax=VMAX); frame(ax)
        ax.set_title("Ground truth", fontsize=FS, pad=3)
        for a in range(3):
            for k, (arr, name) in enumerate(((rex, "REXPACO"), (md, "MD-REX"))):
                xa = x_cols[a][k]
                ax = fig.add_axes([fx(xa), fy(y_rec), fx(s), fy(s)])
                ax.imshow(arr[a, si] / FLUXES[a], cmap="hot", vmin=0, vmax=VMAX); frame(ax)
                ax.set_title(name, fontsize=FS, pad=3)
                if a == 0 and k == 0:
                    ax.set_ylabel("Reconstruction", fontsize=FS)
                ax = fig.add_axes([fx(xa), fy(y_err), fx(s), fy(s)])
                ime = ax.imshow(np.abs(arr[a, si] - gtf[a, si]) / FLUXES[a], cmap="gray", vmin=0, vmax=1); frame(ax)
                if a == 0 and k == 0:
                    ax.set_ylabel("Relative error", fontsize=FS)
        # colorbars: reconstructions (x alpha) and relative errors
        cax = fig.add_axes([fx(x_cb), fy(y_rec), fx(cb_w), fy(s)])
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=FS - 1)
        cb.ax.text(0.0, 1.03, r"$\times\,\alpha$", transform=cb.ax.transAxes, ha="left", va="bottom", fontsize=FS)
        cax = fig.add_axes([fx(x_cb), fy(y_err), fx(cb_w), fy(s)])
        cb = fig.colorbar(ime, cax=cax); cb.ax.tick_params(labelsize=FS - 1)
        y = y_err - g_blocks
    fig.savefig(ROOT / "figures" / out)
    print(ROOT / "figures" / out)
    plt.close(fig)


compact([(r"$\langle\widehat{\mathbf{x}}_{\lambda}\rangle_{\lambda}$", "medium_ellipse", False),
         (r"$\langle\mathbf{H}_{\lambda}\widehat{\mathbf{x}}_{\lambda}\rangle_{\lambda}$", "medium_ellipse", True)],
        "comp_compact_medium_ellipse.pdf")
compact([("Spiral", "spiral", False), ("Circle", "circle", False)], "comp_compact_spiral_circle.pdf")
compact([("Spiral", "spiral", True), ("Circle", "circle", True)], "comp_compact_conv_spiral_circle.pdf")
