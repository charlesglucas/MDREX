"""Grey rounded boxes with the contrast alpha, placed on top of the columns of the comparison figures in the paper
(same style as the row / column labels of plot_grids_combined.py). Output: figures/alpha_box_<1em6|5em6|1em5>.pdf"""
import pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[2]
LABELS = {"1em6": r"$\alpha = 1\cdot 10^{-6}$", "5em6": r"$\alpha = 5\cdot 10^{-6}$", "1em5": r"$\alpha = 1\cdot 10^{-5}$"}
for tag, label in LABELS.items():
    fig = plt.figure(figsize=(3.0, 0.36))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.04, 0.08), 0.92, 0.84, transform=ax.transAxes,
                                boxstyle="round,pad=0,rounding_size=0.12", facecolor="0.85", edgecolor="0.5",
                                mutation_aspect=0.12))
    ax.text(0.5, 0.5, label, transform=ax.transAxes, ha="center", va="center", fontsize=13)
    fig.savefig(ROOT / "figures" / f"alpha_box_{tag}.pdf")
    plt.close(fig)
