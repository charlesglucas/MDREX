import sys, pathlib, os


sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
ROOT = pathlib.Path(__file__).resolve().parents[2]

from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, PowerNorm
from matplotlib.cm import ScalarMappable
import torch
import hydra
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    
    data_file = 'HR_4796'
    data = np.load(ROOT / f"results/realdata/{data_file}.npz", allow_pickle=True)
    y = data["y"]
    x_disk_store = data["x"]
    Ax_disk_store = data["Ax"]
    nsmooth = data["n_smooth"]
    nsparse = data["n_sparse"]

    k,j = 1,2
    plt.figure(1)
    plt.subplot(2,2,1); plt.imshow(np.squeeze(y[0,0,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0^{(1)}$"); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(np.squeeze(y[0,1,0,:,:]),vmin=0,vmax=50); plt.title(r"$\mathbf{y}_0^{(2)}$"); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(np.squeeze(x_disk_store[k,j][0,:,:])); plt.title(r"$x$"); plt.colorbar()
    plt.subplot(2,2,4); plt.imshow(np.squeeze(x_disk_store[k,j][1,:,:])); plt.title(r"$x$"); plt.colorbar()
    plt.tight_layout()
    plt.show()
    # plt.savefig("gridsearch.jpg", dpi=300)

    plt.figure(2)
    for k in range(3):
        for j in range(3):
            h=k*3+j+1
            sm = nsmooth+k; 
            sp = nsparse+j
            plt.subplot(3,3,h); plt.imshow(np.squeeze(np.mean(x_disk_store[k+3,j+3], axis=0))); plt.title(rf"$(10^{{{sm}}},\, 10^{{{sp}}})$"); plt.colorbar()
            plt.suptitle(r"$\mathbf{x}$");
            plt.tight_layout()
            plt.show()

    plt.figure(2)
    for k in range(3):
        for j in range(3):
            h=k*4+j+1
            sm = nsmooth+k; 
            sp = nsparse+j
            plt.subplot(1,4,h); plt.imshow(np.squeeze(np.mean(x_disk_store[2,2], axis=0))); plt.title(rf"$(10^{{{sm}}},\, 10^{{{sp}}})$"); plt.colorbar()
            plt.suptitle(r"$\mathbf{x}$");
            plt.tight_layout()
            plt.show()

    plt.figure(3)
    plt.imshow(np.squeeze(np.mean(x_disk_store[3,4], axis=0)), cmap='inferno'); plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.axis('off')
    plt.savefig("figures/reconstruction.jpg", dpi=300, bbox_inches='tight', pad_inches=0)

    k = 3; j = 4
    fig, axs = plt.subplots(1, 2, figsize=(6, 2))
    data1 = x_disk_store[k, j][0]
    data2 = x_disk_store[k, j][1]
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())
    im1 = axs[0].imshow(data1, cmap='hot', vmin=vmin, vmax=vmax)
    im2 = axs[1].imshow(data2, cmap='hot', vmin=vmin, vmax=vmax)
    axs[0].axis('off')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs, shrink=0.8, pad=0.1)
    sm = nsmooth + k
    sp = nsparse + j
    fig.savefig(f"figures/{data_file}_musmooth{sm}_musparse{sp}.jpg", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


    k = 4; j = 4
    sm = nsmooth + k
    sp = nsparse + j
    fig, ax = plt.subplots(figsize=(4, 4))
    data1 = x_disk_store[k, j][0].astype(np.float32)
    data2 = x_disk_store[k, j][1].astype(np.float32)
    eps = 1e-12
    # d1 = np.sqrt(np.abs(data1) + eps)
    # d2 = np.sqrt(np.abs(data2) + eps)
    d1 = data1
    d2 = data2
    d1 = (d1 - d1.min()) / (d1.max() - d1.min() + eps)
    d2 = (d2 - d2.min()) / (d2.max() - d2.min() + eps)
    color_map = np.array([[0.0, 0.5, 1.0],
                          [1.0, 0.5, 0.0]], dtype=np.float32)
    rgb = np.zeros(data1.shape + (3,), dtype=np.float32)
    rgb[..., 0] = color_map[0, 0] * d1 + color_map[1, 0] * d2
    rgb[..., 1] = color_map[0, 1] * d1 + color_map[1, 1] * d2
    rgb[..., 2] = color_map[0, 2] * d1 + color_map[1, 2] * d2
    rgb = np.clip(rgb, 0.0, 1.0)
    ax.imshow(1*rgb)
    ax.set_title("Bleu = canal 1, rouge = canal 2")
    ax.axis("off")
    # plt.savefig(f"figures/{data_file}_musmooth{sm}_musparse{sp}_RGB.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.show()



    ## Plot in RGB with colorbars
    k = 4; j = 4
    sm = nsmooth + k
    sp = nsparse + j
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0.08, right=0.82, bottom=0.14, top=0.96)
    data1 = x_disk_store[k, j][0].astype(np.float32)
    data2 = x_disk_store[k, j][1].astype(np.float32)
    eps = 1e-12
    # d1 = np.sqrt(np.abs(data1) + eps)
    # d2 = np.sqrt(np.abs(data2) + eps)
    d1 = data1
    d2 = data2
    common_vmin = min(data1.min(), data2.min())
    common_vmax = max(data1.max(), data2.max())
    gamma = 0.5
    norm = PowerNorm(gamma=gamma, vmin=common_vmin, vmax=common_vmax)
    d1 = norm(data1)
    d2 = norm(data2)
    color_map = np.array([[0.0, 0.5, 1.0],
                          [1.0, 0.5, 0.0]], dtype=np.float32)
    rgb = np.zeros(data1.shape + (3,), dtype=np.float32)
    rgb[..., 0] = color_map[0, 0] * d1 + color_map[1, 0] * d2
    rgb[..., 1] = color_map[0, 1] * d1 + color_map[1, 1] * d2
    rgb[..., 2] = color_map[0, 2] * d1 + color_map[1, 2] * d2
    rgb = np.clip(rgb, 0.0, 1.0)
    ax.imshow(rgb)
    ax.axis("off")

    blue_map = LinearSegmentedColormap.from_list(
        "black_blue", ["black", "blue"]
    )
    orange_map = LinearSegmentedColormap.from_list(
        "black_orange", ["black", "orange"]
    )

    blue_bar = ScalarMappable(
        norm, blue_map
    )
    orange_bar = ScalarMappable(
        norm, orange_map
    )
    blue_bar.set_array(data1)
    orange_bar.set_array(data2)

    divider = make_axes_locatable(ax)
    blue_cax = divider.append_axes("bottom", size="4%", pad=0.02)
    orange_cax = divider.append_axes("right", size="4%", pad=0.02)

    blue_cbar = fig.colorbar(blue_bar, cax=blue_cax, orientation="horizontal",
                             label="Canal 1")
    blue_cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    blue_cbar.formatter.set_powerlimits((0, 0))
    blue_cbar.update_ticks()
    blue_offset = blue_cbar.ax.xaxis.get_offset_text()
    blue_offset.set_position((.98, 0.1))
    blue_offset.set_horizontalalignment("left")
    blue_offset.set_verticalalignment("top")

    orange_cbar = fig.colorbar(orange_bar, cax=orange_cax, orientation="vertical",
                               label="Canal 2")
    orange_cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    orange_cbar.formatter.set_powerlimits((0, 0))
    orange_cbar.update_ticks()
    fig.canvas.draw()
    orange_offset = orange_cbar.ax.yaxis.get_offset_text()
    orange_offset_text = orange_offset.get_text()
    if not orange_offset_text:
        exponent = orange_cbar.formatter.orderOfMagnitude
        orange_offset_text = rf"$\times 10^{{{exponent}}}$"
    orange_offset.set_visible(False)
    orange_cbar.ax.text(0.5, 1.0, orange_offset_text,
                        transform=orange_cbar.ax.transAxes,
                        ha="center", va="bottom", clip_on=False)
    plt.savefig(f"figures/{data_file}_musmooth{sm}_musparse{sp}_RGB.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


    
    # from astropy.io import fits
    # hdu = fits.PrimaryHDU(data=x_disk_store[1,0]);
    # hdu.writeto(f"results/{data_file}.fits", overwrite=True);

if __name__ == "__main__":
    main()
