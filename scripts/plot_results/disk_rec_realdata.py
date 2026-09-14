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
from astropy.io import fits


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg):
    
    data_file = 'SAO_206462'
    display_titles = {
        "HR_4796": "HR 4796A",
        "RY_lup": "RY Lupi",
        "PDS_70": "PDS 70",
        "HD_169142": "HD 169142",
        "SAO_206462": "SAO 206462",
    }
    title = display_titles.get(data_file, data_file.replace('_', ' '))
    data = np.load(ROOT / f"results/realdata/{data_file}.npz", allow_pickle=True)
    y = data["y"]
    x_disk_store = data["x"]
    Ax_disk_store = data["Ax"]
    nsmooth = data["n_smooth"]
    nsparse = data["n_sparse"]

    lambda_files = list(
        (ROOT / "data/real_data/DISKS_IRDIS_CHARLES" / data_file).glob(
            "**/*-lam.fits"
        )
    )
    if len(lambda_files) != 1:
        raise FileNotFoundError(
            f"Expected one '*-lam.fits' file for {data_file}, found {len(lambda_files)}"
        )
    wavelengths = np.asarray(fits.getdata(lambda_files[0])).reshape(-1)

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


    def render_rgb(data1, data2, annotation, suffix, use_scaled_ticks=False):
        common_vmin = min(data1.min(), data2.min())
        common_vmax = max(data1.max(), data2.max())
        tick_step = 10 ** np.floor(np.log10(common_vmax))
        norm = PowerNorm(gamma=0.5, vmin=common_vmin, vmax=common_vmax)
        d1 = norm(data1)
        d2 = norm(data2)
        color_map = np.array([[0.0, 0.5, 1.0],
                              [1.0, 0.5, 0.0]], dtype=np.float32)
        rgb = np.zeros(data1.shape + (3,), dtype=np.float32)
        rgb[..., 0] = color_map[0, 0] * d1 + color_map[1, 0] * d2
        rgb[..., 1] = color_map[0, 1] * d1 + color_map[1, 1] * d2
        rgb[..., 2] = color_map[0, 2] * d1 + color_map[1, 2] * d2
        rgb = np.clip(rgb, 0.0, 1.0)

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.subplots_adjust(left=0.08, right=0.82, bottom=0.14, top=0.90)
        ax.imshow(rgb, origin="lower")
        if data_file == "RY_lup":
            height, width = rgb.shape[:2]
            ax.set_xlim(width / 4, 3 * width / 4)
            ax.set_ylim(height / 4, 3 * height / 4)
        ax.set_title(title, fontsize=16)
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
                ha="left", va="top", color="white", fontsize=18)
        ax.axis("off")

        blue_map = LinearSegmentedColormap.from_list(
            "black_blue", ["black", "blue"]
        )
        orange_map = LinearSegmentedColormap.from_list(
            "black_orange", ["black", "orange"]
        )
        blue_bar = ScalarMappable(norm, blue_map)
        orange_bar = ScalarMappable(norm, orange_map)
        blue_bar.set_array(data1)
        orange_bar.set_array(data2)
        divider = make_axes_locatable(ax)
        blue_cax = divider.append_axes("bottom", size="4%", pad=0.03)
        orange_cax = divider.append_axes("right", size="4%", pad=0.03)
        blue_cbar = fig.colorbar(
            blue_bar, cax=blue_cax, orientation="horizontal",
            label=rf"$\lambda = {wavelengths[0]:.3f}\,\mu\mathrm{{m}}$",
        )
        blue_cbar.locator = ticker.MultipleLocator(tick_step)
        exponent = int(np.floor(np.log10(common_vmax))) if common_vmax > 0 else 0
        scale = 10.0 ** exponent
        if use_scaled_ticks:
            blue_cbar.formatter = ticker.FuncFormatter(
                lambda value, position: f"{value / scale:g}"
            )
        else:
            blue_cbar.formatter = ticker.ScalarFormatter(useMathText=True)
            blue_cbar.formatter.set_powerlimits((0, 0))
        blue_cbar.update_ticks()
        blue_cbar.ax.tick_params(labelsize=13)
        blue_cbar.set_label(blue_cbar.ax.get_xlabel(), fontsize=14)
        if use_scaled_ticks and exponent != 0:
            fig.canvas.draw()
            blue_cbar.ax.xaxis.get_offset_text().set_visible(False)
            blue_cbar.ax.text(1.15, -1.4, rf"$\times 10^{{{exponent}}}$",
                              transform=blue_cbar.ax.transAxes,
                              ha="right", va="top", clip_on=False)

        orange_cbar = fig.colorbar(
            orange_bar, cax=orange_cax, orientation="vertical",
            label=rf"$\lambda = {wavelengths[1]:.3f}\,\mu\mathrm{{m}}$",
        )
        orange_cbar.locator = ticker.MultipleLocator(tick_step)
        orange_cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        orange_cbar.formatter.set_powerlimits((0, 0))
        orange_cbar.update_ticks()
        orange_cbar.ax.tick_params(labelsize=13)
        orange_cbar.set_label(orange_cbar.ax.get_ylabel(), fontsize=14)
        fig.canvas.draw()
        orange_cbar.ax.yaxis.get_offset_text().set_visible(False)
        fig.savefig(f"figures/{data_file}_{suffix}_RGB.pdf", dpi=300,
                    bbox_inches="tight", pad_inches=0.1)
        plt.show()
        return rgb, data1.copy(), data2.copy(), norm, blue_map, orange_map

    k = 4; j = 4
    sm = nsmooth + k
    sp = nsparse + j
    x_result = render_rgb(
        x_disk_store[k, j][0].astype(np.float32),
        x_disk_store[k, j][1].astype(np.float32),
        r"$\widehat{\mathbf{x}}_{\lambda}$", "x",
    )
    ax_result = render_rgb(
        Ax_disk_store[k, j][0, 0, 0].astype(np.float32),
        Ax_disk_store[k, j][0, 1, 0].astype(np.float32),
        r"$\mathbf{A}_{\lambda,0}\,\widehat{\mathbf{x}}_{\lambda}$", "Ax",
        use_scaled_ticks=True,
    )

    combined_fig = plt.figure(figsize=(4.8, 9))
    combined_fig.subplots_adjust(
        left=0.12, right=0.82, bottom=0.12, top=0.94, hspace=0.23,
    )
    combined_grid = combined_fig.add_gridspec(2, 1)
    combined_fig.add_subplot(combined_grid[0, 0]).set_title(title, fontsize=18, pad=10)
    combined_plots = [
        (x_result, r"$\widehat{\mathbf{x}}_{\lambda}$"),
        (ax_result, r"$\mathbf{A}_{\lambda,0}\,\widehat{\mathbf{x}}_{\lambda}$"),
    ]
    for row, (result, annotation) in enumerate(combined_plots):
        image, channel1, channel2, image_norm, blue_map, orange_map = result
        combined_ax = combined_fig.axes[0] if row == 0 else combined_fig.add_subplot(combined_grid[row, 0])
        combined_ax.imshow(image, origin="lower")
        if data_file == "RY_lup":
            height, width = image.shape[:2]
            combined_ax.set_xlim(width / 4, 3 * width / 4)
            combined_ax.set_ylim(height / 4, 3 * height / 4)
        combined_ax.text(0.02, 0.98, annotation, transform=combined_ax.transAxes,
                         ha="left", va="top", color="white", fontsize=18)
        combined_ax.axis("off")
        combined_divider = make_axes_locatable(combined_ax)
        combined_blue_cax = combined_divider.append_axes("bottom", size="4%", pad=0.10)
        combined_orange_cax = combined_divider.append_axes("right", size="4%", pad=0.10)
        combined_blue_bar = ScalarMappable(image_norm, blue_map)
        combined_orange_bar = ScalarMappable(image_norm, orange_map)
        combined_blue_bar.set_array(channel1)
        combined_orange_bar.set_array(channel2)
        combined_tick_step = 10 ** np.floor(np.log10(max(channel1.max(), channel2.max())))
        combined_blue_cbar = combined_fig.colorbar(
            combined_blue_bar, cax=combined_blue_cax, orientation="horizontal",
            label=rf"$\lambda = {wavelengths[0]:.3f}\,\mu\mathrm{{m}}$",
        )
        combined_blue_cbar.locator = ticker.MultipleLocator(combined_tick_step)
        if row == 1:
            combined_exponent = int(np.floor(np.log10(max(channel1.max(), channel2.max()))))
            combined_scale = 10.0 ** combined_exponent
            combined_blue_cbar.formatter = ticker.FuncFormatter(
                lambda value, position: f"{value / combined_scale:g}"
            )
        else:
            combined_blue_cbar.formatter = ticker.ScalarFormatter(useMathText=True)
            combined_blue_cbar.formatter.set_powerlimits((0, 0))
        combined_blue_cbar.update_ticks()
        combined_blue_cbar.ax.tick_params(labelsize=13)
        combined_blue_cbar.set_label(combined_blue_cbar.ax.get_xlabel(), fontsize=14)
        if row == 1 and combined_exponent != 0:
            combined_fig.canvas.draw()
            combined_blue_cbar.ax.xaxis.get_offset_text().set_visible(False)
            combined_blue_cbar.ax.text(
                1.15, -1.4, rf"$\times 10^{{{combined_exponent}}}$",
                transform=combined_blue_cbar.ax.transAxes,
                ha="right", va="top", clip_on=False,
            )
        combined_orange_cbar = combined_fig.colorbar(
            combined_orange_bar, cax=combined_orange_cax, orientation="vertical",
            label=rf"$\lambda = {wavelengths[1]:.3f}\,\mu\mathrm{{m}}$",
        )
        combined_orange_cbar.locator = ticker.MultipleLocator(combined_tick_step)
        combined_orange_cbar.formatter = ticker.ScalarFormatter(useMathText=True)
        combined_orange_cbar.formatter.set_powerlimits((0, 0))
        combined_orange_cbar.update_ticks()
        combined_orange_cbar.ax.tick_params(labelsize=13)
        combined_orange_cbar.set_label(combined_orange_cbar.ax.get_ylabel(), fontsize=14)
        combined_fig.canvas.draw()
        combined_orange_cbar.ax.yaxis.get_offset_text().set_visible(False)
    combined_fig.savefig(
        f"figures/{data_file}_musmooth{sm}_musparse{sp}_RGB.pdf",
        dpi=300, bbox_inches="tight", pad_inches=0.1,
    )
    plt.show()


    
    # from astropy.io import fits
    # hdu = fits.PrimaryHDU(data=x_disk_store[1,0]);
    # hdu.writeto(f"results/{data_file}.fits", overwrite=True);

if __name__ == "__main__":
    main()