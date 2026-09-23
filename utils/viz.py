import numpy as np
import numpy as np
import os
import matplotlib as mpl
from einops import rearrange
from matplotlib import cm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, RangeSlider
from PIL import Image, ImageDraw
from utils.coords import polar_to_cart

from matplotlib.backend_bases import MouseButton

all_colors = {
    "FP": "orange",
    "U": "pink",
    "A": "cyan",
    "B": "magenta",
    "C": "g",
    "add": "red",
    "edit": "blue",
}


def to_pil(samples, caption, cmap="viridis", vmax=1, vmin=0):
    """(B, W, C)""" ""
    b, h, w = samples.shape
    # vmax = -0.5
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # color_map = cm[cmap]
    mapping = cm.ScalarMappable(norm=norm, cmap=cmap)
    pil_imgs = []
    # T = len(samples)
    for i, sample in enumerate(samples):
        # sample = ((sample + 1) * 127.5).clip(0, 255).to(np.uint8)
        # sample_c = cm.viridis(sample, vmin=-1, vmax=1) * 255
        # sample_c = np.uint8(sample_c)
        # t = (T - i - 1) * factor
        # t = (T - i - 1) * factor
        rgba = mapping.to_rgba(sample)
        # print(rgba.shape)
        rgba = np.uint8(rgba * 255)
        img = Image.fromarray(rgba)
        ImageDraw.Draw(img).text(  # Image
            (0, 0),
            caption.format(i),
            (255, 255, 255),  # Coordinates  # Text  # Color
        )

        # draw = ImageDraw.Draw(img)
        # # font = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # # draw.text((x, y),"Sample Text",(r,g,b))
        # draw.text((0, 0),f"t={t}",(255,255,255),font=font)
        pil_imgs.append(img)

    return pil_imgs


# def save_gif(data, path, prefix, reverse=True, vmax=1, cmap="jet"):
def save_gif(
    data, path, reverse=True, vmax=1, vmin=-1, cmap="jet", caption="i={}"
):
    """
    save numpy array as a gif

    Args:
    -----
    * data: (C, H, W)
    """

    # path_gif = os.path.join(path, f"{prefix}.gif")
    if reverse:
        data = data[::-1]
    pil_imgs = to_pil(data, vmax=vmax, vmin=vmin, cmap=cmap, caption=caption)
    pil_imgs[0].save(
        path,
        format="GIF",
        append_images=pil_imgs + 5 * [pil_imgs[-1]],
        save_all=True,
        duration=100,
        loop=0,
    )
    print(f"GIF saved to:\n{path}")


def maps_viewer(
    maps,
    labels_pred,
    all_coords,
    status=None,
    quantile=0.02,
    range_hist=[-5, 15],
    bins=50,
):
    """
    maps: np.array(T, C, H, W)
    labels_pred: np.array(T, C, H, W)
    """
    matplotlib.rcParams["keymap.back"].remove("left")
    matplotlib.rcParams["keymap.forward"].remove("right")

    assert maps.ndim == 4
    assert labels_pred.ndim == 4
    T, C, H, W = maps.shape

    if range_hist == "auto":
        mask_inf = np.isinf(maps)
        maps[mask_inf] = np.nan
        mask_valid = ~np.isnan(maps)
        r_min = np.quantile(maps[mask_valid], quantile)
        r_max = np.quantile(maps[mask_valid], 1 - quantile)
        range_hist = [r_min, r_max]
        print(f"{range_hist=}")

    if range_hist is not None:
        width_hist = (range_hist[1] - range_hist[0]) / bins
        print("clipping..")
        # cube = np.clip(cube, range_hist[0], range_hist[1])
        maps.clip(range_hist[0], range_hist[1], out=maps)
        print("done..")
        range_min, range_max = range_hist[0], range_hist[1]
    else:
        range_min = np.min(maps)
        range_max = np.max(maps)
        width_hist = (range_max - range_min) / bins

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=C + 1,
        ncols=3,
        height_ratios=[10] * C + [1],
        width_ratios=[1, 1, 1],
    )
    axes_img = []
    axes_img_labels = []
    for c in range(C):
        axes_img.append(fig.add_subplot(gs[c, 0]))
        axes_img_labels.append(fig.add_subplot(gs[c, 1]))
    # ax_img = fig.add_subplot(gs[:2, 0])
    # ax_img_labels = fig.add_subplot(gs[:2, 1])
    # ax_quality = fig.add_subplot(gs[2, 0])
    ax_time = fig.add_subplot(gs[-1, 0])

    ax_hist = fig.add_subplot(gs[0, 2])
    ax_contrast = fig.add_subplot(gs[1, 2])
    # ax_buttons = fig.add_subplot(gs[2:, 1])
    extent = (W / 2) * 0.01225
    extent = [-extent, extent, -extent, extent]
    params = {"t": 0, "vmin": -1, "vmax": 1, "mask": True, "coords": True}

    all_img = []
    all_labels = []
    all_scatter = []
    for c in range(C):
        all_img.append(axes_img[c].imshow(maps[0, c], vmin=-1, vmax=1))
        all_labels.append(
            axes_img_labels[c].imshow(
                labels_pred[0, c],
                vmin=0,
                vmax=1,
                cmap="binary",
            )
        )
        all_scatter.append(
            axes_img_labels[c].scatter(
                all_coords[c]["coords_x"][0],
                all_coords[c]["coords_y"][0],
            )
        )

    time_slider = Slider(
        ax=ax_time,
        label="Timestep",
        valmin=0,
        valmax=T - 1,
        valinit=0,
        valstep=1,
        orientation="horizontal",
    )

    # print(f"{range_min=}, {range_max=}")
    slider = RangeSlider(
        ax_contrast,
        "Contrast",
        range_min,
        range_max,
        valinit=range_hist,
        valfmt="%0.1f",
    )

    def data_hist(t=None):
        if t is None:
            data_hist = maps.flatten()
        else:
            data_hist = maps[t].flatten()
        return data_hist

    print("computing hist global..")
    # H_all, bins_all = np.histogram(
    # data_hist(), bins=100, range=range_hist, density=True
    # )
    # ax_hist.bar(
    # bins_all[:-1], H_all, width=width_hist, color="gray", alpha=0.5
    # )
    print("computing hist global done")
    H_t, bins_t = np.histogram(
        data_hist(0), bins=100, range=range_hist, density=True
    )
    hist_t = ax_hist.bar(
        bins_t[:-1], H_t, width=width_hist, color="blue", alpha=0.5
    )
    # hist = ax_hist.hist(cube[0].flatten(), bins=bins, range=[-10, 10])
    lower_limit_line = ax_hist.axvline(slider.val[0], color="k")
    upper_limit_line = ax_hist.axvline(slider.val[1], color="k")

    def update_plots():
        nonlocal params
        t = params["t"]
        vmin = params["vmin"]
        vmax = params["vmax"]

        lower_limit_line.set_xdata([vmin, vmin])
        upper_limit_line.set_xdata([vmax, vmax])

        # ax_hist.cla()
        H_t, bins_t = np.histogram(
            data_hist(t), bins=100, range=range_hist, density=True
        )
        for rect, h in zip(hist_t, H_t):
            rect.set_height(h)

        for c in range(C):
            # all_scatter[c].remove()
            axes_img_labels[c].cla()
            # all_labels[c].clear()
        # axes_img_labels = []
        for c in range(C):
            ax_labels = axes_img_labels
            # ax_labels = axes_img
            ax_labels[c].cla()

            all_img[c].set_data(maps[t, c])
            # all_labels[c].set_data(labels_pred[t].reshape(C * H, W))
            # all_labels.append(
            # axes_img_labels[c].imshow(labels_pred[t, c])
            # )
            all_img[c].norm.vmin = vmin
            all_img[c].norm.vmax = vmax

            # all_labels.append(
            axes_img_labels[c].imshow(
                labels_pred[t, c],
                vmin=0,
                vmax=1,
                cmap="binary",
            )
            # )
            # all_scatter[c] = axes_img_labels[c].scatter(
            # all_scatter.append(

            x = all_coords[c]["coords_x_gt"][t]
            y = all_coords[c]["coords_y_gt"][t]
            ax_labels[c].scatter(
                x,
                y,
                s=1000,
                edgecolors="purple",
                linewidth=3,
                facecolors="none",
                marker="^",
            )
            mask = all_coords[c]["status_tp"][t].astype(bool)
            print(f"{c} tp={np.sum(mask)}")
            x = all_coords[c]["coords_x"][t][mask]
            y = all_coords[c]["coords_y"][t][mask]
            ax_labels[c].scatter(
                x,
                y,
                s=1000,
                edgecolors="g",
                linewidth=3,
                facecolors="none",
            )

            mask = all_coords[c]["status_fp"][t].astype(bool)
            print(f"{c} fp={np.sum(mask)}")
            print(mask)
            x = all_coords[c]["coords_x"][t][mask]
            y = all_coords[c]["coords_y"][t][mask]
            ax_labels[c].scatter(
                x,
                y,
                s=1000,
                edgecolors="r",
                linewidth=3,
                facecolors="none",
            )

            mask = all_coords[c]["status_real"][t].astype(bool)
            print(f"{c} n_real={np.sum(mask)}")
            x = all_coords[c]["coords_x"][t][mask]
            y = all_coords[c]["coords_y"][t][mask]
            ax_labels[c].scatter(
                x,
                y,
                s=1000,
                edgecolors="b",
                linewidth=3,
                facecolors="none",
            )

            # )
            fig.canvas.draw_idle()

    def update(changed_params):
        nonlocal params
        for k, v in changed_params.items():
            # print(f"{k}:{v}")
            params[k] = v
        update_plots()

    time_slider.on_changed(lambda x: update({"t": x}))
    # check_buttons.on_clicked(lambda x: update({x: not params[x]}))
    slider.on_changed(lambda x: update({"vmin": x[0], "vmax": x[1]}))

    def key_event(event):
        # print(event.key)
        t_current = params["t"]
        if event.key == "right":
            t_new = t_current + 1
        elif event.key == "left":
            t_new = t_current - 1
        else:
            return
        if (t_new >= 0) and (t_new < T):
            update({"t": t_new})
            time_slider.set_val(t_new)

    fig.canvas.mpl_connect("key_press_event", key_event)
    plt.show()


def patches_viewer(cube, **kwargs):
    T, C, H, W = cube.shape
    assert np.sqrt(C) % 1 == 0
    ps = int(np.sqrt(C))
    print(f"{ps=}")

    cube = rearrange(
        cube, "t (psh psw) h w -> t (h psh) (w psw)", psh=ps, psw=ps
    )
    cube_3d_viewer(cube, **kwargs)


def cube_3d_viewer(
    cube,
    x_coords=None,
    y_coords=None,
    masks=None,
    status=None,
    quality=None,
    labels=None,
    cmap=None,
    alpha=0.2,
    bins=100,
    # range_hist=[-3, 60],
    # range_hist=[-5, 5],
    range_hist="auto",
    slider_init=(-1, 1),
    channels=None,
    dim_format="TCHW",
    quantile=0.02,
):
    """
    data: (T, H, W)
    x_coords: (T, n_sources)
    y_coords: (T, n_sources)
    masks: (T, H, W)
    status: (n_sources,)
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, CheckButtons, RangeSlider

    try:
        matplotlib.rcParams["keymap.back"].remove("left")
        matplotlib.rcParams["keymap.forward"].remove("right")
    except ValueError:
        print(f"could not update hotkeys graphical interface")
        pass

    cube = cube.copy()
    if channels is not None:
        # C, T, H, W = cube.shape
        if dim_format == "TCHW":
            T, C, H, W = cube.shape
            cube = rearrange(cube, "t c h w -> t h (c w)")
        elif dim_format == "CTHW":
            C, T, H, W = cube.shape
            cube = rearrange(cube, "c t h w -> t h (c w)")
        assert channels == C, f"{cube.shape}"
        # cube = rearrange(cube, "c t h w -> t h (c w)")
        # cube = rearrange(cube, "t c h w -> t h (c w)")
        W = W * C
    else:
        T, H, W = cube.shape

    if range_hist == "auto":
        mask_inf = np.isinf(cube)
        cube[mask_inf] = np.nan
        mask_valid = ~np.isnan(cube)
        r_min = np.quantile(cube[mask_valid], quantile)
        r_max = np.quantile(cube[mask_valid], 1 - quantile)
        range_hist = [r_min, r_max]
        print(f"{range_hist=}")

    # fig, ax = plt.subplots(
    # nrows=5,
    # ncols=1,
    # gridspec_kw={"height_ratios": [5, 0.25, 0.25, 0.25, 0.5]},
    # )
    if quality is None:
        quality = -np.ones(len(cube))
    # range_hist = [-2.5, 2.5]
    # range_hist = [-.5, .5]
    # range_hist = [-10, 10]
    if range_hist is not None:
        width_hist = (range_hist[1] - range_hist[0]) / bins
        print("clipping..")
        # cube = np.clip(cube, range_hist[0], range_hist[1])
        cube.clip(range_hist[0], range_hist[1], out=cube)
        print("done..")
        range_min, range_max = range_hist[0], range_hist[1]
    else:
        range_min = np.min(cube)
        range_max = np.max(cube)
        width_hist = (range_max - range_min) / bins

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        height_ratios=[10, 0.5, 0.5, 0.5],
        width_ratios=[2, 1],
    )
    ax_img = fig.add_subplot(gs[:2, 0])
    ax_quality = fig.add_subplot(gs[2, 0])
    ax_time = fig.add_subplot(gs[3, 0])

    ax_hist = fig.add_subplot(gs[0, 1])
    ax_contrast = fig.add_subplot(gs[1, 1])
    ax_buttons = fig.add_subplot(gs[2:, 1])
    extent = (W / 2) * 0.01225
    extent = [-extent, extent, -extent, extent]
    params = {"t": 0, "vmin": -1, "vmax": 1, "mask": True, "coords": True}

    # img = ax[0].imshow(frame, extent=extent)
    # mg = ax[0].imshow(frame)
    img = ax_img.imshow(cube[0], vmin=-1, vmax=1, cmap=cmap)
    if masks is not None:
        img_mask = ax_img.imshow(masks[0], alpha=alpha)

    if x_coords is not None:
        if status is not None:
            # [FP, U, A, B, C]
            # all_colors = ["orange", "pink", "cyan", "magenta", "g"]
            colors = [all_colors[s.split("_")[0]] for s in status]
        else:
            colors = "black"
        sc = ax_img.scatter(x_coords[0], y_coords[0], marker="+", c=colors)
    if labels is not None:
        ax_img.set_title(labels[0])

    time_slider = Slider(
        ax=ax_time,
        label="Timestep",
        valmin=0,
        valmax=T - 1,
        valinit=0,
        valstep=1,
        orientation="horizontal",
    )

    check_buttons = CheckButtons(
        ax_buttons,
        labels=["mask", "coords"],
        actives=[params["mask"], params["coords"]],
    )

    # print(f"{range_min=}, {range_max=}")
    slider = RangeSlider(
        ax_contrast,
        "Contrast",
        range_min,
        range_max,
        # valinit=(-1, 1),
        # valinit=(range_hist[0], range_hist[1]),
        # valinit=slider_init,
        valinit=range_hist,
        valfmt="%0.1f",
    )
    # if masks is not None:
    # data_hist = cube[masks].flatten()
    # else:
    # data_hist = cube.flatten()

    def data_hist(t=None):
        if t is None:
            if masks is not None:
                data_hist = cube[masks].flatten()
            else:
                data_hist = cube.flatten()
        else:
            if masks is not None:
                data_hist = cube[t, masks[t]].flatten()
            else:
                data_hist = cube[t].flatten()
        return data_hist

    print("computing hist global..")
    # H_all, bins_all = np.histogram(
    # data_hist(), bins=100, range=range_hist, density=True
    # )
    # ax_hist.bar(
    # bins_all[:-1], H_all, width=width_hist, color="gray", alpha=0.5
    # )
    print("computing hist global done")
    H_t, bins_t = np.histogram(
        data_hist(0), bins=100, range=range_hist, density=True
    )
    # out = np.histogram(cube[0].flatten(), bins=10, range=[-10, 10])
    # breakpoint()
    hist_t = ax_hist.bar(
        bins_t[:-1], H_t, width=width_hist, color="blue", alpha=0.5
    )
    # hist = ax_hist.hist(cube[0].flatten(), bins=bins, range=[-10, 10])
    lower_limit_line = ax_hist.axvline(slider.val[0], color="k")
    upper_limit_line = ax_hist.axvline(slider.val[1], color="k")

    colors = {-1: "gray", 0: "red", 1: "orange", 2: "green"}
    quality_colors = [colors[q] for q in quality]
    ax_quality.bar(
        np.arange(T),
        np.ones(T),
        width=0.9,
        color=quality_colors,
        alpha=0.5,
    )
    quality_t = ax_quality.axvline(0, color="k")
    ax_quality.set_xlim(-0.5, T - 0.5)

    def update_plots():
        nonlocal params
        t = params["t"]
        vmin = params["vmin"]
        vmax = params["vmax"]
        visi_mask = params["mask"]
        visi_coords = params["coords"]

        lower_limit_line.set_xdata([vmin, vmin])
        upper_limit_line.set_xdata([vmax, vmax])
        quality_t.set_xdata([t, t])

        # ax_hist.cla()
        H_t, bins_t = np.histogram(
            data_hist(t), bins=100, range=range_hist, density=True
        )
        for rect, h in zip(hist_t, H_t):
            rect.set_height(h)
        # ax_hist.bar(bins_all[:-1], H_all, width=wid, color="gray", alpha=0.5)
        # ax_hist.bar(bins_t[:-1], H_t, width=1, color="blue", alpha=0.5)
        # # hist = ax_hist.hist(cube[0].flatten(), bins=bins, range=[-10, 10])
        # ax_hist.axvline(slider.val[0], color="k")
        # ax_hist.axvline(slider.val[1], color="k")

        # H, bins = np.histogram(
        # cube[t].flatten(), bins=100, range=[-10, 10], density=True
        # )
        # ax_hist.hist(cube[t].flatten(), bins=bins, range=[-10, 10])

        # frame = np.clip(cube[_t, :, :], vmin, vmax)
        # frame = (frame - vmin) / (vmax - vmin)
        img.set_data(cube[t])
        img.norm.vmin = vmin
        img.norm.vmax = vmax
        fig.canvas.draw_idle()
        if masks is not None:
            img_mask.set_data(masks[t])
            img_mask.set_visible(visi_mask)
        if x_coords is not None:
            sc.set_offsets(np.c_[x_coords[t], y_coords[t]])
            sc.set_visible(visi_coords)
        if labels is not None:
            ax_img.set_title(labels[t])

    def update(changed_params):
        nonlocal params
        for k, v in changed_params.items():
            # print(f"{k}:{v}")
            params[k] = v
        update_plots()

    time_slider.on_changed(lambda x: update({"t": x}))
    check_buttons.on_clicked(lambda x: update({x: not params[x]}))
    slider.on_changed(lambda x: update({"vmin": x[0], "vmax": x[1]}))

    # plt.tight_layout()
    def mouse_event(event):
        if event.button == MouseButton.RIGHT:
            t = params["t"]
            if event.dblclick:
                x = event.xdata
                y = event.ydata
                txt = f"{labels[t]} x={x:.2f} y={y:.2f}"
                # print('x: {} and y: {}'.format(event.xdata, event.ydata))
            else:
                txt = f"{labels[t]}"
            print(txt)
            # pyperclip.copy(txt)

    def key_event(event):
        # print(event.key)
        t_current = params["t"]
        if event.key == "right":
            t_new = t_current + 1
        elif event.key == "left":
            t_new = t_current - 1
        else:
            return
        if (t_new >= 0) and (t_new < T):
            update({"t": t_new})
            time_slider.set_val(t_new)

    fig.canvas.mpl_connect("button_press_event", mouse_event)
    fig.canvas.mpl_connect("key_press_event", key_event)
    plt.show()


def view_dataset_item(item):
    frame = item["frame"]
    mask = item["mask"]
    quality = item["quality"]
    status = item["status"]
    idx = item["idx"]
    # x_coords = item["x_coords"]
    # k
    x_coords, y_coords = polar_to_cart(
        pa=item["pa"], sep=item["sep"], rot=item["rot"]
    )
    # print(item["dm"])
    cube_3d_viewer(
        cube=frame,
        masks=mask,
        quality=quality,
        status=status,
        x_coords=x_coords,
        y_coords=y_coords,
        labels=idx,
    )


def save_image(data, filename, vmin=None, vmax=None):
    fig = plt.figure(figsize=(1, 1))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    # ax.imshow(data, cmap="gray")
    ax.imshow(data, vmin=vmin, vmax=vmax)
    fig.savefig(filename, dpi=data.shape[0])
    plt.close(fig)
