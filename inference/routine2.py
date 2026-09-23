#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:55:15 2022

@author: julesdallant
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as opt
import glob
import os
import sys
from math import pi, cos, sin
from scipy.ndimage import fourier_shift
import time

# plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def get_sortframes_vector(path):
    """
    Returns the sortframes vector contained in the directory
    '*sortframes_vector*' found in `path`.
    """
    filepath = glob.glob(os.path.join(path, "*-frame_selection_vector.fits"))
    if len(filepath) == 1:
        print(f"path sortframes: {filepath[0]}")
        return fits.getdata(filepath[0])
    else:
        dirpath = glob.glob(os.path.join(path, "*sortframes_vector*"))
        if len(dirpath) == 1 and os.path.isdir(dirpath[0]):
            filepath = glob.glob(
                os.path.join(dirpath[0], "*-frame_selection_vector.fits")
            )
            if len(filepath) == 1:
                print(f"path sortframes: {filepath[0]}")
                return fits.getdata(filepath[0])
            else:
                return None
        else:
            return None


def get_lambda_info(path):
    """
    Returns the vector contained in '*-lam.fits' found in `path`.
    """
    filepath = glob.glob(os.path.join(path, "*-lam.fits"))
    if len(filepath) == 1:
        return fits.getdata(filepath[0])
    else:
        raise ValueError("No '*-lam.fits' file found in path.")


def get_master_cube(path):
    """
    Returns both data and header contained in '*-center_im.fits'
    found in `path`.
    """
    filepath = glob.glob(os.path.join(path, "*-center_im.fits"))
    if len(filepath) == 1:
        return (fits.getdata(filepath[0]), fits.getheader(filepath[0]))
    else:
        raise ValueError("No '*-center_im.fits' file found in path.")


def image_center(NX, NY):
    """
    Returns the center coordinates of an image of size `NX` x `NY` using
    the SPHERE-DC's convention.
    """
    if NX == NY == 1024:  # IRDIS
        return np.array([512, 512])
    elif NX == NY == 290:  # IFS
        return np.array([145, 145])
    else:
        raise ValueError("Not a standard SPHERE cube size.")


def round_2_nearest_half(x):
    """
    Rounds `x` to nearest half.
    """
    return np.round(x * 2) / 2


def waffles_approx_pos(lam, im_center, waffle_orient, pixres):
    """
    Returns the approximate x and y positions of the four waffles for
    wavelengths `lam` given the image center coordinates `im_center`, the
    waffle orientation `waffle_orient` ('x' or '+') and the pixel resolution
    `pixres`.
    """
    nλ = len(lam)
    d = 14e-6 * lam / 8.0 * 648000.0 / pi * 1e3 / pixres
    XY = np.zeros((nλ, 4, 2))

    if waffle_orient == "x":
        angles = np.radians([315.0, 225.0, 135.0, 45.0])
    elif waffle_orient == "+":
        angles = np.radians([270.0, 180.0, 90.0, 0.0])
    else:
        raise ValueError("Waffle orientation not recognized.")

    if all(im_center == np.array([145, 145])):
        angles += np.radians(-100.48)

    for λ in range(nλ):
        for k in range(4):
            XY[λ, k, :] = im_center + d[λ] * np.array(
                [np.cos(angles[k]), np.sin(angles[k])]
            )

    return XY


def extract_waffles(idx_frame, cube_im, waffles_pos, ROI):
    """
    Extracts four imagettes (containing the waffles) of size `ROI` x `ROI`
    at frame `idx_frame` and for each spectral channel of data cube `cube_im`.
    The approximate positions of the waffles `waffles_pos` should also be
    given (computed with function `waffles_approx_pos()`).
    """

    nλ = waffles_pos.shape[0]
    waffles = np.zeros((nλ, 4, ROI, ROI))

    for λ in range(nλ):
        # central_pos = np.round(waffles_pos[λ,:]).astype(int)
        central_pos = round_2_nearest_half(waffles_pos[λ])
        for w in range(4):
            nx_min = int(central_pos[w, 0] - (ROI // 2 - 0.5))
            nx_max = int(central_pos[w, 0] + (ROI // 2 + 0.5))
            ny_min = int(central_pos[w, 1] - (ROI // 2 - 0.5))
            ny_max = int(central_pos[w, 1] + (ROI // 2 + 0.5))
            waffles[λ, w, :, :] = cube_im[
                λ, idx_frame, nx_min:nx_max, ny_min:ny_max
            ]
    return waffles


def get_info_waffles(cube_im, band_filter, lam, wo, ROI, pixres, mask=None):
    """
    Extracts all waffles of the data cube `cube_im` and returns their fitted
    amplitudes and positions. The band filter `band_filter`, the wavelengths
    vector `lam`, the waffles orientation `wo` ('+', 'x'), the size of the
    region of interest `ROI` (in pixels) and the pixel resolution `pixres` are
    required. A 2-D mask can also be provided to account for bad pixels within
    the imagettes.
    """
    print("* Extracting and fitting the waffles...")

    nλ, nf, nx, ny = cube_im.shape
    center = image_center(nx, ny)
    waffle_pos = waffles_approx_pos(lam, center, wo, pixres)

    amp_w = np.zeros((nλ, nf))
    err_amp_w = np.zeros((nλ, nf))
    l_center_w = np.zeros((nλ, nf, 4, 2))
    err_l_center_w = np.zeros((nλ, nf, 4, 2))

    ##### PROGRESS BAR #####
    progress_bar = [int(i) for i in np.round(np.linspace(nf / 20, nf, 20))]
    cpt_progress = 0
    print("Progress - [" + "." * 20 + "] 0%", sep=" ", end="\r", flush=False)
    ########################

    for f in range(nf):
        ##### PROGRESS BAR #####
        if f in progress_bar:
            print(
                "Progress - ["
                + "#" * cpt_progress
                + "." * (20 - cpt_progress)
                + "] "
                + f"{cpt_progress*5}%",
                sep=" ",
                end="\r",
                flush=False,
            )
            cpt_progress += 1
            if cpt_progress == 19:
                print(
                    "Progress - [" + "#" * 20 + "] 100%",
                    sep=" ",
                    end="\n\n",
                    flush=False,
                )
        ########################

        w_f = extract_waffles(f, cube_im, waffle_pos, ROI)
        for λ in range(nλ):
            amp_w_f = np.zeros(4)
            err_amp_w_f = np.zeros(4)
            l_center_w_f = np.zeros((4, 2))
            err_l_center_w_f = np.zeros((4, 2))
            for i in range(4):
                if "BB" in band_filter:
                    angle = pi / 4 * (-1) ** i
                else:
                    angle = 0
                try:
                    im, p, perr = fit_Gaussian2D_withBound(
                        w_f[λ, i, :, :], theta=angle, mask=mask
                    )
                    amp_w_f[i], err_amp_w_f[i] = p[0], perr[0]
                    l_center_w_f[i, :] = p[1], p[2]
                    err_l_center_w_f[i, :] = perr[1], perr[2]
                    if err_amp_w_f[i] > amp_w_f[i] / 2:
                        amp_w_f[i], err_amp_w_f[i] = np.nan, np.nan
                        l_center_w_f[i, :] = np.nan
                        err_l_center_w_f[i, :] = np.nan
                except RuntimeError:
                    amp_w_f[i] = err_amp_w_f[i] = np.nan
                    l_center_w_f[i, :] = err_l_center_w_f[i, :] = np.nan
                except ValueError:
                    # print(f'ValueError --> frame:{f}, wvl:{λ}, waffle:{i}')
                    # raise
                    amp_w_f[i] = err_amp_w_f[i] = np.nan
                    l_center_w_f[i, :] = err_l_center_w_f[i, :] = np.nan

            if any(~np.isnan(amp_w_f)):
                amp_w[λ, f] = np.nanmean(amp_w_f)
            else:
                amp_w[λ, f] = np.nan

            if any(~np.isnan(err_amp_w_f)):
                err_amp_w[λ, f] = np.sqrt(
                    np.nansum(err_amp_w_f**2)
                ) / np.sum(~np.isnan(err_amp_w_f))
            else:
                err_amp_w[λ, f] = np.nan

            l_center_w[λ, f] = l_center_w_f
            err_l_center_w[λ, f] = err_l_center_w_f

    np.seterr(invalid="ignore")
    for λ in range(nλ):
        amp_w_σ = np.nanstd(amp_w[λ])
        amp_w_m = np.nanmedian(amp_w[λ])

        idx_2_remove = np.where(np.abs(amp_w[λ] - amp_w_m) > 3 * amp_w_σ)[0]
        amp_w[λ, list(idx_2_remove)] = np.nan
        err_amp_w[λ, list(idx_2_remove)] = np.nan
        l_center_w[λ, list(idx_2_remove)] = np.nan
        err_l_center_w[λ, list(idx_2_remove)] = np.nan
    np.seterr(invalid="warn")

    return (l_center_w, 3 * err_l_center_w, amp_w, 3 * err_amp_w, waffle_pos)


def create_circular_mask(h, w, center=None, radius=None):
    """
    Creates a circular mask of height `h`, width `w`. Optional arguments
    `center` (array of length = 2) and `radius` (in pixels) can be provided
    to specify the coordinates of the center of the circular mask and its size.
    """
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def Gaussian2D(xy, A, x0, y0, sig_x, sig_y, theta, offset, shape="vector"):
    """
    Evaluates at `xy` the 2-D anisotropic gaussian of amplitude `A`, centered
    at position (`x0`,`y0`), with standard deviation `sig_x` and `sig_y`,
    rotation angle `theta` and offset `offset`. Optional parameter `shape`
    specifies the format of the returned result ('vector' for 1-D array, and
                                                 'array' for 2-D array)
    """
    x, y = xy
    cosθ2, sinθ2 = cos(theta) ** 2, sin(theta) ** 2
    sin2θ = sin(2 * theta)
    αx, αy = 2 * sig_x**2, 2 * sig_y**2
    βx, βy = 2 * αx, 2 * αy

    a = cosθ2 / αx + sinθ2 / αy
    b = -sin2θ / βx + sin2θ / βy
    c = sinθ2 / αx + cosθ2 / αy

    res = (
        A
        * np.exp(
            -(
                a * (x - x0) ** 2
                + 2 * b * (x - x0) * (y - y0)
                + c * (y - y0) ** 2
            )
        )
        + offset
    )

    if shape == "vector":
        return res.ravel()
    elif shape == "array":
        return res


def fit_Gaussian2D_withBound(data2fit, theta=0, mask=None):
    """
    Fits a 2-D gaussian to the 2-D data `data2fit` and returns its parameters.
    A 2-D mask can also be provided to reject eventual outliers. The 2-D
    gaussian's rotation angle `theta` and the waffle orientation `w_orient`
    are used for the fit. A 2-D mask can also be provided to reject eventual
    outliers.
    """
    nx, ny = data2fit.shape

    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    y, x = np.meshgrid(x, y)

    if mask is not None:
        x2, y2 = x[mask], y[mask]
        data2 = data2fit[mask]
    else:
        x2, y2 = x, y
        data2 = data2fit.ravel()

    minA, maxA = np.min(data2), np.max(data2)
    minA = 0 if minA < 0 else minA

    initial_guess = (
        maxA * 0.8,
        nx // 2 - 0.5,
        ny // 2 - 0.5,
        1,
        1,
        theta,
        minA,
    )

    bounds_lower = (minA, 0, 0, 0, 0, -pi, 0)
    bounds_upper = (maxA, nx - 1, ny - 1, nx // 2, ny // 2, pi, maxA)

    popt, pcov = opt.curve_fit(
        Gaussian2D,
        (x2, y2),
        data2,
        p0=initial_guess,
        bounds=(bounds_lower, bounds_upper),
        maxfev=3000,
    )

    popterr = np.sqrt([i if i > 0 else np.nan for i in np.diag(pcov)])
    data_fitted = Gaussian2D((x, y), *popt).reshape(nx, ny)

    return (data_fitted, popt, popterr)


def shift_frame(image, shift_x, shift_y):
    """
    Shifts the `image` of `shift_y` pixels along the y-axis and `shift_y`
    pixels to the y-axis.
    """
    shift_val = (shift_x, shift_y)
    array_shifted = fourier_shift(np.fft.fftn(image), shift_val)
    array_shifted = np.fft.ifftn(array_shifted)
    return array_shifted.real


def shift_cube(cube, dist_pos):
    """
    Shifts all frames of the data cube `cube` along the x and y-axis by an
    amount encoded in the array `dist_pos`.

    """
    print("* Centering the cube...")

    nλ, nf, _, _ = cube.shape
    shifted_frames = np.ones(nf, dtype=int)

    ##### PROGRESS BAR #####
    progress_bar = [int(i) for i in np.round(np.linspace(nf / 20, nf, 20))]
    cpt_progress = 0
    print("Progress - [" + "." * 20 + "] 0%", sep=" ", end="\r", flush=False)
    ########################

    for f in range(nf):
        ##### PROGRESS BAR #####
        if f in progress_bar:
            print(
                "Progress - ["
                + "#" * cpt_progress
                + "." * (20 - cpt_progress)
                + "] "
                + f"{cpt_progress*5}%",
                sep=" ",
                end="\r",
                flush=False,
            )
            cpt_progress += 1
            if cpt_progress == 19:
                print(
                    "Progress - [" + "#" * 20 + "] 100%",
                    sep=" ",
                    end="\n\n",
                    flush=False,
                )
        ########################

        for λ in range(nλ):
            dx, dy = dist_pos[λ, f]
            if not any(np.isnan([dx, dy])):
                cube[λ, f] = shift_frame(cube[λ, f], dx, dy)
                # cube[λ,f] = shift_frame(cube[λ,f], dy, dx)
            else:
                shifted_frames[f] = 0
    return cube, shifted_frames


def centroid(arr):
    """
    Computes the centroid of the arr `arr`.
    """
    return np.mean(arr, axis=1)


def centroid_err(arr):
    """
    Propagates the error through centroid formula.
    """
    return np.sqrt(np.sum(arr**2, axis=1)) / np.sqrt(arr.shape[1])


def global_waffle_centers(centers, err_centers, waffle_central_pos, ROI):
    """
    Uses the local waffles centers `centers` and their associated errors
    `err_centers` to compute the global cube centers at all frames. The
    approximate waffles positions `waffle_central_pos` and the size of the
    region of interest `ROI` (in pixels) should be given.
    """
    nλ, nf, _, _ = centers.shape
    central_pos = round_2_nearest_half(waffle_central_pos)
    xy_beg = np.floor(central_pos - (ROI // 2 - 0.5))
    global_centers = np.zeros((nλ, nf, 2))
    err_global_centers = np.zeros((nλ, nf, 2))
    waffle_centers = np.zeros((nλ, nf, 4, 2))
    for f in range(nf):
        waffle_centers[:, f] = centers[:, f] + xy_beg
        global_centers[:, f] = centroid(waffle_centers[:, f])
        err_global_centers[:, f] = centroid_err(err_centers[:, f])
    return global_centers, err_global_centers, waffle_centers, err_centers


def global_dist_from_center(c, g_c, err_g_c):
    """
    Computes the shift (in x and y) between the centers estimated with the
    waffles `g_c` (with errors `err_g_c`) and the theoretical centers `c`.
    The absolute distances (cartesian distances) and their associated errors
    are also computed and returned.
    """
    dist = c - g_c
    abs_dist = np.sqrt(np.sum(dist**2, axis=2))
    err_abs_dist = np.sqrt(
        np.sum(dist**2 * err_g_c**2, axis=2) / abs_dist**2
    )
    return dist, abs_dist, err_abs_dist


def compute_centers_from_waffles(cube_im, hdr, lam):
    """
    Computes the centers (estimated with the waffles) of all frames and for
    all wavelengths `lam` of data cube `cube_im`. The header of the cube
    `hdr` should also be provided.
    """
    nλ, nf, nx, ny = cube_im.shape
    center_coord = image_center(nx, ny)

    # if 'ESO DPR TYPE' in hdr:
    #     waffle_flag = 'CENTER' in hdr['ESO DPR TYPE']
    # else:
    #     if 'PIPEFILE' in hdr:
    #         waffle_flag =  'CENTER_WAFFLE' in hdr['PIPEFILE']
    #     else:
    #         waffle_flag = True
    waffle_flag = True

    if waffle_flag:
        pixres = hdr["PIXTOARC"]
        wo = hdr["ESO OCS WAFFLE ORIENT"]
        band_filter = hdr["ESO INS COMB IFLT"]
        ROI = 24
        mask = create_circular_mask(ROI, ROI, radius=8)
        l_c_w, err_l_c_w, _, _, waffles_pos = get_info_waffles(
            cube_im, band_filter, lam, wo, ROI, pixres, mask=mask
        )

        print("* Computing the waffles centroids...\n")
        g_c, err_g_c, w_c, w_c_err = global_waffle_centers(
            l_c_w, err_l_c_w, waffles_pos, ROI
        )
        # breakpoint()

        dist, abs_dist, err_abs_dist = global_dist_from_center(
            center_coord, g_c, err_g_c
        )
        return (
            g_c,
            err_g_c,
            dist,
            err_g_c,
            abs_dist,
            err_abs_dist,
            w_c,
            w_c_err,
        )
    else:
        return None, None, None, None, None


def plot_distance_from_center_abs(pathin, pathout, abs_d, err_abs_d):
    """
    Plots the absolute distance `abs_d` (and errors `err_abs_d`) between the
    theoretical cube center and the centers computed with the waffles.
    The input path and output path (where to save the plot) should be provided.
    """
    _, hdr = get_master_cube(pathin)
    nλ, nf = abs_d.shape

    name = hdr["OBJECT"]
    date = hdr["DATE-OBS"]
    band = hdr["ESO INS COMB IFLT"]

    xrange = np.arange(0, nf)

    maxd = np.nanmax(np.abs(abs_d))

    pp = PdfPages(os.path.join(pathout, "dist_from_center_abs.pdf"))

    for λ in range(nλ):
        f, ax = plt.subplots(figsize=(12, 5))
        plt.suptitle(
            f"{name} / {band} / {date}", fontsize=18, fontweight="bold"
        )
        plt.errorbar(
            xrange,
            abs_d[λ],
            yerr=err_abs_d[λ],
            color="royalblue",
            linestyle="-",
            marker="o",
            markersize=4,
            capsize=4,
            linewidth=1.5,
            alpha=0.85,
        )

        plt.fill_between(
            [xrange[0] - 10, xrange[-1] + 10],
            [0, 0],
            [1 / 4, 1 / 4],
            color="green",
            alpha=0.15,
            linewidth=0,
        )
        plt.fill_between(
            [xrange[0] - 10, xrange[-1] + 10],
            [1 / 4, 1 / 4],
            [1 / 2, 1 / 2],
            color="darkorange",
            alpha=0.15,
            linewidth=0,
        )
        plt.fill_between(
            [xrange[0] - 10, xrange[-1] + 10],
            [1 / 2, 1 / 2],
            [1000, 1000],
            color="orangered",
            alpha=0.15,
            linewidth=0,
        )

        plt.xlabel("Frames", fontsize=17)
        plt.xticks(fontsize=15)

        plt.ylabel("Absolute distance from center [pix]", fontsize=17)
        if maxd < 1:
            yticks = [0, 1 / 4, 1 / 2, 3 / 4, 1]
            ylabels = ["0", "1/4", "1/2", "3/4", "1"]
            maxticks = 1
        else:
            maxticks = int(np.ceil(maxd))
            yticks = np.arange(0, maxticks + 1, 1)
            ylabels = yticks

        plt.yticks(ticks=yticks, labels=ylabels, fontsize=15)

        anchored_text = AnchoredText(
            "$\lambda_{" + str(λ) + "}$",
            loc="upper right",
            borderpad=0.0,
            frameon=True,
            prop=dict(fontweight="bold", fontsize=20),
        )
        ax.add_artist(anchored_text)

        plt.ylim(0, maxticks)
        plt.xlim(xrange[0] - 1, xrange[-1] + 1)

        plt.subplots_adjust(
            top=0.92,
            bottom=0.115,
            left=0.090,
            right=0.995,
            hspace=0.0,
            wspace=0.0,
        )

        pp.savefig(f)
    pp.close()
    plt.close()


def plot_distance_from_center_xy(pathin, pathout, d, err_d):
    """
    Plots the distance (in x an y) `d` (and errors `err_d`) between the
    theoretical cube center and the centers computed with the waffles.
    The input path and output path (where to save the plot) should be provided.
    """
    _, hdr = get_master_cube(pathin)
    nλ, nf, _ = d.shape

    name = hdr["OBJECT"]
    date = hdr["DATE-OBS"]
    band = hdr["ESO INS COMB IFLT"]

    maxxy = np.nanmax(np.abs(d))

    xrange = np.arange(0, nf)

    pp = PdfPages(os.path.join(pathout, "dist_from_center_xy.pdf"))

    for λ in range(nλ):
        f, ax = plt.subplots(figsize=(12, 5))
        plt.suptitle(
            f"{name} / {band} / {date}", fontsize=18, fontweight="bold"
        )
        plt.errorbar(
            xrange,
            d[λ, :, 0],
            yerr=err_d[λ, :, 0],
            color="royalblue",
            linestyle="-",
            marker="o",
            markersize=4,
            capsize=4,
            linewidth=1.5,
            alpha=0.6,
            label="$\Delta X$",
        )
        plt.errorbar(
            xrange,
            d[λ, :, 1],
            yerr=err_d[λ, :, 1],
            color="darkorange",
            linestyle="-",
            marker="o",
            markersize=4,
            capsize=4,
            linewidth=1.5,
            alpha=0.6,
            label="$\Delta Y$",
        )

        plt.fill_between(
            [xrange[0] - 10, xrange[-1] + 10],
            [-1 / 4, -1 / 4],
            [1 / 4, 1 / 4],
            color="green",
            alpha=0.15,
            linewidth=0,
        )
        plt.fill_between(
            [xrange[0] - 10, xrange[-1] + 10],
            [1 / 4, 1 / 4],
            [1 / 2, 1 / 2],
            color="darkorange",
            alpha=0.15,
            linewidth=0,
        )
        plt.fill_between(
            [xrange[0] - 10, xrange[-1] + 10],
            [-1 / 2, -1 / 2],
            [-1 / 4, -1 / 4],
            color="darkorange",
            alpha=0.15,
            linewidth=0,
        )
        plt.fill_between(
            [xrange[0] - 10, xrange[-1] + 10],
            [1 / 2, 1 / 2],
            [1000, 1000],
            color="orangered",
            alpha=0.15,
            linewidth=0,
        )
        plt.fill_between(
            [xrange[0] - 10, xrange[-1] + 10],
            [-1000, -1000],
            [-1 / 2, -1 / 2],
            color="orangered",
            alpha=0.15,
            linewidth=0,
        )

        plt.xlabel("Frames", fontsize=17)
        plt.xticks(fontsize=15)

        plt.ylabel("Distance from center\n(true-measured) [pix]", fontsize=17)

        if maxxy < 1 / 2:
            yticks = [-1 / 2, -1 / 4, 0, 1 / 4, 1 / 2]
            ylabels = ["-1/2", "-1/4", "0", "1/4", "1/2"]
            maxticks = 1 / 2
        else:
            maxticks = int(np.ceil(maxxy))
            yticks = np.arange(-maxticks, maxticks + 1, 1)
            ylabels = yticks

        plt.yticks(ticks=yticks, labels=ylabels, fontsize=15)

        anchored_text = AnchoredText(
            "$\lambda_{" + str(λ) + "}$",
            loc="upper right",
            borderpad=0.0,
            frameon=True,
            prop=dict(fontweight="bold", fontsize=20),
        )
        ax.add_artist(anchored_text)
        plt.legend(loc="lower right", fontsize=15)
        plt.xlim(xrange[0] - 1, xrange[-1] + 1)
        plt.ylim(-maxticks, maxticks)

        plt.subplots_adjust(
            top=0.92,
            bottom=0.115,
            left=0.090,
            right=0.995,
            hspace=0.0,
            wspace=0.0,
        )

        pp.savefig(f)
    pp.close()
    plt.close()


def merge_sortframes_vec(shifted_frames, sortframes_vec):
    """
    Merges the vector containing the actual shifted frames `shifted_frames`
    and the sortframes vector `sortframes_vec`.
    """
    if sortframes_vec is not None:
        if len(sortframes_vec.shape) == 2:
            sortframes_vec2 = np.zeros((sortframes_vec.shape))
            for k in range(sortframes_vec.shape[1]):
                sortframes_vec2[:, k] = shifted_frames * sortframes_vec[:, k]
            return sortframes_vec2
        elif len(sortframes_vec.shape) == 1:
            return shifted_frames * sortframes_vec
        else:
            return shifted_frames
    else:
        return shifted_frames


def formatTimeSeconds(t):
    """
    Converts a time `t` in a fancy and prettier format.
    """
    t_m, t_s = divmod(t, 60)
    t_h, t_m = divmod(t_m, 60)
    t_d, t_h = divmod(t_h, 24)

    t_s = str(round(t_s)).rjust(2, "0")
    t_m = str(round(t_m)).rjust(2, "0")
    t_h = str(round(t_h)).rjust(2, "0")
    t_d = str(round(t_d)).rjust(2, "0")

    if int(t_d) > 0:
        t_str = f"{t_d}d:{t_h}h:{t_m}m:{t_s}s"
    else:
        t_str = f"{t_h}h:{t_m}m:{t_s}s"

    return t_str


def built_in_sortframes_vector(c, err_c, thresh=0.5):
    nλ, nf, _ = c.shape
    vec = np.ones(nf, dtype=np.int16)
    if nλ == 2:  # IRDIS
        for f in range(0, nf):
            if (
                np.any(np.isnan(c[:, f, :]))
                or np.any(err_c[:, f, :] > thresh)
                or np.any(np.isnan(err_c[:, f, :]))
            ):
                vec[f] = 0
    elif nλ == 39:  # IFS
        vecX = np.ones(nf, dtype=np.int16)
        vecY = np.ones(nf, dtype=np.int16)
        for f in range(0, nf):
            if (
                np.sum(np.isnan(c[:, f, 0])) > nλ / 3
                or np.sum(err_c[:, f, 0] > thresh) > nλ / 3
                or np.sum(np.isnan(err_c[:, f, 0])) > nλ / 3
            ):
                vecX[f] = 0
            if (
                np.sum(np.isnan(c[:, f, 1])) > nλ / 3
                or np.sum(err_c[:, f, 1] > thresh) > nλ / 3
                or np.sum(np.isnan(err_c[:, f, 1])) > nλ / 3
            ):
                vecY[f] = 0
            vec = vecX * vecY
    return vec


def process_folder(path, save_new_cube=True):
    print("\n#######################################")
    print("#      Calibration using waffles      #")
    print("#######################################\n")

    print(f"* Starting calibration on directory:\n{os.path.basename(path)}\n")
    t0 = time.time()

    cube_im, hdr = get_master_cube(path)
    lam = get_lambda_info(path)
    sortframes_vec = get_sortframes_vector(path)

    (
        g_c,
        err_g_c,
        d,
        err_d,
        abs_d,
        err_abs_d,
        w_c,
        w_c_err,
    ) = compute_centers_from_waffles(cube_im, hdr, lam)

    # Inversion of the X and Y axis due to IDL being column-major
    d, err_d = d[:, :, [1, 0]], err_d[:, :, [1, 0]]
    g_c, err_g_c = g_c[:, :, [1, 0]], err_g_c[:, :, [1, 0]]

    if d is not None:
        path_res = os.path.join(path, "centering2")
        if not os.path.isdir(path_res):
            os.mkdir(path_res)

        # saving results
        print("* Saving results...\n")
        plot_distance_from_center_abs(path, path_res, abs_d, err_abs_d)
        plot_distance_from_center_xy(path, path_res, -d, err_d)
        # breakpoint()
        fits.writeto(
            os.path.join(path_res, "hdr.fits"),
            np.array([]),
            hdr,
            overwrite=True,
        )
        fits.writeto(
            os.path.join(path_res, "waffle_pos.fits"), w_c, overwrite=True
        )
        fits.writeto(
            os.path.join(path_res, "err_waffle_pos.fits"),
            w_c_err,
            overwrite=True,
        )
        fits.writeto(os.path.join(path_res, "dist_xy.fits"), d, overwrite=True)
        fits.writeto(
            os.path.join(path_res, "err_dist_xy.fits"), err_d, overwrite=True
        )
        fits.writeto(
            os.path.join(path_res, "abs_dist.fits"), abs_d, overwrite=True
        )
        fits.writeto(
            os.path.join(path_res, "err_abs_dist.fits"),
            err_abs_d,
            overwrite=True,
        )
        fits.writeto(
            os.path.join(path_res, "global_centers.fits"), g_c, overwrite=True
        )
        fits.writeto(
            os.path.join(path_res, "err_global_centers.fits"),
            err_g_c,
            overwrite=True,
        )

        if save_new_cube:
            sortframes_vec = built_in_sortframes_vector(g_c, err_g_c)
            shifted_cube, shifted_frames = shift_cube(cube_im, d)
            sortframes_vec = merge_sortframes_vec(
                shifted_frames, sortframes_vec
            )
            fits.writeto(
                os.path.join(path_res, "shifted_cube.fits"),
                np.array(shifted_cube, dtype=np.float32),
                hdr,
                overwrite=True,
            )
            fits.writeto(
                os.path.join(path_res, "new_sortframes_vec.fits"),
                sortframes_vec,
                overwrite=True,
            )
            fits.writeto(
                os.path.join(path_res, "shifted_frames_flag.fits"),
                shifted_frames,
                overwrite=True,
            )

        t1 = time.time()
        print(f"--> Elapsed time: {formatTimeSeconds(t1-t0)}\n")

    else:
        print("No waffles were found.\n")

    print("#######################################\n")


if __name__ == "__main__":
    t0 = time.time()

    idx = next(
        (i for i, arg in enumerate(sys.argv) if arg.startswith("dir=")), None
    )
    idx2 = next(
        (
            i
            for i, arg in enumerate(sys.argv)
            if arg.startswith("save_new_cube=")
        ),
        None,
    )

    if idx is None:
        print(
            "Error: Invalid command ! Script should be called as:"
            " python routine2.py dir=... save_new_cube=0/1 or"
            " python /home/sphere-data/WAFFLE_CENTERING/routine2.py dir=... save_new_cube=0/1"
        )
        sys.exit()
    else:
        path = sys.argv[idx].split("dir=")[1]

    if idx2 is None:
        save_new_cube = False
    else:
        save_new_cube = bool(int(sys.argv[idx2].split("save_new_cube=")[1]))

    if not os.path.isdir(path):
        print(f"Error: Path '{path}' is not a directory...")
        sys.exit()

    print("\n#######################################")
    print("#      Calibration using waffles      #")
    print("#######################################\n")

    print(f"* Starting calibration on directory:\n{os.path.basename(path)}\n")

    cube_im, hdr = get_master_cube(path)
    lam = get_lambda_info(path)
    sortframes_vec = get_sortframes_vector(path)

    (
        g_c,
        err_g_c,
        d,
        err_d,
        abs_d,
        err_abs_d,
        w_c,
        w_c_err,
    ) = compute_centers_from_waffles(cube_im, hdr, lam)

    # Inversion of the X and Y axis due to IDL being column-major
    d, err_d = d[:, :, [1, 0]], err_d[:, :, [1, 0]]
    g_c, err_g_c = g_c[:, :, [1, 0]], err_g_c[:, :, [1, 0]]

    if d is not None:
        path_res = os.path.join(path, "centering2")
        if not os.path.isdir(path_res):
            os.mkdir(path_res)

        # saving results
        print("* Saving results...\n")
        plot_distance_from_center_abs(path, path_res, abs_d, err_abs_d)
        plot_distance_from_center_xy(path, path_res, -d, err_d)
        fits.writeto(
            os.path.join(path_res, "hdr.fits"),
            np.array([]),
            hdr,
            overwrite=True,
        )
        fits.writeto(
            os.path.join(path_res, "waffle_pos.fits"), w_c, overwrite=True
        )
        fits.writeto(
            os.path.join(path_res, "err_waffle_pos.fits"),
            w_c_err,
            overwrite=True,
        )
        fits.writeto(os.path.join(path_res, "dist_xy.fits"), d, overwrite=True)
        fits.writeto(
            os.path.join(path_res, "err_dist_xy.fits"), err_d, overwrite=True
        )
        fits.writeto(
            os.path.join(path_res, "abs_dist.fits"), abs_d, overwrite=True
        )
        fits.writeto(
            os.path.join(path_res, "err_abs_dist.fits"),
            err_abs_d,
            overwrite=True,
        )
        fits.writeto(
            os.path.join(path_res, "global_centers.fits"), g_c, overwrite=True
        )
        fits.writeto(
            os.path.join(path_res, "err_global_centers.fits"),
            err_g_c,
            overwrite=True,
        )

        if save_new_cube:
            sortframes_vec = built_in_sortframes_vector(g_c, err_g_c)
            shifted_cube, shifted_frames = shift_cube(cube_im, d)
            sortframes_vec = merge_sortframes_vec(
                shifted_frames, sortframes_vec
            )
            fits.writeto(
                os.path.join(path_res, "shifted_cube.fits"),
                np.array(shifted_cube, dtype=np.float32),
                hdr,
                overwrite=True,
            )
            fits.writeto(
                os.path.join(path_res, "new_sortframes_vec.fits"),
                sortframes_vec,
                overwrite=True,
            )
            fits.writeto(
                os.path.join(path_res, "shifted_frames_flag.fits"),
                shifted_frames,
                overwrite=True,
            )

        t1 = time.time()
        print(f"--> Elapsed time: {formatTimeSeconds(t1-t0)}\n")

    else:
        print("No waffles were found.\n")

    print("#######################################\n")
