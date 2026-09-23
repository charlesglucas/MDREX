import numpy as np


def contrast_curve_sigma(
    sigma_map,
    threshold=5,
    width_pix=5,
    min_as=0.1,
    max_as=1.4,
    pix2as=0.01225,
    n_points=100,
):
    H, W = sigma_map.shape

    yy, xx = np.mgrid[:H, :W] - H // 2
    rr = np.sqrt(yy**2 + xx**2)
    all_r = np.linspace(min_as / pix2as, max_as / pix2as, num=n_points)

    all_contrast = []
    all_sep_as = []
    for r in all_r:
        mask = (rr <= r + width_pix / 2) & (rr > r - width_pix / 2)
        sigma_mean = np.mean(sigma_map[mask])
        contrast = threshold * sigma_mean
        sep_as = r * pix2as
        all_contrast.append(contrast)
        all_sep_as.append(sep_as)

    return np.array(all_sep_as), np.array(all_contrast)
