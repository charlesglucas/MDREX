import numpy as np


gaussian_sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
gaussian_fwhm_to_sigma = 1.0 / gaussian_sigma_to_fwhm


def fit_2d_gaussian(psf, verbose=False):
    from vip_hci.var import fit_2dgaussian
    from astropy.modeling import models

    df_params = fit_2dgaussian(
        psf, crop=True, cropsize=8, debug=False, full_output=True
    )
    psf_params = df_params.iloc[0].to_dict()
    gaussian_params = {
        "x_mean": psf_params["centroid_x"],
        "y_mean": psf_params["centroid_y"],
        "y_stddev": psf_params["fwhm_y"] * gaussian_fwhm_to_sigma,
        "x_stddev": psf_params["fwhm_x"] * gaussian_fwhm_to_sigma,
        "amplitude": psf_params["amplitude"],
        "theta": np.deg2rad(psf_params["theta"]),
    }
    H, W = psf.shape
    yy, xx = np.mgrid[:H, :W]
    y = yy.flatten()
    x = xx.flatten()
    rec = models.Gaussian2D.evaluate(x=x, y=y, **gaussian_params)
    rec = rec.reshape(H, W)
    return psf_params, rec
