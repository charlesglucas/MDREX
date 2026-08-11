import numpy as np
import scipy.optimize as opt
from abc import ABC, abstractmethod
from scipy.interpolate import RegularGridInterpolator


class FittablePSFModel(ABC):
    @abstractmethod
    def fit(self, psf):
        pass

    @abstractmethod
    def __call__(self, shift_x, shift_y):
        pass


class PSFInterpolator(FittablePSFModel):
    """Low level injection"""

    def __init__(self, method, size):
        """
        Args:
        -----
        * method: in [“linear”, “nearest”, “slinear”, “cubic”,
                “quintic”, “pchip”]
        * size: cropped size (int)
        """
        self.method = method
        # print(f"interpolation method: {self.method}")
        self.size = size

    def fit(self, psf, psf_size_inj):
        self.psf = psf
        self.psf_size_inj = psf_size_inj
        assert psf_size_inj <= self.psf.shape[-1], "psf_size_inj too large"

        self.x_grid = np.arange(self.psf_size_inj)
        self.y_grid = np.arange(self.psf_size_inj)

        self.yy, self.xx = np.mgrid[: self.psf_size_inj, : self.psf_size_inj]

    def __call__(self, shift_x, shift_y):
        assert np.abs(shift_x) < 1
        assert np.abs(shift_y) < 1

        interp = RegularGridInterpolator(
            (self.y_grid, self.x_grid),
            values=self.psf,
            method=self.method,
            bounds_error=False,
            fill_value=None,
        )
        new_val = interp((self.yy + shift_y, self.xx + shift_x))
        return new_val


class PSFGaussian2D(FittablePSFModel):
    """Low level injection"""

    def fit(self, psf, psf_size_inj):
        self.crop = psf.shape[-1]
        self.psf_size_inj = psf_size_inj
        # (10, 10)
        params = fit_gaussian_2d(psf)
        self.amplitude = params["amplitude"]
        self.sigma_x = params["sigma_x"]
        self.sigma_y = params["sigma_y"]
        self.theta = params["theta"]
        # self.offset = popt[6] * factor

        yy, xx = (
            np.mgrid[: self.psf_size_inj, : self.psf_size_inj]
            - self.psf_size_inj // 2
        )
        self.xy = (xx, yy)

        # breakpoint()

    def __call__(self, shift_x, shift_y):
        assert np.abs(shift_x) < 1
        assert np.abs(shift_y) < 1
        x0 = -shift_x
        y0 = -shift_y
        amplitude = self.amplitude
        offset = 0
        params = (
            amplitude,
            x0,
            y0,
            self.sigma_x,
            self.sigma_y,
            self.theta,
            offset,
        )
        new_psf = gaussian_2d(self.xy, *params)
        new_psf = new_psf.reshape(self.psf_size_inj, self.psf_size_inj)
        # breakpoint()

        assert new_psf.shape[-1] == self.psf_size_inj
        return new_psf


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    x0 = float(x0)
    y0 = float(y0)
    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )
    g = offset + amplitude * np.exp(
        -(
            a * ((x - x0) ** 2)
            + 2 * b * (x - x0) * (y - y0)
            + c * ((y - y0) ** 2)
        )
    )
    return g.ravel()


def fit_gaussian_2d(psf):
    assert psf.ndim == 2
    assert psf.shape[-1] % 2 == 0
    sub_psf = psf[1:, 1:]
    # (9, 9)

    psf_size = sub_psf.shape[-1]
    yy, xx = np.mgrid[:psf_size, :psf_size]
    xy_sub = (xx, yy)

    # normalize for optimization
    factor = np.max(sub_psf)
    sub_psf = sub_psf / factor
    initial_guess = (
        1,
        psf_size // 2,
        psf_size // 2,
        1.5,
        1.5,
        0,
        0,
    )
    if False:
        import matplotlib.pyplot as plt

        data = gaussian_2d((xx, yy), *initial_guess)
        plt.imshow(data.reshape(9, 9))
        plt.show()

    popt, pcov = opt.curve_fit(
        gaussian_2d, xy_sub, sub_psf.ravel(), p0=initial_guess
    )
    params = {
        "amplitude": popt[0] * factor,
        "x0": popt[1],
        "y0": popt[2],
        "sigma_x": popt[3],
        "sigma_y": popt[4],
        "theta": popt[5],
        "offset": popt[6] * factor,
    }
    return params
