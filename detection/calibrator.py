import numpy as np
import scipy.stats as stats
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PieceWiseRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, x):
        shape = x.shape
        x = x.reshape(-1, 1)
        # ndim = x.ndim
        # x = x[None, ...]
        x_ref = self.x.reshape(1, -1)
        # y_ref = self.y.reshape(1, -1)
        dist = np.abs(x - x_ref)
        idx = np.argsort(dist, axis=1)
        idx0 = idx[:, 0]
        idx1 = idx[:, 1]
        y0 = self.y[idx0]
        y1 = self.y[idx1]
        x0 = self.x[idx0]
        x1 = self.x[idx1]

        a = (y1 - y0) / (x1 - x0)
        b = (y0 * x1 - y1 * x0) / (x1 - x0)

        y = a * x.flatten() + b
        y = y.reshape(shape)
        return y


class Calibrator:
    def __init__(self, path_calib, discard_quantile=1e-5, plot=False):
        self.path_calib = path_calib
        self.discard_quantile = discard_quantile

        path_calib = to_absolute_path(self.path_calib)
        calib = torch.load(path_calib)
        thresholds = calib["bins"][:-1]
        counts = calib["counts"]
        n_pix_tot = calib["n_pix_tot"]
        p_sup = counts / n_pix_tot

        mask_quantile = p_sup > self.discard_quantile
        mask_zero = (p_sup > 0) & (p_sup < 1)
        mask = mask_quantile & mask_zero
        p_sup = p_sup[mask]
        thresholds = thresholds[mask]
        p_cumul = 1 - p_sup
        thresh_gauss = stats.norm.ppf(p_cumul)
        print(f"{np.sum(mask_quantile)=}")

        # self.spl = UnivariateSpline(thresholds, thresh_gauss, s=1, k=1)
        # tck = interpolate.splrep(thresholds, thresh_gauss, k=2, s=0)
        self.piecewise_linear = PieceWiseRegression(
            x=thresholds, y=thresh_gauss
        )

        # self.p, self.e = optimize.curve_fit(
        # piecewise_linear, thresholds, thresh_gauss
        # )
        # dev_2 = interpolate.splev(thresholds, tck, der=2)
        if plot:
            # thresh_new = thresholds[30:] * 1.5 - 1
            thresh_new = np.linspace(-1, 1, 100)
            interpolated = self.piecewise_linear(thresh_new)
            breakpoint()
            plt.plot(thresholds, thresh_gauss)
            plt.scatter(thresholds, thresh_gauss)
            plt.plot(thresh_new, interpolated, c="r")
            plt.show()

    def __call__(self, x):
        return self.piecewise_linear(x)
