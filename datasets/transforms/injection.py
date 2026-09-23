import torch
import copy
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from abc import ABC, abstractmethod
from utils.prefix_log import PrefixLogMixin
import torch.nn as nn
from utils.viz import cube_3d_viewer

# from vip_hci.var import fit_2dgaussian
import scipy.optimize as opt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Injector2D:
    def __init__(self):
        pass


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


def gaussian_2d_grad(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
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
    delta_x = x - x0
    delta_y = y - y0
    exp = np.exp(
        -(a * (delta_x**2) + 2 * b * delta_x * delta_y + c * (delta_y**2))
    ).ravel()
    # (ps, ps)
    g = offset + amplitude * exp
    grad = []

    # 1. d alpha h / d alpha = h
    grad_alpha = amplitude * exp

    # 2. d alpha h / d x0
    pre_x = -2 * a * delta_x - 2 * b * delta_y
    pre_y = -2 * c * delta_y - 2 * b * delta_x
    grad_x0 = amplitude * pre_x.ravel() * exp
    grad_y0 = amplitude * pre_y.ravel() * exp

    grad = np.array([grad_alpha, grad_y0, grad_x0])
    # (3, ps ** 2)

    return g.ravel(), grad


class BatchPSFSampler(nn.Module):
    def __init__(self, psf_size, mode):
        super().__init__()
        self.psf_size = psf_size
        assert mode in ["centered", "all_pixels", "shift"]
        self.mode = mode
        # self.return_grad = return_grad

        yy, xx = np.mgrid[: self.psf_size, : self.psf_size]
        self.yy = torch.tensor(yy)[None, ...].to(device)
        self.xx = torch.tensor(xx)[None, ...].to(device)
        # (1, h, w)
        if self.mode == "all_pixels":
            x0 = (
                torch.arange(self.psf_size)[None, ...]
                .expand(self.psf_size, -1)
                .reshape(-1, 1, 1)
                .float()
                .to(device)
            )
            y0 = (
                torch.arange(self.psf_size)[..., None]
                .expand(-1, self.psf_size)
                .reshape(-1, 1, 1)
                .float()
                .to(device)
            )
        elif self.mode == "centered":
            x0 = self.psf_size // 2
            y0 = self.psf_size // 2
        elif self.mode == "shift":
            return
        else:
            raise ValueError
        self.delta_x = self.xx - x0
        self.delta_y = self.yy - y0
        # (ps * ps, ps, ps)

    def forward(
        self,
        amplitude,
        sigma_x,
        sigma_y,
        theta,
        offset,
        x0=None,
        y0=None,
        return_grad=False,
    ):
        """
        Args:
        ----
            amplitude: (bs,) or (1,)
            x0: (bs,) or (1,)
            y0: (bs,) or (1,)
            sigma_x: (bs,) or (1,)
            sigma_y: (bs,) or (1,)
            theta: (bs,) or (1,)

        Returns:
        -------
            psf: (bs, h * w)
        """
        # amplitude = amplitude.view(-1, 1, 1)
        # sigma_x = sigma_x.view(-1, 1, 1)
        # sigma_y = sigma_y.view(-1, 1, 1)
        # theta = theta.view(-1, 1, 1)
        # offset = offset.view(-1, 1, 1)
        if self.mode == "shift":
            assert self.xx.ndim == 3
            assert x0 is not None
            assert y0 is not None
            delta_x = self.xx - x0.view(-1, 1, 1)
            delta_y = self.yy - y0.view(-1, 1, 1)
        else:
            delta_x = self.delta_x
            delta_y = self.delta_y
            # (bs, h, w)

        a = (torch.cos(theta) ** 2) / (2 * sigma_x**2) + (
            torch.sin(theta) ** 2
        ) / (2 * sigma_y**2)
        b = -(torch.sin(2 * theta)) / (4 * sigma_x**2) + (
            torch.sin(2 * theta)
        ) / (4 * sigma_y**2)
        c = (torch.sin(theta) ** 2) / (2 * sigma_x**2) + (
            torch.cos(theta) ** 2
        ) / (2 * sigma_y**2)
        # breakpoint()
        # print(f"{a=}")
        # print(f"{b=}")
        # print(f"{c=}")

        exp = torch.exp(
            -(a * delta_x**2 + 2 * b * delta_x * delta_y + c * delta_y**2)
        )
        all_psf = offset + amplitude * exp
        # (bs, h, w)
        if return_grad:
            # the grad must be return w.r.t. x_t, y_t, OK !
            # (bs, h, w, 2)
            assert self.mode == "shift"

            pre_x = -2 * a * delta_x - 2 * b * delta_y
            pre_y = -2 * c * delta_y - 2 * b * delta_x
            grad_x0 = amplitude * pre_x * exp
            grad_y0 = amplitude * pre_y * exp
            # (bs, h, w)
            # breakpoint()

            return all_psf, grad_y0, grad_x0

        return all_psf


def batch_gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    Args:
    ----
        xy: (x, y) (h, w)
        amplitude: (bs,) or (1,)
        x0: (bs,) or (1,)
        y0: (bs,) or (1,)
        sigma_x: (bs,) or (1,)
        sigma_y: (bs,) or (1,)
        theta: (bs,) or (1,)

    Returns:
    -------
        psf: (bs, h * w)
    """
    breakpoint()
    x, y = xy
    assert x.ndim == 2
    assert y.ndim == 2
    # x0 = float(x0)
    # y0 = float(y0)
    a = (torch.cos(theta) ** 2) / (2 * sigma_x**2) + (
        torh.sin(theta) ** 2
    ) / (2 * sigma_y**2)
    b = -(torch.sin(2 * theta)) / (4 * sigma_x**2) + (
        torch.sin(2 * theta)
    ) / (4 * sigma_y**2)
    c = (torch.sin(theta) ** 2) / (2 * sigma_x**2) + (
        torch.cos(theta) ** 2
    ) / (2 * sigma_y**2)
    g = offset + amplitude * torch.exp(
        -(
            a * ((x - x0) ** 2)
            + 2 * b * (x - x0) * (y - y0)
            + c * ((y - y0) ** 2)
        )
    )
    return g.flatten()


def batch_gaussian_2d(
    xy,
    amplitude,
    shift_x,
    shift_y,
    sigma_x,
    sigma_y,
    theta,
    offset,
):
    """
    Args:
    -----
    * xy: torch.tensor(2, ps, ps)
    * amplitude: float
    * sigma_x: float
    * sigma_y: float
    * theta: float
    * offset: float
    * x0: (bs,)
    * y0: (bs,)

    Returns:
    --------
    * (bs, ps, ps)
    """
    xx, yy = xy
    # x0 = float(x0)
    # y0 = float(y0)
    xx = xx.view(1, -1)
    yy = yy.view(1, -1)
    # (1, K)

    shift_x = shift_x.view(-1, 1)
    shift_y = shift_y.view(-1, 1)
    # (M, 1)

    a = (np.cos(theta) ** 2) / (2 * sigma_x**2) + (np.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (np.sin(theta) ** 2) / (2 * sigma_x**2) + (np.cos(theta) ** 2) / (
        2 * sigma_y**2
    )

    g = offset + amplitude * torch.exp(
        -(
            a * ((xx - shift_x) ** 2)
            + 2 * b * (xx - shift_x) * (yy - shift_y)
            + c * ((yy - shift_y) ** 2)
        )
    )
    # (M, K)
    return g


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


class PSFGaussian2D(FittablePSFModel):
    """Low level injection"""

    def fit(self, psf, psf_size_inj):
        self.crop = psf.shape[-1]
        self.psf_size_inj = psf_size_inj
        # (10, 10)
        assert psf.ndim == 2
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


def compute_trajectory_simple(x0, y0, t0, rho, theta, T, rot, patch_size):
    """
    Compute local positions of a sources following a rotation

    Args:
    -----
    * x0, y0: local position of the reference position in the patch (float)
    * t0: reference time frame (int)
    * rho: global position of the integer center of the
           patch (radius, in pixels)
    * theta: global position of the integer center of the
             patch (angle, in RADIANS)
    * T: number of frames (int)
    * rot: rotation vector, (in DEGREES) np.array(T)
    * patch_size: size of the frame (int), usually 256


    Returns:
    --------
    * x_loc_t: local cartesian x, np.array(T)
    * y_loc_t: local cartesian y, np.array(T)

    Remarks:
    --------
    * O is the rotational/integer center of the frame
    * C is the integer center of the patch
    * P is the center of the pixel in the upper left corner of the patch
    * S is the position of the center of the source at time 0
    * St is the position of the center of the source at time t
    """
    assert rot.ndim == 1
    # print(f"{rho=}")

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2).astype(np.float32)
        theta = np.arctan2(y, x).astype(np.float32)
        return rho, theta

    def pol2cart(rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    PS_x = x0
    PS_y = y0

    PC = patch_size // 2
    # typically +16.0

    CP = -PC

    CS_x = CP + PS_x
    CS_y = CP + PS_y

    OC_x, OC_y = pol2cart(rho, theta)

    OS_x = OC_x + CS_x
    OS_y = OC_y + CS_y

    rho_0, theta_0 = cart2pol(OS_x, OS_y)

    rot_rad = np.deg2rad(rot - rot[t0])

    theta_t = theta_0 + rot_rad

    OSt_x, OSt_y = pol2cart(rho_0, theta_t)

    CSt_x = OSt_x - OC_x
    CSt_y = OSt_y - OC_y

    PSt_x = PC + CSt_x
    PSt_y = PC + CSt_y
    # print(f"{PSt_x=}")

    return PSt_x, PSt_y


def compute_trajectory(x0, y0, t0, rho, theta, T, rot, patch_size):
    """
    Compute local positions of a sources following a rotation

    Args:
    -----
    * x0, y0: local position of the reference position in the patch (float)
    * t0: reference time frame (int)
    * rho: global position of the integer center of the
           patch (radius, in pixels)
    * theta: global position of the integer center of the
             patch (angle, in RADIANS)
    * T: number of frames (int)
    * rot: rotation vector, (in DEGREES) np.array(T)
    * patch_size: size of the frame (int), usually 256


    Returns:
    --------
    * x_loc_t: local cartesian x, np.array(T)
    * y_loc_t: local cartesian y, np.array(T)

    Remarks:
    --------
    * O is the rotational/integer center of the frame
    * C is the integer center of the patch
    * P is the center of the pixel in the upper left corner of the patch
    * S is the position of the center of the source at time 0
    * St is the position of the center of the source at time t
    """
    assert rot.ndim == 1
    # print(f"{rho=}")

    def cart2pol(x, y):
        rho = np.sqrt(x**2 + y**2).astype(np.float32)
        theta = np.arctan2(y, x).astype(np.float32)
        return rho, theta

    def pol2cart(rho, theta):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y

    PS_x = x0
    PS_y = y0

    PC = patch_size // 2
    # typically +16.0

    CP = -PC

    CS_x = CP + PS_x
    CS_y = CP + PS_y

    OC_x, OC_y = pol2cart(rho, theta)

    OS_x = OC_x + CS_x
    OS_y = OC_y + CS_y

    rho_0, theta_0 = cart2pol(OS_x, OS_y)

    rot_rad = np.deg2rad(rot - rot[t0])

    theta_t = theta_0 + rot_rad

    OSt_x, OSt_y = pol2cart(rho_0, theta_t)

    CSt_x = OSt_x - OC_x
    CSt_y = OSt_y - OC_y

    PSt_x = PC + CSt_x
    PSt_y = PC + CSt_y
    # print(f"{PSt_x=}")

    return PSt_x, PSt_y


class Injector3D(PrefixLogMixin):
    """Injects multiple sources with a given PSF in a full frame of
    arbitrary size"""

    # def __init__(self, size, T, crop, method, debug=False):
    def __init__(
        self,
        method,
        ensure_key_error,
        scale,
        # n_channels,
        psf_model,
        psf_size_inj=None,
        debug=False,
    ):
        assert psf_model in ["interpolation", "gaussian_2d"]
        # self.psf_model = psf_model
        # self.n_channels = n_channels
        self.psf_model = psf_model
        self.log(f"{self.psf_model=}")
        self.method = method
        self.ensure_key_error = ensure_key_error
        self.debug = debug
        self.scale = float(scale)
        self.psf_size_inj = psf_size_inj

        self.initialized = False

    def init(self, T, C, size, crop):
        self.crop = crop
        self.size = size
        self.T = T
        if self.psf_size_inj is None:
            self.psf_size_inj = self.crop
        # assert C == self.n_channels
        # print(f"{self.psf_size_inj=}")
        assert self.psf_size_inj % 2 == 0
        self.size_ext = self.size + 2 * self.psf_size_inj
        # self.size_ext = self.size
        self.obj_ext = np.zeros(
            (C, self.T, self.size_ext, self.size_ext),
            dtype=np.float32,
        )
        # self.psf_model =
        # self.psf_model = PSFInterpolator(method=self.method, size=self.crop)
        if self.psf_model == "gaussian_2d":
            fittable_psf_model = PSFGaussian2D()
        elif self.psf_model == "interpolation":
            fittable_psf_model = PSFInterpolator(
                method=self.method, size=self.crop
            )
        else:
            raise ValueError()
        self.fittable_psf_models = [
            copy.deepcopy(fittable_psf_model) for _ in range(C)
        ]
        self.initialize = True

    # def __call__(
    # self, rho, theta, n_sources, all_coords, all_t0, all_alpha, psf, rot
    # ):
    def __call__(self, item):
        psf = item["psf"]
        s_0 = item["s_0"]
        rot = item["rot"]
        # (C, T, H, W)
        C, T, H, W = s_0.shape
        C_psf, h, w = psf.shape
        assert C_psf == C
        assert s_0.ndim == 4, f"{s_0.shape}"
        # assert s_0.shape[1] == self.n_channels

        # if C != self.n_channels:
        # print(f"Wrong number of channels: {self.n_channels=}")
        # breakpoint()
        # raise ValueError

        if not self.initialized:
            self.init(
                T=rot.shape[-1],
                C=C,
                size=s_0.shape[-1],
                crop=psf.shape[-1],
            )

        all_alpha = item["alpha"]
        all_coords = item["coords"]
        n_sources = item["n_sources"]
        # channels = item["channels"]

        try:
            all_t0 = item["t0"]
            rho = item["rho"]
            theta = item["theta"]
        except KeyError:
            assert not self.ensure_key_error
            all_t0 = np.zeros(n_sources, dtype=int)
            # coordinates of the center of the patch
            rho = 0.0
            theta = 0.0

        obj_ext = self.obj_ext.copy()
        # (T, C, H, W)
        if self.debug:
            print(f"{all_alpha=}")
            print(f"{n_sources=}")
            print(f"{all_coords=}")
            print(f"{all_t0=}")
            print(f"{n_sources=}")
            print(f"{channels=}")

        for c in range(C):
            self.fittable_psf_models[c].fit(
                psf[c], psf_size_inj=self.psf_size_inj
            )
        for i in range(n_sources):
            # coords of the center of the PSF
            y0 = all_coords[i, 0] + self.crop // 2
            x0 = all_coords[i, 1] + self.crop // 2

            t0 = all_t0[i]
            # channel = channels[i]
            alpha = all_alpha[i]

            # trajectory of the CENTER of the cropped PSF, i.e. PSt
            x_t, y_t = compute_trajectory(
                x0=x0,
                y0=y0,
                t0=t0,
                rho=rho,
                theta=theta,
                T=self.T,
                rot=rot,
                patch_size=self.size,
            )
            if self.debug:
                import matplotlib.pyplot as plt

                plt.scatter(x_t, y_t)
                plt.gca().invert_yaxis()
                plt.show()

            # inj_size = self.crop
            for c in range(C):
                for t in range(self.T):
                    # coordinates of the center of the psf in the patch
                    y = y_t[t]
                    x = x_t[t]

                    # coordinates of the upper left corner of the psf
                    # in the extended patch
                    # x_ext = x - self.crop // 2 + self.crop
                    # y_ext = y - self.crop // 2 + self.crop
                    x_ext = x - self.psf_size_inj // 2 + self.psf_size_inj
                    y_ext = y - self.psf_size_inj // 2 + self.psf_size_inj

                    # discard out of view
                    if (
                        (x_ext < 0)
                        or (x_ext > self.size + self.psf_size_inj)
                        or (y_ext < 0)
                        or (y_ext > self.size + self.psf_size_inj)
                    ):
                        continue

                    x_anchor = int(x_ext)
                    y_anchor = int(y_ext)

                    shift_x = x_ext - x_anchor
                    shift_y = y_ext - y_anchor

                    new_psf = self.fittable_psf_models[c](
                        shift_x=-shift_x, shift_y=-shift_y
                    )
                    obj_ext[
                        c,
                        t,
                        y_anchor : y_anchor + self.psf_size_inj,
                        x_anchor : x_anchor + self.psf_size_inj,
                    ] += (
                        new_psf * alpha
                    )
        obj = obj_ext[
            :,
            :,
            self.psf_size_inj : -self.psf_size_inj,
            self.psf_size_inj : -self.psf_size_inj,
        ].copy()
        # obj = obj_ext.copy()
        # if self.debug:
        if False:
            import matplotlib.pyplot as plt
            from utils.viz import cube_3d_viewer

            print(obj_ext.shape)
            # cube_3d_viewer(1e5 * obj_ext[0])
            cube_3d_viewer(obj_ext[0])

            # plt.imshow(np.mean(obj_ext, axis=1)[0])

            # plt.show()
            breakpoint()

        y = s_0 + obj * self.scale
        assert y.shape == s_0.shape
        item["y"] = y
        item["obj"] = obj * self.scale
        item["coords"] += self.crop // 2

        # breakpoint()
        return item


class Injector:
    def __init__(self, psf_model="gaussian_2d"):
        self.initialized = False
        self.psf_size_inj = None

        self.psf_model = psf_model
        # if self.psf_model == "gaussian_2d":
        # self.fittable_psf_model = PSFGaussian2D()
        # elif self.psf_model == "interpolation":
        # self.fittable_psf_model = PSFInterpolator(
        # method=self.method, size=self.crop
        # )
        # else:
        # raise ValueError()

    def init(self, T, C, size, crop):
        self.crop = crop
        self.size = size
        self.T = T
        if self.psf_size_inj is None:
            self.psf_size_inj = self.crop
        # print(f"{self.psf_size_inj=}")
        assert self.psf_size_inj % 2 == 0
        self.size_ext = self.size + 2 * self.psf_size_inj
        # self.size_ext = self.size
        self.obj_ext = np.zeros(
            (C, self.T, self.size_ext, self.size_ext),
            dtype=np.float32,
        )
        # self.psf_model =
        # self.psf_model = PSFInterpolator(method=self.method, size=self.crop)
        if self.psf_model == "gaussian_2d":
            fittable_psf_model = PSFGaussian2D()
        elif self.psf_model == "interpolation":
            fittable_psf_model = PSFInterpolator(
                method=self.method, size=self.crop
            )
        else:
            raise ValueError()
        self.fittable_psf_models = [
            copy.deepcopy(fittable_psf_model) for _ in range(C)
        ]
        self.initialize = True

    def __call__(self, psf, rot, coords, all_alpha):
        assert rot.ndim == 2, f"{rot.shape=}"
        assert psf.ndim == 4, f"{psf.shape=}"
        assert coords.ndim == 3, f"{coords.shape=}"
        assert all_alpha.ndim == 2, f"{all_alpha.shape=}"
        rot = rot[0].cpu().numpy()
        psf = psf[0].cpu().numpy()
        coords = coords[0].cpu().numpy()
        all_alpha = all_alpha[0].cpu().numpy()
        C, h, w = psf.shape

        if not self.initialized:
            self.init(
                T=rot.shape[-1],
                C=C,
                size=256,
                crop=psf.shape[-1],
            )

        for c in range(C):
            self.fittable_psf_models[c].fit(
                psf[c], psf_size_inj=self.psf_size_inj
            )

        n_sources = coords.shape[0]

        obj_ext = self.obj_ext.copy()
        # (T, C, H, W)

        for i in range(n_sources):
            # coords of the center of the PSF
            # y0 = coords[i, 0] + self.crop // 2
            # x0 = coords[i, 1] + self.crop // 2

            y0 = coords[i, 0]
            x0 = coords[i, 1]

            t0 = 0
            # channel = 0
            alpha = all_alpha[i]
            rho = 0.0
            theta = 0.0

            x_t, y_t = compute_trajectory(
                x0=x0,
                y0=y0,
                t0=t0,
                rho=rho,
                theta=theta,
                T=self.T,
                rot=rot,
                patch_size=self.size,
            )
            if False:
                import matplotlib.pyplot as plt

                plt.scatter(x_t, y_t)
                plt.gca().invert_yaxis()
                plt.show()

            # inj_size = self.crop
            for c in range(C):
                for t in range(self.T):
                    # coordinates of the center of the psf in the patch
                    y = y_t[t]
                    x = x_t[t]

                    # coordinates of the upper left corner of the psf
                    # in the extended patch
                    # x_ext = x - self.crop // 2 + self.crop
                    # y_ext = y - self.crop // 2 + self.crop
                    x_ext = x - self.psf_size_inj // 2 + self.psf_size_inj
                    y_ext = y - self.psf_size_inj // 2 + self.psf_size_inj

                    # discard out of view
                    if (
                        (x_ext < 0)
                        or (x_ext > self.size + self.psf_size_inj)
                        or (y_ext < 0)
                        or (y_ext > self.size + self.psf_size_inj)
                    ):
                        continue

                    x_anchor = int(x_ext)
                    y_anchor = int(y_ext)

                    shift_x = x_ext - x_anchor
                    shift_y = y_ext - y_anchor

                    new_psf = self.fittable_psf_models[c](
                        shift_x=-shift_x, shift_y=-shift_y
                    )
                    obj_ext[
                        c,
                        t,
                        y_anchor : y_anchor + self.psf_size_inj,
                        x_anchor : x_anchor + self.psf_size_inj,
                    ] += (
                        new_psf * alpha
                    )
        obj = obj_ext[
            :,
            :,
            self.psf_size_inj : -self.psf_size_inj,
            self.psf_size_inj : -self.psf_size_inj,
        ].copy()
        return obj


class Injector2:
    def __init__(self, n_channels, psf_size, psf_model="gaussian_2d"):
        self.psf_size = psf_size
        self.initialized = False
        self.n_channels = n_channels
        # self.psf_size_inj = None

        self.psf_model = psf_model
        if self.psf_model == "gaussian_2d":
            self.fittable_psf_model = PSFGaussian2D()
        elif self.psf_model == "interpolation":
            self.fittable_psf_model = PSFInterpolator(
                method=self.method, size=self.crop
            )
        else:
            raise ValueError()

    def fit_psf(self, psf):
        self.fittable_psf_model.fit(psf, psf_size_inj=self.psf_size)

    def init(self, T, size):
        self.size = size
        self.T = T
        # print(f"{self.psf_size_inj=}")
        assert self.psf_size % 2 == 0
        self.size_ext = self.size + 2 * self.psf_size
        # self.size_ext = self.size
        self.obj_ext = np.zeros(
            (self.T, self.n_channels, self.size_ext, self.size_ext),
            dtype=np.float32,
        )
        # self.psf_model =
        # self.psf_model = PSFInterpolator(method=self.method, size=self.crop)
        # if self.psf_model == "gaussian_2d":
        # self.fittable_psf_model = PSFGaussian2D()
        # elif self.psf_model == "interpolation":
        # self.fittable_psf_model = PSFInterpolator(
        # method=self.method, size=self.crop
        # )
        # else:
        # raise ValueError()
        self.initialize = True

    def __call__(self, rot, coords, all_alpha):
        assert rot.ndim == 2, f"{rot.shape=}"
        assert coords.ndim == 3, f"{coords.shape=}"
        assert all_alpha.ndim == 2, f"{all_alpha.shape=}"
        rot = rot[0].cpu().numpy()
        # psf = psf[0].cpu().numpy()
        coords = coords[0].cpu().numpy()
        all_alpha = all_alpha[0].cpu().numpy()

        if not self.initialized:
            self.init(
                T=rot.shape[-1],
                size=256,
            )

        n_sources = coords.shape[0]

        obj_ext = self.obj_ext.copy()
        # (T, C, H, W)

        for i in range(n_sources):
            # coords of the center of the PSF
            # y0 = coords[i, 0] + self.crop // 2
            # x0 = coords[i, 1] + self.crop // 2

            y0 = coords[i, 0]
            x0 = coords[i, 1]

            t0 = 0
            channel = 0
            alpha = all_alpha[i]
            rho = 0.0
            theta = 0.0

            x_t, y_t = compute_trajectory(
                x0=x0,
                y0=y0,
                t0=t0,
                rho=rho,
                theta=theta,
                T=self.T,
                rot=rot,
                patch_size=self.size,
            )
            if False:
                import matplotlib.pyplot as plt

                plt.scatter(x_t, y_t)
                plt.gca().invert_yaxis()
                plt.show()

            # inj_size = self.crop
            for t in range(self.T):
                # coordinates of the center of the psf in the patch
                y = y_t[t]
                x = x_t[t]

                # coordinates of the upper left corner of the psf
                # in the extended patch
                # x_ext = x - self.crop // 2 + self.crop
                # y_ext = y - self.crop // 2 + self.crop
                x_ext = x - self.psf_size // 2 + self.psf_size
                y_ext = y - self.psf_size // 2 + self.psf_size

                # discard out of view
                if (
                    (x_ext < 0)
                    or (x_ext > self.size + self.psf_size)
                    or (y_ext < 0)
                    or (y_ext > self.size + self.psf_size)
                ):
                    continue

                x_anchor = int(x_ext)
                y_anchor = int(y_ext)

                shift_x = x_ext - x_anchor
                shift_y = y_ext - y_anchor

                new_psf = self.fittable_psf_model(
                    shift_x=-shift_x, shift_y=-shift_y
                )
                obj_ext[
                    t,
                    channel,
                    y_anchor : y_anchor + self.psf_size,
                    x_anchor : x_anchor + self.psf_size,
                ] += (
                    new_psf * alpha
                )
        obj = obj_ext[
            :,
            :,
            self.psf_size : -self.psf_size,
            self.psf_size : -self.psf_size,
        ].copy()
        return obj
