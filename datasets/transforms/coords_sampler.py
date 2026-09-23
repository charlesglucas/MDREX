import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PIX2AS = 0.01225


def get_coords_sampler(name, filter_real, radius, sep_max, **kwargs):
    """
    filter_real: bool, no overlap with real sources
    """
    if name == "uniform":
        coords_sampler = UniformSampler(**kwargs)
    elif name == "centered":
        coords_sampler = CenteredSampler(**kwargs)
    else:
        raise ValueError()
    if filter_real:
        coords_sampler.add_filter(FilterReal(crop=coords_sampler.crop))
    if radius:
        coords_sampler.add_filter(FilterSynthetic(radius=radius))
    if sep_max:
        coords_sampler.add_filter(
            FilterSep(
                crop=coords_sampler.crop,
                sep_max=sep_max,
                center=coords_sampler.size // 2,
            )
        )
    return coords_sampler


class UniformSampler:
    REQUIRES_REAL = False

    def __init__(
        self,
        n_min,
        n_max,
        size,
        crop,
        n_channels,
        sampling_mode,
        rv_mode,
        exp_coeff,
        seed=0,
        fixed_t0=False,
        timesteps=1,
    ):
        self.n_min = n_min
        self.n_max = n_max
        assert self.n_min <= self.n_max
        # size of the patch or full frame
        self.size = size
        self.crop = crop
        self.fixed_t0 = fixed_t0
        self.coords_filters = []
        self.timesteps = timesteps
        self.n_channels = n_channels
        self.sampling_mode = sampling_mode
        print(f"[UniformSampler] {self.sampling_mode=}")
        assert self.sampling_mode in ["polar", "cartesian"]
        print(f"[UniformSampler] {self.n_channels=}")
        self.rv_mode = rv_mode
        print(f"[UniformSampler] {self.rv_mode=}")
        self.exp_coeff = exp_coeff
        print(f"[UniformSampler] {self.exp_coeff=}")
        assert self.rv_mode in ["linear", "exp"]
        # Not too close of the borders
        self.effective_size = self.size - self.crop
        self.seed = seed
        print(f"[UniformSampler] {self.seed=}")
        assert self.effective_size > 0
        print(f"[UniformSampler] {self.size=}")
        print(f"[UniformSampler] {self.effective_size=}")

    def get_gen(self, seed):
        return np.random.RandomState(seed=self.seed + seed)

    def get_n_sources(self, n_max, gen=None):
        if gen:
            return gen.randint(self.n_min, n_max)
        return np.random.randint(self.n_min, n_max)

    def get_linear(self, gen=None):
        if gen:
            return gen.uniform(size=(self.n_max,))
        return np.random.uniform(size=(self.n_max,))

    def get_exp(self, gen=None):
        return np.exp(-self.exp_coeff * self.get_linear(gen=gen))

    def get_rv(self, gen=None):
        if self.rv_mode == "linear":
            return self.get_linear(gen=gen)
        elif self.rv_mode == "exp":
            return self.get_exp(gen=gen)

    def get_time(self, gen=None):
        if self.fixed_t0:
            return np.ones(self.n_max).astype(int) * (self.timesteps // 2)
        if gen:
            return gen.randint(0, self.timesteps, size=(self.n_max,))
        return np.random.randint(0, self.timesteps, size=(self.n_max,))

    def get_channels(self, gen=None):
        if gen:
            return gen.randint(0, self.n_channels, size=(self.n_max,))
        return np.random.randint(0, self.n_channels, size=(self.n_max,))

    def get_sign(self, gen=None):
        if gen:
            binary = gen.randint(0, 2, size=(self.n_max,))
        else:
            binary = np.random.randint(0, 2, size=(self.n_max,))
        sign = 2 * binary - 1
        return sign

    def sample_coord(self, gen=None):
        if self.sampling_mode == "cartesian":
            return self.sample_coord_cartesian()
        elif self.sampling_mode == "polar":
            return self.sample_coord_polar()
        else:
            raise ValueError()

    def sample_coord_cartesian(self, gen=None):
        if gen:
            return gen.uniform(size=(2,)) * self.effective_size
        return np.random.uniform(size=(2,)) * self.effective_size

    def sample_coord_polar(self, gen=None):
        if gen:
            a, b = gen.uniform(size=(2,))
        else:
            a, b = np.random.uniform(size=(2,))
        rho_pix = a * 1.38 / PIX2AS
        theta = np.deg2rad(b * 360)

        y = -np.cos(theta) * rho_pix
        x = -np.sin(theta) * rho_pix
        center = self.size // 2
        x += center - self.crop // 2
        y += center - self.crop // 2
        return np.array([y, x])

    def add_filter(self, coords_filter):
        print(f"Adding filter: {coords_filter}")
        self.coords_filters.append(coords_filter)
        if isinstance(coords_filter, FilterReal):
            self.REQUIRES_REAL = True
            print(f"coords sampler status changed: {self.REQUIRES_REAL=}")

    def get_coords(self, gen=None, **kwargs):
        """
        Coordinates of the upper left corner of the PSF
        """
        coords = []
        n_fail = 0
        n_max = self.n_max
        while len(coords) < self.n_max:
            # if n_fail > 200:
            if n_fail > 500:
                n_max = len(coords)
                missing = self.n_max - n_max
                coords = coords + [coords[-1]] * missing
                print(f"Sampling failed, {n_max=}")
                break

            candidate = self.sample_coord(gen)

            accepted = True
            for filt in self.coords_filters:
                if not filt(candidate=candidate, coords=coords, **kwargs):
                    accepted = False
                    continue
            if accepted:
                coords.append(candidate)
            else:
                n_fail += 1

        # print(f"{n_fail=}")
        return np.array(coords), n_max


class CenteredSampler(UniformSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_n_sources(self, gen=None):
        return 1

    def sample_coord(self, gen=None):
        return np.ones(2) * (self.effective_size // 2)


class FilterSynthetic:
    def __init__(self, radius):
        self.radius = radius
        print(f"FilterSynthetic instanciated, {self.radius=}")

    def __call__(self, candidate, coords, **kwargs):
        if len(coords) == 0:
            return True
        coords_array = np.array(coords).reshape(-1, 2)
        # (N, 2)
        dist = np.sqrt(
            np.sum((coords_array - candidate[None, ...]) ** 2, axis=1)
        )
        if np.min(dist) > self.radius:
            return True
        return False


class FilterSep:
    def __init__(self, crop, sep_max, center):
        self.crop = crop
        self.center = center
        self.sep_pix = sep_max / PIX2AS
        print(
            f"FilterSep instanciated: {sep_max=}, {PIX2AS=}, {self.sep_pix=} "
            f"{self.center=}"
        )

    def __call__(self, candidate, **kwargs):
        # candidate: np.array(2)
        candidate = candidate + self.crop // 2
        candidate_centered = candidate - self.center
        radius_pix = np.sqrt(np.sum(candidate_centered**2))
        if radius_pix < self.sep_pix:
            return True
        return False


class FilterReal:
    def __init__(self, crop, margin=18):
        # assert coords_real.ndim == 2
        # self.coords_real = coords_real
        self.crop = crop
        assert self.crop % 2 == 0
        self.margin = margin
        print("FilterReal instanciated")
        # (N, 2)
        # self.radius_real = radius_real
        # (N,)

    def __call__(self, candidate, coords_real, radius_real, **kwargs):
        if len(radius_real) == 0:
            return True
        # coords_array = np.array(coords).reshape(-1, 2)
        # NOTE: this is probably not good, coords_candidate=candidate
        # NOTE: in fact, it is OK
        coords_candidate = candidate + self.crop // 2
        # (N, 2)
        dist = np.sqrt(
            np.sum((coords_real - coords_candidate[None, ...]) ** 2, axis=1)
        )
        # (N,)
        if np.all(dist > (radius_real + self.margin)):
            return True
        return False


class CoordsSamplerTransform:
    def __init__(
        self,
        deterministic,
        alpha_range,
        adaptive,
        range_max=None,
        symmetric_flux=False,
        **kwargs,
    ):
        self.deterministic = deterministic
        self.adaptive = adaptive
        print(f"[CoordsSampler] {self.adaptive=}")
        self.coords_sampler = get_coords_sampler(**kwargs)
        self.symmetric_flux = symmetric_flux
        print(f"{self.symmetric_flux=}")
        self.range_max = range_max
        print(f"[CoordsSampler] {self.range_max=}")

        self.alpha_min = float(alpha_range[0])
        self.alpha_max = float(alpha_range[1])
        print(f"[CoordsSampler] {self.alpha_min=} - {self.alpha_max=}")

    def __call__(self, item):
        coords_real = None
        radius_real = None
        if self.coords_sampler.REQUIRES_REAL:
            coords_real = item["coords_real"]
            radius_real = item["radius_real"]
            n_sources_real = item["n_sources_real"]
            if n_sources_real > 0:
                coords_real = coords_real[:n_sources_real]
                radius_real = radius_real[:n_sources_real]

        gen = None
        if self.deterministic:
            idx = item["idx"]
            gen = self.coords_sampler.get_gen(idx)

        # NOTE: coords of the UPPER LEFT corner of the CROPPED PSF
        all_coords, n_max = self.coords_sampler.get_coords(
            gen, coords_real=coords_real, radius_real=radius_real
        )
        n_sources = self.coords_sampler.get_n_sources(gen=gen, n_max=n_max)
        rv = self.coords_sampler.get_rv(gen)
        if self.adaptive:
            psf = item["psf"]
            C, h, w = psf.shape
            y0 = all_coords[:, 0] + h // 2
            x0 = all_coords[:, 1] + w // 2
            try:
                y = item["frame"]
                C, T, H, W = y.shape
                std = np.std(y, axis=1)[0]
                std_0 = std[y0.astype(int), x0.astype(int)]
            except KeyError as e:
                raise e
                breakpoint()
                y = item["s_0"]
                T, C, H, W = y.shape
                std = np.std(y, axis=0)
                std_0 = std[0, y0.astype(int), x0.astype(int)]

            max_psf = np.max(psf[0])

            # breakpoint()
            # all_alpha

            # coords center
            assert self.range_max is not None
            all_alpha = self.alpha_min + self.range_max * rv * std_0 / (
                max_psf * np.sqrt(T)
            )
            # all_alpha = self.range_max * uniform * std_0 / max_psf
            # print(all_alpha)

            # breakpoint()
        else:
            all_alpha = self.alpha_min + rv * (self.alpha_max - self.alpha_min)
        all_t0 = self.coords_sampler.get_time(gen)
        # breakpoint()
        channels = self.coords_sampler.get_channels(gen)
        if self.symmetric_flux:
            all_sign = self.coords_sampler.get_sign(gen)
            all_alpha *= all_sign
        item["coords"] = all_coords
        item["n_sources"] = n_sources
        item["alpha"] = all_alpha
        item["t0"] = all_t0
        item["channels"] = channels

        return item
