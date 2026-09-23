import numpy as np
import torch


def get_radius(size, x, y):
    yy, xx = np.mgrid[0:size, 0:size]
    xx = xx - size // 2 - x
    yy = yy - size // 2 - y
    rr = np.sqrt(xx**2 + yy**2)
    return rr


def fake_source(size, b=0.5, a=1, x=0, y=0):
    rr = get_radius(size, x, y)
    out = a * np.exp(-b * rr)

    # rr_half = - np.log(0.5) / b
    # print(f"radius: {rr_half}")
    # mask = rr < rr_half
    # flux = np.mean(out[mask])

    return torch.tensor(out, dtype=torch.float32)


def polar_to_cart(pa, sep, rot, center=512):
    if isinstance(sep, np.ndarray):
        sep = sep.reshape(1, -1)
    sep = sep / 1000
    if isinstance(pa, np.ndarray):
        pa = pa.reshape(1, -1)
    if isinstance(rot, np.ndarray):
        rot = rot.reshape(-1, 1)
    angle = np.deg2rad(-rot - pa + 180)
    y_coords = -np.cos(angle) * sep / 0.01225
    x_coords = -np.sin(angle) * sep / 0.01225
    x_coords += center
    y_coords += center
    return x_coords, y_coords


def cart_to_polar(x, y, rot, center=512):
    x -= center
    y -= center
    radius = np.hypot(x, y)
    sep = radius * 12.25
    angle_rad = np.arctan2(x, y)
    angle_deg = np.degrees(angle_rad)
    pa = angle_deg + rot
    pa = (360 - pa) % 360
    return sep, pa


def check_polar_to_cart():
    sep = 6079.07
    pa = 304.72
    np_sep = np.array(
        [
            sep,
        ]
    )
    np_pa = np.array(
        [
            pa,
        ]
    )
    x0, y0 = polar_to_cart(pa=pa, sep=sep, rot=0, center=0)
    x1, y1 = polar_to_cart(pa=np_pa, sep=np_sep, rot=0, center=0)
    print(x0, y0)
    print(x1, y1)
