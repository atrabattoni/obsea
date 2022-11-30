"""
Ray tracing module.
"""

import numpy as np
import xarray as xr
from scipy.integrate import quad, solve_ivp


def munk(z, c0=1492.0, zc=1300.0, B=1300.0, gamma=1.14e-5):
    epsilon = B * gamma / 2.0
    zbar = 2.0 * (z - zc) / B
    c = c0 * (1.0 + epsilon * (zbar - 1.0 + np.exp(-zbar)))
    dcdz = c0 * epsilon * (2.0 / zc) * (1.0 - np.exp(-zbar))
    return [c, dcdz]


def equivalent_celerity(ssp, depth):
    t = quad(lambda z: 1 / ssp(z)[0], 0.0, depth)[0]
    return depth / t


# def toa(r, i, c, z):
#     return np.sqrt((i * z)**2 + (r**2)) / c


# def tdoa(r, i, j, c, z):
#     return toa(r, j, c, z) - toa(r, i, c, z)


def initial(z0, theta, ssp):
    r0 = t0 = 0.0
    (
        c0,
        _,
    ) = ssp(z0)
    ksi0 = np.cos(theta) / c0
    zeta0 = np.sin(theta) / c0
    return [r0, z0, ksi0, zeta0, t0]


def reflectioncoeff(ksi, zeta, c, rho):
    rho0 = 1000.0
    ksi = np.abs(ksi)  # boundary coordinate
    zeta = np.abs(zeta)  # boundary coordinate
    ksic = np.sqrt(ksi**2 - (1.0 / complex(c)) ** 2)
    return -(ksic * rho0 - 1j * zeta * rho) / (ksic * rho0 + 1j * zeta * rho)


# def reflectioncoeff(ksi, zeta, cp, rho):
#     cs = 500.0
#     rho0 = 1000.0
#     ksi = np.abs(ksi)  # boundary coordinate
#     zeta = np.abs(zeta)  # boundary coordinate
#     ksip = np.sqrt(ksi ** 2 - (1.0 / complex(cp)) ** 2)
#     ksis = np.sqrt(ksi ** 2 - (1.0 / complex(cs)) ** 2)
#     mu = rho * cs**2
#     f = ksip * (ksi**2 - ksis**2)
#     g = ((ksis**2 + ksi**2)**2 - 4.0 * ksis * ksip * ksi**2) * mu
#     return - (rho0 * f - 1j * zeta * g) / (rho0 * f + 1j * zeta * g)


def trace_direct(ssp, depth, c, rho, n, smax, **kwargs):
    def fun(s, y):
        (
            _,
            z,
            ksi,
            zeta,
            _,
        ) = y
        c, dcdz = ssp(z)
        dyds = [
            c * ksi,
            c * zeta,
            0.0,
            -dcdz / c**2,
            1.0 / c,
        ]
        return dyds

    def hit_ground(s, y):
        return y[1] - depth

    hit_ground.terminal = True
    hit_ground.direction = 1

    theta = np.linspace(np.pi / 2.0, 0.0, n)
    r = np.zeros(n)
    ksi = np.zeros(n)
    zeta = np.zeros(n)
    toa = np.zeros(n)

    for k in range(n):
        y0 = initial(0.0, theta[k], ssp)
        sol = solve_ivp(fun, (0.0, smax), y0, events=[hit_ground], **kwargs)
        y = np.ravel(sol.y_events)
        r[k], _, ksi[k], zeta[k], toa[k] = y

    coeff = reflectioncoeff(ksi, zeta, c, rho)

    r[0] = 0.0

    return xr.Dataset(
        data_vars={"toa": ("distance", toa), "poa": ("distance", coeff)},
        coords={"distance": r},
    )


def reflect_direct(direct, n):
    reflected = direct.copy()
    reflected["distance"] = n * reflected["distance"]
    reflected["toa"] = n * reflected["toa"]
    reflected["poa"] = (1.0 + reflected["poa"]) * (-reflected["poa"]) ** ((n - 1) // 2)
    reflected["poa"] = np.arctan2(reflected["poa"].imag, reflected["poa"].real)
    return reflected


def compose_multipath(direct, n, r):
    paths = []
    for nk in n:
        path = reflect_direct(direct, nk)
        path = path.interp({"distance": r})
        paths.append(path)
    return xr.concat(paths, xr.IndexVariable("path", n))


def compute_tdoa(multipath):
    tdoa = multipath["toa"].diff("path")
    tdoa = tdoa.rename(path="interference")
    tdoa["interference"] = ["13", "35", "57"]
    return tdoa


def compute_pdoa(multipath):
    pdoa = -multipath["poa"].diff("path") % (2 * np.pi)
    pdoa = pdoa.rename(path="interference")
    pdoa["interference"] = ["13", "35", "57"]
    return pdoa
