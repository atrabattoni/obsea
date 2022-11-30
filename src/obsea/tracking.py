"""
Tracking module.

Used to perform temporal segmetation and rectilinear trajectories estimation.
"""


import numpy as np
import xarray as xr
from numba import njit
from scipy.signal import gaussian
from scipy.ndimage import convolve


# Temporal segmentation


def delimit(dtc):
    """
    Performs temporal segmentation.

    Parameters
    ----------
    dtc : Dataset
        Detection probability considering the general case ("all") the approaching
        case ("vm") and the leaving case ("vp").

    Returns
    -------
    list of tuple :
        The temporal segments as tuples (starttime, cpa_time, endtime).

    """
    lims = [limit(dtc[tag], tag) for tag in ["all", "vm", "vp"]]
    lims = lims[0] + lims[1] + lims[2]
    lims = sorted(lims)
    segments = []
    status = "null"
    prevlim = ()
    for lim in lims:
        if lim[1] == "all":
            segments.append((lim[0], None, lim[2]))
        if lim[1] == "vm":
            if status == "vp":
                segment = segments.pop()
                delta = (prevlim[2] - lim[0]) / 2 - np.timedelta64(45, "s")
                segments.append((segment[0], segment[1], prevlim[2] + delta))
                segments.append((lim[0] - delta, None, segment[2]))
        if lim[1] == "vp":
            if status == "vm":
                segment = segments.pop()
                delta = (prevlim[2] - lim[0]) / 2
                cpa = lim[0] + delta
                segments.append((segment[0], cpa, segment[2]))
        status = lim[1]
        prevlim = lim
        segments = [
            segment
            for segment in segments
            if segment[2] - segment[0] >= np.timedelta64(3600, "s")
        ]
    return segments


def limit(mask, tag):
    starttime = xr.DataArray(
        data=(~mask[:-1].values) & mask[1:].values,
        coords={"time": mask[1:]["time"]},
        dims="time",
    )
    endtime = xr.DataArray(
        data=mask[:-1].values & (~mask[1:].values),
        coords={"time": mask[1:]["time"]},
        dims="time",
    )
    starttime = starttime[starttime]["time"].values
    endtime = endtime[endtime]["time"].values
    limits = [(s, tag, e) for s, e in zip(starttime, endtime)]
    return limits


def segment(ell, n):
    ell = np.log(ell)
    ell = ell.rolling(time=n, center=True).mean() / 2
    ell = np.exp(ell)
    return ell


def select_segment(data, segment, convert=None):
    if convert is None:
        start = segment["starttime"]
        end = segment["endtime"]
    elif convert == "posix":
        start = to_posix(segment["starttime"])
        end = to_posix(segment["endtime"])
    else:
        raise (ValueError("convert must be None or 'posix'"))
    return data.sel(time=slice(start, end))


def chunk(xarr, date_range):
    N = len(date_range) - 1
    return [xarr.sel(time=slice(date_range[i], date_range[i + 1])) for i in range(N)]


# Track parameters estimation


def make_brute_force(loglik):
    line_loglik = make_line_loglik(loglik)

    @njit
    def brute_force(t, r, a, v):
        index = (-1, -1, -1, -1)
        loglik = 0.0
        for i in range(len(t)):
            for j in range(len(r)):
                for k in range(len(a)):
                    for l in range(len(v)):
                        value = line_loglik(t[i], r[j], a[k], v[l])
                        if value > loglik:
                            index = (i, j, k, l)
                            loglik = value
        return index, value

    return brute_force


def make_line_loglik(loglik):
    t = loglik["time"].values
    r0 = loglik["distance"][0].values
    dr = loglik["distance"][1].values - r0
    v0 = loglik["speed"][0].values
    dv = loglik["speed"][1].values - v0
    a0 = np.deg2rad(loglik["azimuth"][0].values)
    da = np.deg2rad(loglik["azimuth"][1].values - a0)
    z = loglik["r"].values
    y = loglik["a"].values

    @njit
    def line_loglik(t_cpa, r_cpa, a_inf, v_inf):
        r, a, v = generate_line(t_cpa, r_cpa, a_inf, v_inf, t)
        cost_r = interp2d(r, r0, dr, v, v0, dv, z)
        cost_a = interp1d(a, a0, da, y)
        return cost_r + cost_a

    return line_loglik


@njit
def generate_line(t_cpa, r_cpa, a_inf, v_inf, t):
    x0 = -r_cpa * np.cos(a_inf)
    y0 = r_cpa * np.sin(a_inf)
    vx = v_inf * np.sin(a_inf)
    vy = v_inf * np.cos(a_inf)
    x = x0 + vx * (t - t_cpa)
    y = y0 + vy * (t - t_cpa)
    r = np.sqrt(x**2 + y**2)
    a = np.arctan2(x, y) % (2 * np.pi)
    vr = vx * np.sin(a) + vy * np.cos(a)
    return r, a, vr


def smooth(ell, sigma_r, sigma_a, sigma_v):
    out = ell.copy()

    out["r"].values = convolve(
        out["r"].values,
        kernel(sigma_r)[None, :, None],
        mode="constant",
    )
    out["r"].values = convolve(
        out["r"].values,
        kernel(sigma_v)[None, None, :],
        mode="constant",
    )
    out["a"].values = convolve(
        out["a"].values,
        kernel(sigma_a)[None, :],
        mode="wrap",
    )

    return out


def kernel(sigma):
    M = 2 * 6 * sigma + 1
    out = gaussian(M, sigma)
    out /= np.sum(out)
    return out


# Interpolation


@njit
def interp1d(x, x0, dx, y):
    out = 0.0
    for k in range(len(x)):
        out += linear1d(x[k], x0, dx, y[k])
    return out / len(x)


@njit
def interp2d(x, x0, dx, y, y0, dy, z):
    out = 0.0
    for k in range(len(x)):
        out += linear2d(x[k], x0, dx, y[k], y0, dy, z[k])
    return out / len(x)


@njit
def linear1d(x, x0, dx, y):
    i, mu = x2imu(x, x0, dx)
    if inbounds(i, len(y)):
        c0 = y[i]
        c1 = y[i + 1] - y[i]
        return c1 * mu + c0
    else:
        return 0.0


@njit
def linear2d(x, x0, dx, y, y0, dy, z):
    i_x, mu_x = x2imu(x, x0, dx)
    i_y, mu_y = x2imu(y, y0, dy)
    if inbounds(i_x, z.shape[0]) and inbounds(i_y, z.shape[1]):
        c0 = z[i_x, i_y]
        cx = z[i_x + 1, i_y] - z[i_x, i_y]
        cy = z[i_x, i_y + 1] - z[i_x, i_y]
        cxy = z[i_x + 1, i_y + 1] + z[i_x, i_y] - z[i_x + 1, i_y] - z[i_x, i_y + 1]
        return cxy * mu_x * mu_y + cx * mu_x + cy * mu_y + c0
    else:
        return 0.0


@njit
def x2imu(x, x0, dx):
    d = (x - x0) / dx
    i = int(d)
    mu = d - i
    return i, mu


@njit
def inbounds(i, N):
    return (0 <= i) and (i < N - 1)


# Utilities


def marginal(ell):
    if "distance" in ell.dims:
        distance = ell["distance"]
        marginal = (ell * distance).sum("distance") / distance.sum("distance")
    else:
        marginal = ell
    dims = [dim for dim in ell.dims if not dim == "time"]
    return ell.mean(dims)


def detection_probability(ell, pd):
    return (1.0 - pd) + pd * ell


def ell2proba(ell):
    return ell / (1.0 + ell)


def to_posix(timestamp):
    return (timestamp - np.datetime64(0, "s")) / np.timedelta64(1, "s")
