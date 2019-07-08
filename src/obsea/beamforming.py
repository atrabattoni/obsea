"""
Beamforming module.

Used to relocalize recoding instruments on the seafloor.

"""
import numpy as np
import xarray as xr
from numba import njit
from scipy.ndimage import gaussian_filter


@njit
def x2imu(x, x0, dx):
    """
    Locate a value in a 1D evenly spaced grid as index and remainder.

    Parameters
    ----------
    x : float
        Value which location must be retreived.
    x0 : float
        Grid starting location.
    dx : float
        Grid spacing.

    Returns
    -------
    i : float
        Floor index.
    mu : float
        Remainder (between 0 and 1).

    """
    d = (x - x0) / dx
    i = int(d)
    mu = d - i
    return i, mu


@njit
def linear(x, x0, dx, y):
    """
    One-dimensional linear interpolation of evenly spaced data.

    Parameters
    ----------
    x : float
        Value at which the interpolation in performed
    x0 : Grid starting location
        Grid starting location.
    dx : float
        Grid spacing.
    y : array_like
        Values at each grid location.

    Returns
    -------
    float
        Interpolated value.

    """
    i, mu = x2imu(x, x0, dx)
    c0 = y[i]
    c1 = y[i + 1] - y[i]
    return c1 * mu + c0


@njit
def cubic(x, x0, dx, y):
    """
    One-dimensional cubic interpolation of evenly spaced data.

    Parameters
    ----------
    x : float
        Value at which the interpolation in performed
    x0 : Grid starting location
        Grid starting location.
    dx : float
        Grid spacing.
    y : array_like
        Values at each grid location.

    Returns
    -------
    float
        Interpolated value.

    """
    i, mu = x2imu(x, x0, dx)
    mu2 = mu * mu
    mu3 = mu2 * mu
    c0 = y[i]
    c1 = -y[i - 1] / 3 - y[i] / 2 + y[i + 1] - y[i + 2] / 6
    c2 = y[i - 1] / 2 - y[i] + y[i + 1] / 2
    c3 = -y[i - 1] / 6 + y[i] / 2 - y[i + 1] / 2 + y[i + 2] / 6
    return c3 * mu3 + c2 * mu2 + c1 * mu + c0


@njit
def interp(x, xp, fp, kind='cubic'):
    """
    One-dimensional interpolation of evenly spaced data.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points (xp, fp), evaluated at x.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp : 1-D sequence of floats
        The x-coordinates of the data points
    fp : 1-D sequence of float or complex
        The y-coordinates of the data points, same length as xp.
    kind : str, optional
        Type of interpolation. Default to cubic.

    Returns
    -------
    float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as x.

    Raises
    ------
    NotImplementedError
        For now only kind='cubic' is accepted.

    """
    if not kind == 'cubic':
        raise NotImplementedError
    y = np.zeros_like(x)
    xp0 = xp[0]
    dxp = xp[1] - xp[0]
    for i, xi in enumerate(x):
        y[i] = cubic(xi, xp0, dxp, fp)
    return y


def make_delay(track_xarr):
    """
    Build a delay function according to a range independent simple model.

    Parameters
    ----------
    track_xarr : xarray.dataArray
        A ship trajectory.

    Returns
    -------
    delay(r, t, c, d) : function
        Delay function which returns the proper delay according to the a given
        position r, a time t, a celerity c and a depth d.

    """
    t = track_xarr['time'].values
    t0 = t[0]
    dt = t[1] - t[0]
    y = track_xarr.values

    def delay(r, t, c, d):
        rt = linear(t, t0, dt, y)
        l = np.abs(rt - r)
        ti = np.sqrt((1 * d)**2 + (l**2)) / c
        tj = np.sqrt((3 * d)**2 + (l**2)) / c
        return tj - ti
    return delay


def make_beamform(x, y, xarr, delay, interpolation='cubic'):
    """
    Build a beamformer.

    Parameters
    ----------
    x : array_like
        The x-coordinates.
    y : array_like
        The y-coordinates.
    xarr : xarray.DataArray
        Data from which beamforming is performed.
    delay : function
        Function used to compute  delays.
    interpolation : str, optional
        Interpolation scheme used (Default cubic).

    Returns
    -------
    beamform(*args, enveloppe=True) : function
        Beamform function ready to be run. Given arguments to that function are
        transmetted to the delay function in order to explore different
        beamforming parameters. The eveloppe kwarg dictate wether to return the
        enveloppe of the computed image or not.

    Raises
    ------
    NotImplementedError
        For now only interpolation='cubic' is accepted.

    """
    if not interpolation == 'cubic':
        raise NotImplementedError
    zp = xarr.values
    xp = xarr['time'].values
    yp = xarr['quefrency'].values
    delay = njit(delay)

    @njit
    def core(*args):
        img = np.zeros(y.shape + x.shape, dtype=zp.dtype)
        yp0 = yp[0]
        dyp = yp[1] - yp[0]
        for i, yi in enumerate(y):
            for j, xj in enumerate(x):
                rij = xj + 1j * yi
                for k, xpk in enumerate(xp):
                    d = delay(rij, xpk, *args)
                    signal = zp[..., k]
                    img[i, j] += cubic(d, yp0, dyp, signal)
        return img / xp.size

    def beamform(*args, enveloppe=True):
        img = core(*args)
        if enveloppe:
            img = np.abs(img)
        return xr.DataArray(img, coords={'x': x, 'y': y}, dims=('y', 'x'))
    return beamform


def blur(img, sigma):
    """
    Blur a beamformed image.

    Parameters
    ----------
    img : xarray.DataArray
        Beamformed image.
    sigma : float
        Gaussian kernel standard deviation in meters used to blur.

    Returns
    -------
    xarray.DataArray
        Blured beamformed image.

    """
    x = img['x'].values
    y = img['y'].values
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    sx = sigma / dx
    sy = sigma / dy
    img.values = gaussian_filter(img.values, (sy, sx))
    return img
