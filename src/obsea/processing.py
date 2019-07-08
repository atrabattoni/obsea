"""
Processing module.

Used to post-process 2D representations.

"""
import numpy as np
import xarray as xr
import scipy.signal as sp
from sklearn.decomposition import NMF


def decompose(xarr):
    """
    Decompose orientation-frequency representation into signal and noise.

    Parameters
    ----------
    xarr : xarray.DataArray
        Orientation-frequecy representation.

    Returns
    -------
    xarray.DataArray
        Pseudo Probality Function Densuity of instrument orientation.

    """
    # NMF
    model = NMF(n_components=2)
    W = model.fit_transform(xarr.values)
    H = model.components_
    # Renormalization
    Hn = H * W.max(axis=0)[..., np.newaxis]
    # MRL selection
    theta = np.deg2rad(xarr.coords['orientation'].values)
    mrl = np.abs(np.mean(Hn * np.exp(1j * theta), axis=-1) * (2 * np.pi))
    data = Hn[0 if mrl[0] >= mrl[1] else 1]
    coords = {'orientation': xarr.coords['orientation'].values}
    return xr.DataArray(data=data, coords=coords, dims=['orientation'])


def find_knee(s):
    """
    Find knee of a decreading curve using the kneedle algorithm.

    Parameters
    ----------
    s : array_like
        Decreasing curve.

    Returns
    -------
    int
        Knee as an index.

    """
    yn = s / np.max(s, axis=-1)[..., np.newaxis]
    xn = np.linspace(0, 1, yn.shape[-1])
    dn = 1 - yn - xn
    knee = np.argmax(dn, axis=-1)
    return knee


def svd_filter(xarr):
    """
    Perform SVD clutter filtering.

    Parameters
    ----------
    xarr : xarray.DataArray
        2D representation to filter.

    Returns
    -------
    xarray.DataArray
        Filtered 2D representation.

    """
    xarr = xarr.dropna(dim='time')
    u, s, vh = np.linalg.svd(
        (xarr - xarr.mean(dim='time')).values,
        full_matrices=False)
    knee = find_knee(s)
    mask = np.arange(s.shape[0]) < knee
    s[mask] = 0
    filtered = u @ np.apply_along_axis(np.diag, -1, s) @ vh
    return xr.DataArray(filtered, coords=xarr.coords, dims=xarr.dims)


def decimate(xarr, dim, factor):
    """
    Perform decimation along the horizontal axis of a Cepstrogram.

    Parameters
    ----------
    xarr : xarray.DataArray
        Cepstrogram to decimate.
    factor : int
        The downsampling factor.

    Returns
    -------
    xarray.DataArray
        Decimeted Cepstrogram.

    """
    xarr = xarr.dropna(dim=dim)
    axis = xarr.dims.index(dim)
    decimated = sp.decimate(xarr.values, factor, axis=axis)
    coords = {dim: xarr[dim].values for dim in xarr.dims}
    coords[dim] = coords[dim][::factor]
    return xr.DataArray(data=decimated, coords=coords, dims=xarr.dims)


def highpass(xarr, dim, freq=0.25, order=2):
    """
    Apply a highpass filter of normalized frequency 0.25 on the vertical axis.

    Parameters
    ----------
    xarr : xarray.DataArray
        2D representation to filter.

    Returns
    -------
    xarray.DataArray
        Filtered 2D representation.

    """
    xarr = xarr.dropna(dim=dim)
    axis = xarr.dims.index(dim)
    b, a = sp.butter(order, freq, btype='highpass')
    filtered = sp.filtfilt(b, a, xarr.values, axis=axis)
    return xr.DataArray(data=filtered, coords=xarr.coords, dims=xarr.dims)


def resample(xarr, dim, factor):
    """
    Perform resampling along the vertical axis of a Cepstrogram.

    Parameters
    ----------
    xarr : xarray.DataArray
        Cepstrogram to decimate.
    factor : int
        The upsampling factor.

    Returns
    -------
    xarray.DataArray
        Resampled Cepstrogram.

    """
    xarr = xarr.dropna(dim=dim)
    axis = xarr.dims.index(dim)
    n = xarr[dim].size
    resampled = sp.resample(xarr.values, factor * n, axis=axis)
    d = (xarr[dim][1].values - xarr[dim][0].values) / factor
    coords = {dim: xarr[dim].values for dim in xarr.dims}
    coords[dim] = d * np.arange(factor * n) + xarr[dim][0].values
    return xr.DataArray(data=resampled, coords=coords, dims=xarr.dims)
