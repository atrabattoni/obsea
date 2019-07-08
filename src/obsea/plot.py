"""
Plot module.

Used to plot complex representations.

"""
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc


def plot_azigram(xarr, **kwargs):
    """
    Plot an azigram with transparency.

    Parameters
    ----------
    xarr : xarray.DataArray
        Azigram
    **kwargs
        Additional arguments to pass to pcolormesh.

    Returns
    -------
    matplotlib.QuadMesh
        Azigram plot.

    """
    result = np.rad2deg(np.arctan2(xarr.real, xarr.imag)) % 360
    if xarr.attrs['double_angle']:
        result /= 2
        vmax = 180
    else:
        vmax = 360
    alpha = np.abs(xarr).values.ravel()[..., np.newaxis]
    img = result.plot(cmap=cc.cm.colorwheel, vmin=0, vmax=vmax, **kwargs)
    plt.draw()
    white = np.array([1, 1, 1, 1])
    img.set_facecolor(img.get_facecolor() * alpha + white * (1 - alpha))
    plt.draw()
    return img
