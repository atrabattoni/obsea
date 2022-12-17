"""
Detection module.

Used to perform azimuthal and radial detection.
"""

import numpy as np
import xarray as xr
from numba import njit
from scipy.ndimage import gaussian_filter

from .beamforming import linear_beamform


def build_model(mu, sigma, tdoa, dtau, fs):
    """
    Build propagative model.

    Parameters
    ----------
    mu : DataArray
        Expected mean of the cepstrum as a function of quefrency and distance.
    sigma : DataArray
        Expected std of the cepstrum as a function of quefrency.
    tdoa : DataArray
        Time difference of arrival as a function of the interference and distance.
    dtau : float
        Cepstral pulse width in seconds. Can be chosen as the inverse of the signal bandwidth.
    fs : float
        Sampling rate
    """
    mu.values = np.nan_to_num(mu.values)
    sigma.values = np.nan_to_num(sigma.values)
    tdoa.values = np.nan_to_num(tdoa.values)
    thresh, weight = ms2tw(mu.values, sigma.values)
    index = v2i(tdoa.values, 1 / fs)
    halfwidth = v2i(3 * dtau, 1 / fs)
    model = {
        "thresh": thresh,
        "weight": weight,
        "index": index,
        "halfwidth": halfwidth,
    }
    return model


def tonal_detection(u, n, orientation, R, dt, endpoint=True, t_step=None, t=None):
    """
    Compute the azimuthal likelihood ratio for tonal sources.

    Parameters
    ----------
    u : DataArray
        Time-frequency azimuthal measurements as polar unitary complex values.
        Dimenions must be "time" and "frequency".
    n : int
        Number of azimuthal bins from 0 to 360 degrees.
    orientation : float
        Sensor azimuthal orientation in degrees.
    gamma : float
        Wrapped Cauchy parameter. R = exp(-gamma).
    dt : Float
        Time window duration in seconds used for likelihood computation.
    endpoint : bool, optional
        Weather to include 360 as the last sample. Default is True
    t_step : float, optional
        Time increment at which compute values. Default to dt/2.

    Returns
    -------
    DataArray
        Likelihood ratio with dimensions "time" and "azimuth".
    """
    gamma = R2gamma(R)
    if t is None:
        t = t_range(u["time"].values, dt, t_step=t_step)
    zeta = zeta_range(n, orientation, gamma, endpoint=endpoint)
    ell = xr.DataArray(
        data=np.zeros((len(t), len(zeta))),
        coords={
            "time": t,
            "azimuth": zeta["azimuth"],
        },
        dims=("time", "azimuth"),
    )

    for t0 in t:
        ell.loc[{"time": t0}] = (
            wrapcauchy(u.loc[{"time": slice(t0 - dt / 2, t0 + dt / 2)}], zeta)
            .prod("time")
            .mean("frequency")
        )

    return ell


def impulsive_detection(
    u, n, orientation, R, dt, df, pd, endpoint=True, t_step=None, t=None
):
    """
    Compute the azimuthal likelihood ratio for impulsive sources.

    Parameters
    ----------
    u : DataArray
        Time-frequency azimuthal measurements as polar unitary complex values.
        Dimenions must be "time" and "frequency".
    n : int
        Number of azimuthal bins from 0 to 360 degrees.
    orientation : float
        Sensor azimuthal orientation in degrees.
    gamma : float
        Wrapped Cauchy parameter. R = exp(-gamma).
    dt : Float
        Time window duration in seconds used for likelihood computation.
    df : Float
        Source frequency bandwidth in Hertz.
    pd : Float
        Source probability of detection during one time sample.
    endpoint : bool, optional
        Weather to include 360 as the last sample. Default is True
    t_step : float, optional
        Time increment at which compute values. Default to dt/2.

    Returns
    -------
    DataArray
        Likelihood ratio with dimensions "time" and "azimuth".
    """
    gamma = R2gamma(R)
    f = u["frequency"].values
    nf = int(round(df / (f[1] - f[0])))
    if t is None:
        t = t_range(u["time"].values, dt, t_step=t_step)
    zeta = zeta_range(n, orientation, gamma, endpoint=endpoint)
    ell = xr.DataArray(
        data=np.zeros((len(t), len(zeta))),
        coords={
            "time": t,
            "azimuth": zeta["azimuth"],
        },
        dims=("time", "azimuth"),
    )

    for t0 in t:
        ell.loc[{"time": t0}] = (
            wrapcauchy(u.loc[{"time": slice(t0 - dt / 2, t0 + dt / 2)}], zeta)
            # .rolling(frequency=nf, center=True)
            # .construct("w", stride=nf//2)
            # .prod("w")
            # .mean("frequency")
            .pipe(np.log)
            .sum("frequency")
            .pipe(lambda x: x / 2)
            .pipe(np.exp)
            .pipe(lambda x: (1 - pd) + pd * x)
            .pipe(np.log)
            .sum("time")
            .pipe(np.exp)
        )

    return ell


def cepstral_detection(ceps, model, grid, nsigma, t_step=None, t=None):
    """
    Compute the radial log likelihood ratio.

    Parameters
    ----------
    ceps: DataArray
        The cepstrogram from which the detection with be performed. Dimensions must be
        (`quefrency`, `time`).
    model: DataArray
        The propagative model. Give the expected cepstram mean for each `interference`,
        `distance` and `quefrency` dimensions.
    grid: dict
        Parameters of the grid over which probabilities are computed. Must include
        `dt` the lenght of the sliding window (seconds), `dr` and `rmax` the radial
        step and max value (meters), and, `dv` and `vmax the speed step and max value
        (m/s).
    nsigma: float
        Smoothing value along the quefrency dimension in number of samples.
    t_step: float
        Temporal stepping (seconds) at which compute probabilities.
    t: 1d array
        Time (seconds) at wich probabilties must be computed. Usefull to synchronize
        with another detector. If provided `t_step` is ignored.
    """
    # Logell computation
    dt, dr, rmax, dv, vmax = map(grid.get, ("dt", "dr", "rmax", "dv", "vmax"))
    data = compute_logell(ceps.values, **model)
    logell = xr.DataArray(
        data=data,
        coords={
            "interference": [1, 2, 3],
            "distance": dr * np.arange(data.shape[1]),  # TODO
            "time": ceps["time"],
        },
        dims=("interference", "distance", "time"),
    )

    # Output allocation
    if t is None:
        t = t_range(logell["time"].values, dt, t_step=t_step)
    r = np.arange(0, rmax + dr, dr)
    v = np.arange(-vmax, vmax + dv, dv)
    out = xr.DataArray(
        data=np.zeros((len(t), len(r), len(v))),
        coords={
            "time": t,
            "distance": r,
            "speed": v,
        },
        dims=("time", "distance", "speed"),
    )

    # Speed filtering
    for i in [1, 2, 3]:
        for t0 in t:
            # Beamform on segments
            query = {
                "interference": i,
                "time": slice(t0 - dt / 2, t0 + dt / 2),
            }
            beam = linear_beamform(v, r, logell.loc[query], dims=["distance", "speed"])
            # Process noise
            beam = np.exp(beam)
            beam.values = gaussian_filter(beam.values, (nsigma, 0))
            beam = np.log(beam)
            # Sum interferences
            out.loc[{"time": t0}] += beam

    return np.exp(out)


@njit
def compute_logell(x, thresh, weight, index, halfwidth):
    _, Nt = x.shape
    Ni, Nr, _ = thresh.shape
    logell = np.zeros((Ni, Nr, Nt))
    for i in range(Ni):  # interference
        for j in range(Nr):  # distance
            if index[i, j] == 0:
                continue
            qrange = range(index[i, j] - halfwidth, index[i, j] + halfwidth + 1)
            for k in qrange:  # quefrency
                for l in range(Nt):  # time
                    logell[i, j, l] += weight[i, j, k] * (x[k, l] - thresh[i, j, k])
    return logell


def abs2(z):
    return z.real**2 + z.imag**2


def compute_zeta(mu, gamma):
    return np.exp(1j * (mu + 1j * gamma))


def wrapcauchy(z, zeta):
    return (1 - abs2(zeta)) / abs2(z - zeta)


def R2gamma(R):
    return -np.log(R)


def zeta_range(n, orientation, gamma, endpoint=True):
    azimuth = np.linspace(0, 2 * np.pi, n, endpoint=endpoint)
    angle = (np.pi / 2 + np.deg2rad(orientation) - azimuth) % (2 * np.pi)
    zeta = compute_zeta(angle, gamma)
    return xr.DataArray(
        data=zeta, coords={"azimuth": np.rad2deg(azimuth)}, dims=("azimuth",)
    )


def t_range(t, dt, t_step=None):
    if t_step is None:
        t_step = dt / 2
    t_min = np.min(t) + dt / 2
    t_max = np.max(t) - dt / 2
    return np.arange(t_min, t_max, t_step)


def ms2tw(mu, sigma):
    thresh = mu / 2
    weight = mu / sigma[np.newaxis, np.newaxis, :] ** 2
    return thresh, weight


def v2i(x, dx):
    return np.round(x / dx).astype(int)
