import numpy as np
import xarray as xr
from tqdm import tqdm


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
        data=zeta,
        coords={"azimuth": np.rad2deg(azimuth)},
        dims=("azimuth",)
    )


def t_range(t, dt, t_step=None):
    if t_step is None:
        t_step = dt / 2
    t_min = np.min(t) + dt / 2
    t_max = np.max(t) - dt / 2
    return np.arange(t_min, t_max, t_step)


def tonal_detection(u, n, orientation, R, dt, endpoint=True, t_step=None):
    """ 
    Compute the likelihood ratio for tonal sources.

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

    for t0 in tqdm(t):
        ell.loc[{"time": t0}] = (
            wrapcauchy(u.loc[{"time": slice(t0 - dt/2, t0 + dt/2)}], zeta)
            .prod("time")
            .mean("frequency")
        )

    return ell


def impulsive_detection(u, n, orientation, R, dt, df, pd, endpoint=True, t_step=None):
    """ 
    Compute the likelihood ratio for impulsive sources.

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

    for t0 in tqdm(t):
        ell.loc[{"time": t0}] = (
            wrapcauchy(u.loc[{"time": slice(t0 - dt/2, t0 + dt/2)}], zeta)
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
