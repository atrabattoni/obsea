"""
Particles
---------
w.shape = (Np,)
x.shape = (Np, Nt, 4)
dims = ("r", "a", "vr", "va")

Observations
------------
yr.shape = (Nt, Nr, Nvr)
ya.shape = (Nt, Na) 

Grid
-----
grid.keys() = ["dt", "dr", "da", "dv", "rmax, "vmax"]
"""

import numpy as np
from numba import njit, vectorize
from numpy.random import choice, rand, randint, randn
from tqdm import tqdm


@njit
def argchoice(p, n=1):
    """ 
    Randomly chose a sample from a collection given their probabilities.

    Parameters
    ----------
    p : array
        Probability mass function. Must sum to one.
    n : int, optional
        Number of drawn samples. Default one.

    Returns
    -------
    array
        Samples indices.
    """
    cdf = np.cumsum(p)
    # rng = rand(n) * cdf[-1]  # random sampling
    rng = np.arange(rand() * cdf[-1] / n, cdf[-1],
                    cdf[-1] / n)  # systematic sampling
    return np.searchsorted(cdf, rng)


@vectorize(nopython=True)
def clamp(x, xmin, xmax):
    if x <= xmin:
        return xmin
    elif x >= xmax:
        return xmax
    else:
        return x


@njit
def isempty(state):
    return np.any(np.isnan(state))


@njit
def modulus(x, y):
    return np.sqrt(x**2 + y**2)


@njit
def bearing(x, y):
    return np.arctan2(x, y) % (2 * np.pi)


def checkgrid(grid, yr, ya):

    dr = grid["dr"]
    da = grid["da"]
    dv = grid["dv"]
    rmax = grid["rmax"]
    vmax = grid["vmax"]

    rg = np.arange(0, rmax + dr, dr)
    vg = np.arange(-vmax, vmax + dv, dv)
    ag = np.arange(0, 2 * np.pi + da, da)

    if not len(yr) == len(ya):
        ValueError(
            f"Cepstral duration {len(yr)} do not match azimuthal duration {len(ya)}")
    if not yr.shape[1:] == (len(rg), len(vg)):
        ValueError(
            f"Cepstral observation {yr.shape[1:]} do not match grid {(len(rg), len(vg))}")
    if not ya.shape[1:] == (len(ag),):
        ValueError(
            f"Azimuthal observation {ya.shape[1:]} do not match grid {(len(ag),)}")


class Tracker:

    def __init__(self, Np, pb, ps, pd, q, grid):
        """
        Compile functions with a given model.

        Parameters
        ----------
        pb : float
            Birth probability.
        ps : float
            Survival probability.
        q : function
            Process noise.
        grid : dict
            Coordinate spacing and range.

        Returns
        -------
        function
            Compiled functions.
        """

        self.Np = Np
        self.pb = pb
        self.ps = ps
        self.q = q
        self.grid = grid

        dt = grid["dt"]
        dr = grid["dr"]
        da = grid["da"]
        dv = grid["dv"]
        rmax = grid["rmax"]
        vmin = grid["vmin"]
        vmax = grid["vmax"]

        rg = np.arange(0, rmax + dr, dr)
        vg = np.arange(-vmax, vmax + dv, dv)

        @njit
        def i2r(i):
            return i * dr

        self.i2r = i2r

        @njit
        def r2i(r):
            return int(round(r / dr))

        self.r2i = r2i

        @njit
        def i2a(i):
            return i * da

        self.i2a = i2a

        @njit
        def a2i(a):
            return int(round(a / da))

        self.a2i = a2i

        @njit
        def i2vr(i):
            return -vmax + i * dv

        self.i2vr = i2vr

        @njit
        def vr2i(vr):
            return int(round((vr + vmax) / dv))

        self.vr2i = vr2i

        @njit
        def initialize(Nt):
            """
            Intialize trajectory particles and their respective weights.

            Parameters
            ----------
            Np : int
                Number of particles.
            Nt : int
                Number of time bins.

            Returns
            -------
            w : array
                Uniform weigths of probability 1/Np.
            x : array
                Empty particles filled with nan.
            """
            w = np.full(Np, 1.0 / Np)
            x = np.full((Np, Nt, 4), np.nan)
            return w, x

        self.initialize = initialize

        @njit
        def predict(w, x, yr, ya, n):
            """
            Predict states of particules x at time step n from observations y.

            Parameters
            ----------
            x : array
                Particles.
            y : array
                Observations.
            n : int
                Time step.
            """
            for k in range(len(x)):
                if isempty(x[k, n - 1]):
                    if rand() < pb:
                        w[k], x[k, n] = birth(yr[n], ya[n])
                else:
                    if rand() < ps:
                        x[k, n] = move(x[k, n - 1])

        self.predict = predict

        pr = rg
        pvr = np.sqrt(vmax**2 - vg**2) - \
            np.sqrt(vmin**2 - clamp(vg**2, 0, vmin**2))
        p = np.outer(pr, pvr)

        @njit
        def birth(yr, ya):
            w = 1.0
            # range
            i = argchoice(np.ravel(yr * p))[0]
            w /= np.ravel(yr)[i]
            r = i2r(i // yr.shape[-1])
            vr = i2vr(i % yr.shape[-1])
            # azimuth
            i = argchoice(ya)[0]
            w /= ya[i]
            a = i2a(i)
            vamax = np.sqrt(vmax**2 - vr**2)
            vamin = np.sqrt(vmin**2 - clamp(vr**2, 0, vmin**2))
            va = (vamax - vamin) * rand() + vamin
            va *= choice(np.array([-1.0, 1.0]))
            # output
            state = np.array([r, a, vr, va])
            return w, state

        self.birth = birth

        @njit
        def cartesian(state):
            r, a, vr, va = state
            x = r * np.sin(a)
            y = r * np.cos(a)
            vx = vr * np.sin(a) + va * np.cos(a)
            vy = vr * np.cos(a) - va * np.sin(a)
            state = np.array([x, y, vx, vy])
            return state

        self.cartesian = cartesian

        @njit
        def polar(state):
            x, y, vx, vy = state
            r = modulus(x, y)
            a = bearing(x, y)
            vr = vx * np.sin(a) + vy * np.cos(a)
            va = vx * np.cos(a) - vy * np.sin(a)
            state = np.array([r, a, vr, va])
            return state

        self.polar = polar

        @njit
        def constant_speed(state):
            x, y, vx, vy = state
            # process noise
            a = q * randn(2)
            # motion
            x += vx * dt + a[0] * dt**2 / 2.0
            y += vy * dt + a[1] * dt**2 / 2.0
            vx += a[0] * dt
            vy += a[1] * dt
            vx = clamp(vx, -vmax, vmax)
            vy = clamp(vy, -vmax, vmax)
            state = np.array([x, y, vx, vy])
            return state

        self.constant_speed = constant_speed

        @njit
        def move(state):
            state = cartesian(state)
            state = constant_speed(state)
            state = polar(state)
            return state

        self.move = move

        @njit
        def update(w, x, yr, ya, n):
            for k in range(len(x)):
                if not isempty(x[k, n]):
                    r, a, vr, _ = x[k, n]
                    # range
                    if r > rmax:
                        w[k] *= 1.0
                    else:
                        w[k] *= (1 - pd) + pd * yr[n, r2i(r), vr2i(vr)]
                    # azimuth
                    w[k] *= (1 - pd) + pd * ya[n, a2i(a)]
            # normazlization
            w /= np.sum(w)

        self.update = update

        @njit
        def resample(w, x, n):
            x[:, :n] = x[argchoice(w, len(w)), :n]
            w[:] = 1.0 / len(w)

        self.resample = resample

        def track(yr, ya):

            if not checkgrid(grid, yr, ya):
                return None

            Nt = len(yr)

            w, x = initialize(Nt)
            for n in tqdm(range(len(yr))):
                predict(w, x, yr, ya, n)
                update(w, x, yr, ya, n)
                resample(w, x, n)

            return w, x

        self.track = track
