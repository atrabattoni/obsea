"""
Core module.

Contains all core processing functions to transform raw signal in usefull 2D
representations.

Attributes
----------
CHANNEL_TO_AXIS: dict
    Links channels names to their physical meanings (pressure, velocities)

"""
import numpy as np
import xarray as xr
import scipy.signal as sp
from scipy.ndimage import gaussian_filter1d

CHANNEL_TO_AXIS = {
    'BDH': 'p',
    'BH1': 'vy',
    'BH2': 'vx',
    'BHZ': 'vz',
    'HDH': 'p',
    'EH1': 'vy',
    'EH2': 'vx',
    'EH3': 'vz',
}


def to_dataset(st):
    """
    Convert a stream to a dataset.
    """
    tr = st[0]
    starttime = np.datetime64(tr.stats.starttime.ns, 'ns')
    endtime = np.datetime64(tr.stats.endtime.ns, 'ns')
    delta = np.timedelta64(int(round(1e9 * tr.stats.delta)), 'ns')
    t = np.arange(starttime, endtime + delta, delta)
    data_vars = {CHANNEL_TO_AXIS[tr.stats.channel]:
                 xr.DataArray(tr.data, dims='time', attrs=tr.stats)
                 for tr in st}
    coords = {'time': t}
    attrs = {'starttime': starttime, 'endtime': endtime, 'delta': delta}
    return xr.Dataset(data_vars, coords, attrs)


def stft(da, nperseg, step):
    """
    Compute the Short Time Fourrier Transform (STFT) of a trace.
    """
    # rolling window
    view = (da.rolling(time=nperseg, center=True)
              .construct('frequency', stride=step))
    # tapper
    win = sp.get_window('hann', nperseg)
    win /= win.sum()
    win = xr.DataArray(win, dims='frequency')
    data = view * win
    # fft
    t = data["time"]
    delta = 1.0 / da.attrs["sampling_rate"]
    f = np.fft.rfftfreq(nperseg, delta)
    data = np.fft.rfft(data).T
    result = xr.DataArray(
        data, {'time': t, 'frequency': f}, ('frequency', 'time'))
    result = result.dropna("time")
    return result


def remove_response(tf, response, water_level):
    """
    Remove instrumental response.
    """
    f = tf.coords['frequency']
    response = response.get_evalresp_response_for_frequencies(f)
    w = np.abs(response).max() * 10.0 ** (-water_level / 20.0)
    mask = np.abs(response) < w
    response[mask] = w * np.exp(1j * np.angle(response[mask]))
    response = xr.DataArray(response, {'frequency': f}, 'frequency')
    tf /= response
    return tf


def time_frequency(st, nperseg, step, water_level=None):
    """
    Compute time-frequency representations of trace in stream.

    Instrumental response can be remove by water level deconvolution if it is
    attached to the trace.

    Parameters
    ----------
    st: obspy.Stream
        List like object of multiple traces.
    nperseg: int
        Length of each segment in samples used in the FFT computation.
    step: int
        Number of point between segments.
    water_level: int, optional
        Water level (in dB) used in water level deconvolution. If None, no
        instrumental removal is perform (Defaults).

    Returns
    -------
    xarray.Dataset
        Dataset made of one DataArray per trace with appropriate 'time' and
        'frequency' coordinates.

    """
    ds = to_dataset(st)
    data_vars = {}
    for channel in ds:
        trace = ds[channel]
        result = stft(trace, nperseg, step)
        if water_level is not None:
            response = trace.attrs["response"]
            result = remove_response(result, response, water_level)
        data_vars[channel] = result
    return xr.Dataset(data_vars)


def intensity(z, method='intensity', mode='net'):
    """Compute acoustic intensity.

    Parameters
    ----------
    z: xarray.DataSet
        Must contain a time-frequency representation for horizontal velocities
        ('vx' and 'vy') and for the pressure channel ('p') if method is
        'intensity'
    method: str, optional
        Method used to compute DOA. Either 'intensity' or 'polarization'. If
        polarization is chosen angles are doubled so that the full 360 degree
        range is used (polarization suffers from 180 degree amgiguity)
    mode: str, optional
        Mode used to compute DOA. Either 'net' or 'instantaneous'.

    Returns
    -------
    xarray.DataSet
        Acoustic intensity as a Dataset with components 'vx' and 'vy'.
    """
    if method not in ['intensity', 'polarization']:
        print('error')
    if mode not in ['net', 'instantaneous']:
        print('error')
    # acoustic intensity
    if method == 'intensity':
        if mode == 'net':
            result = np.real(z[['vx', 'vy']] * z['p'].conj()) / 2
        elif mode == 'instantaneous':
            result = z[['vx', 'vy']].real * z['p'].real
        result = result['vx'] + 1j * result['vy']
        result /= np.abs(result)
        double_angle = False
    elif method == 'polarization':
        if mode == 'net':
            result = (z[['vx', 'vy']].to_array(dim='component')
                      .transpose('frequency', 'time', 'component'))
            x = np.stack((result.real.values, result.imag.values), axis=-1)
            u, _, _ = np.linalg.svd(x)
            r = u[..., 0, 0] + 1j * u[..., 0, 1]
            result = xr.DataArray(
                data=r,
                coords={
                    'time': z['time'].values,
                    'frequency': z['frequency'].values},
                dims=('frequency', 'time'))
        elif mode == 'instantaneous':
            result = z[['vx', 'vy']].real
            result = result['vx'] + 1j * result['vy']
            result /= np.abs(result)
        result = result ** 2
        double_angle = True
    result.attrs['double_angle'] = double_angle
    return result


def azigram(z, nperseg, step, method='intensity', mode='net', dim='time',
            iid=1):
    """Compute azigram.

    Parameters
    ----------
    z: xarray.DataSet
        Must contain a time-frequency representation for horizontal velocities
        ('vx' and 'vy') and for the pressure channel ('p') is method is
        'intensity'
    nperseg: int
        Length of each segment in samples used in the mean direction of arrival
        (DOA) and mean running length (MRL) computation.
    step: int
        Number of point between segments.
    method: str, optional
        Method used to compute DOA. Either 'intensity' or 'polarization'. If
        polarization is chosen angles are doubled so that the full 360 degree
        range is used (polarization suffers from 180 degree amgiguity)
    mode: str, optional
        Mode used to compute DOA. Either 'net' or 'instantaneous'.
    dim: str, optional
        Dimension along which to compute the mean running length. Choose 'time'
        for tonal signals and 'frequency' for impulsive signals.
    iid: int, optional
        TODO

    Returns
    -------
    xarray.DataArray
        Azigram of horizontal DOA. DOA is given as a complex values so that the
        real part point toward East and the imaginary part point toward North.
        Modulus of those values are MRL. The DataArray as a attrs.double_angle
        attribute which states if values represent the DOA or its double.

    """
    result = intensity(z, method=method, mode=mode)
    double_angle = result.attrs['double_angle']
    # moving average
    result = result.rolling(**{dim: iid*nperseg}, center=True).construct(
        'w', stride=step)
    result = result.isel(w=slice(None, None, iid))
    result = result.mean('w')
    result.attrs['double_angle'] = double_angle
    return result


def time_azimuth(r, nperseg, step, bins, sigma=None, fmin=None, fmax=None):
    """
    Compute time-azimuth representation of an azigram.

    Approximate pseudo kernel density estimation is performed by smoothing an
    weighted histogram with a Gaussian kernel on temporal windows.

    Parameters
    ----------
    r: xarray.DataArray
        Azigram. Values are complex number which argument are the DOAs and
        modulus are the MRL or any other wanted weight.
    nperseg: int
        Lenght of each segment in samples used in the density estimation.
    step: int
        Number of points between segments.
    bins: int
        Number of bins used to computes histograms on 360 degrees.
    sigma: float, optional
        Standard deviation for Gaussian kernel in degrees. If None, no
        smoothing is applied (default).
    fmin: float, optional
        Values which frequencies are below fmin are not used in the density
        estimation. If None no restriction is applied (default).
    fmax: float, optional
        Values which frequencies are above fmax are not used in the density
        estimation. If None no restriction is applied (default).

    Returns
    -------
    xarray.DataArray
        Time-azimuth representation.

    """
    r = r.sel(frequency=slice(fmin, fmax))
    r = r.dropna(dim='time')
    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)
    rot = np.exp(1j * np.pi / bins)
    result = r * rot
    result = (result
              .rolling(time=nperseg, center=True)
              .construct('h', stride=step))
    data = result.values.swapaxes(-3, -2)
    data = data.reshape(data.shape[:-2] + (data.shape[-2] * data.shape[-1],))

    def histogram(z):
        h = np.histogram(np.arctan2(z.real, z.imag) % (2 * np.pi),
                         bins=bin_edges, density=False, weights=np.abs(z))[0]
        return h * (h.size / z.size) / (2 * np.pi)
    data = np.apply_along_axis(histogram, -1, data)
    # gaussian kernel
    if sigma:
        data = gaussian_filter1d(data, sigma=sigma * bins / 360, mode='wrap')
    coords = {'time': result.coords['time'],
              'azimuth': np.linspace(0, 360, bins, endpoint=False)}
    result = xr.DataArray(data=data, coords=coords, dims=['time', 'azimuth'])
    # transpose
    dims = ([dim for dim in result.dims if dim not in ['azimuth', 'time']]
            + ['azimuth', 'time'])
    result = result.transpose(*dims)
    return result


def orientation_frequency(r, track, bins, sigma=None, fmin=None, fmax=None):
    """
    Compute orientation-frequency representation of an azigram.

    from an azigram and a track of an acoustic source. Approximate pseudo
    kernel density estimation is performed by smoothing an weighted histogram
    with a Gaussian kernel on each frequency.

    Parameters
    ----------
    r: xarray.DataArray
        Azigram. Values are complex number which argument are the DOAs and
        modulus are the MRL or any other wanted weight.
    track: shapely.LineString
        Acoustic source trajectory
    bins: int
        Number of bins used to computes histograms on 360 degrees.
    sigma: float, optional
        Standard deviation for Gaussian kernel in degrees. If None, no
        smoothing is applied (default).
    fmin: float, optional
        Values which frequencies are below fmin are not used in the density
        estimation. If None no restriction is applied (default).
    fmax: float, optional
        Values which frequencies are above fmax are not used in the density
        estimation. If None no restriction is applied (default).

    Returns
    -------
    xarray.DataArray
        Orientation-frequency representation.

    """
    r = r.sel(frequency=slice(fmin, fmax))
    r = r.dropna(dim='time')
    track = track.interp_like(r)
    track /= np.abs(track)
    result = r.conj() * track * np.exp(1j * np.pi / bins)
    bin_edges = np.linspace(0, 2 * np.pi, bins + 1)

    def histogram(z):
        h = np.histogram(-np.arctan2(z.imag, z.real) % (2 * np.pi),
                         bins=bin_edges, density=False, weights=np.abs(z))[0]
        return h * (h.size / z.size) / (2 * np.pi)
    data = np.apply_along_axis(histogram, result.get_axis_num('time'), result)

    # gaussian kernel
    if sigma:
        data = gaussian_filter1d(data, sigma=sigma * bins / 360, mode='wrap')
    coords = {
        'orientation': np.linspace(0, 360, bins, endpoint=False),
        'frequency': result.coords['frequency']}
    result = xr.DataArray(data=data, coords=coords,
                          dims=['frequency', 'orientation'])
    return result


def spectrogram(xarr):
    return 20*np.log10(np.abs(xarr))


def cepstrogram(xarr, analytic=False):
    """
    Compute the cepstrogram of a time-frequency representation.

    Cepstrogram can be usefull to compute time difference between different
    arrivals or to study the harmonic structure of a signal.

    Parameters
    ----------
    xarr: xarray.DataArray
        Time-frequency representation.
    analytic: bool, optional
        Whether to return the analytical signal of the cepstrogram.

    Returns
    -------
    xarray.DataArray
        Computed Cepstrogram.

    """
    f = xarr['frequency'].values
    df = f[1] - f[0]
    q = np.fft.rfftfreq(2*(f.size - 1), df)
    data = np.log(np.abs(xarr.values))
    if analytic:
        data = np.concatenate((
            data[0:1, :],
            2*data[1:-1, :],
            data[-1:, :],
            0*data[-2:0:-1, :]), axis=-2)
        data = np.fft.ifft(data, axis=-2)
    else:
        data = np.fft.irfft(data, axis=-2)
    data = data[..., :q.size, :]
    return xr.DataArray(
        data=data,
        coords={'time': xarr['time'].values, 'quefrency': q},
        dims=('quefrency', 'time'))


def analytic_signal(xarr):
    """
    Compute the analytic_signal of a one sided cepstrogram (or spectrogram).

    Analytic signal can be usefull to compute the envelope and the
    instantaneous phase. It allows coherent summation or incoherent summation
    (adding modulus and discarting the phase).

    Parameters
    ----------
    xarr: xarray.DataArray
        One sided cepstrogram (or spectrogram).

    Returns
    -------
    xarray.DataArray
        Analytic signal of the cepstrogram (or spectrogram).

    """
    q = xarr['quefrency'].values
    data = xarr.values
    data = np.concatenate((data, data[-2:0:-1, :]), axis=-2)
    data = np.fft.rfft(data, axis=-2)
    data = np.concatenate((
        data[0:1, :],
        2*data[1:-1, :],
        data[-1:, :],
        0*data[-2:0:-1, :]), axis=-2)
    data = np.fft.ifft(data, axis=-2)
    data = data[..., :q.size, :]
    return xr.DataArray(
        data=data,
        coords={'time': xarr['time'].values, 'quefrency': q},
        dims=('quefrency', 'time'))
