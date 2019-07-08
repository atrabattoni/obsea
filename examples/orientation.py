import warnings
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as st
import obsea
from obspy.clients.fdsn import Client
from dask import delayed, compute
from dask.diagnostics import ProgressBar


# Inputs
client = Client('RESIF')
inventory = client.get_stations(network='YV', station='RR*', level='response')
station_list = pd.read_csv(
    obsea.get_dataset_path('station_list'), squeeze=True).tolist()
ais_fname = obsea.get_dataset_path('ais_cls')


# Paremeters
timedelta = pd.Timedelta(24, 'h')
radius = 30_000
cpa = 15_000
tf_nperseg = 1024
tf_step = 128
tf_water_level = None
az_nperseg = 8
az_step = 1
af_bins = 3600
af_sigma = 2.0
af_fmin = 1.0
af_fmax = None
nbootstrap = 1000


# Process network
network, = obsea.select_stations(inventory, station_list)


print('------------------\nORIENTATION SCRIPT\n------------------')


# Process AIS
print('Process AIS.')
ais = obsea.read_cls(ais_fname)
global_tracks = obsea.read_ais(ais, timedelta)
local_tracks = {}
for station in network:
    tracks = obsea.select_tracks(
        global_tracks, station, radius, cpa)
    if tracks is not None:
        local_tracks.update({station.code: tracks})
    else:
        pass


# Process data
@delayed
def process(track, station):
    st = obsea.load_stream(track, client, inventory, station, '*')
    if st is None:
        return None
    tf = obsea.time_frequency(st, tf_nperseg, tf_step, tf_water_level)
    r = obsea.azigram(tf, az_nperseg, az_step)
    of = obsea.orientation_frequency(
        r, track, af_bins, sigma=af_sigma, fmin=af_fmin, fmax=af_fmax)
    ppdf = obsea.decompose(of)
    return ppdf


print('Process Data.')
ppdf = {}
for station in network:
    tracks = local_tracks[station.code]
    results = [process(track, station) for track in tracks]
    with ProgressBar(), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        results = compute(*results)
    results = [result for result in results if result is not None]
    if results:
        result = xr.concat(results, dim='route').rename(station.code)
        ppdf.update({station.code: result})


# Performance estimation
@delayed
def process(ppdf, n_sample):
    index = np.random.choice(ppdf.coords['route'].size, n_sample)
    sample = ppdf.isel(route=index)
    result = sample.mean('route')
    result = result.coords['orientation'].isel(
        orientation=result.argmax('orientation')).values
    return result


print('Estimate Uncertainties.')
orientation = pd.DataFrame(
    columns=['station', 'orientation', 'single', 'all', 'nship'])
orientation = orientation.set_index('station')
for station in network:
    ppdf = ppdf[station.code]
    n_sample = ppdf.coords['route'].size
    xarr = ppdf.mean('route')
    mean = xarr.where(xarr == xarr.max('orientation'),
                      drop=True).squeeze().coords['orientation'].values
    diff = ppdf.coords['orientation'].isel(
        orientation=ppdf.argmax('orientation')).values
    diff = (((diff - mean) + 180) % 360) - 180
    iqr = st.iqr(diff, scale='normal')
    with ProgressBar():
        result, = compute(process(ppdf, n_sample) for i in range(nbootstrap))
    diff = (((np.asarray(result) - mean) + 180) % 360) - 180
    std = st.iqr(diff, scale='normal')
    print('orientation for {} routes = {:.1f}° +/- {:.1f}°'.format(
        1, mean, 2 * iqr))
    print('orientation for {} routes = {:.1f}° +/- {:.1f}°'.format(
        n_sample, mean, 2 * std))
    orientation.loc[station.code] = np.round(
        [mean, 2 * iqr, 2 * std, n_sample], 1)
orientation.to_csv('orientation.csv')
print()
