"""
GIS module.

Used to process AIS data, build ship trajectories and retreive route passing
close to a recording instrument.

"""
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from shapely.geometry import Point, LineString
import xarray as xr


def read_ais(ais, timedelta):
    """
    Create a GeoSerie of tracks from AIS data.

    Parameters
    ----------
    ais : DataFrame
        The AIS data with time stored in the 'timestamp' column
        and position in a 'lon and 'lat' column.
    timegap : Timedelta
        The maximum time gap accepted between two AIS logs.

    Returns
    -------
    tracks : GeoSerie
        A GeoSerie containing Linestrings indexed by its MMSI.

    """
    df = ais.copy()
    # convert timedelta in seconds
    td = timedelta / pd.Timedelta(1, 's')
    # sort by time and MMSI
    df = df.sort_values(['mmsi', 'timestamp'])
    # add time in seconds
    df['time'] = (df.timestamp - np.datetime64(0, 's')) / \
        np.timedelta64(1, 's')
    # add time interval between two logs in seconds
    df['dt'] = df.groupby('mmsi')['time'].apply(lambda x: x.diff())
    # add index to separate LineStrings when dt exceeds time_gap
    df['i'] = df.groupby('mmsi')['dt'].apply(lambda x: (x > td).cumsum())
    # remove isolated points
    df = df.groupby(['mmsi', 'i']).filter(lambda x: x.shape[0] > 1)
    # create LineStrings and group in MultiLineString for each MMSI
    tracks = (df
              .groupby(['mmsi', 'i'])
              .apply(lambda x: LineString(x[['lon', 'lat', 'time']].values))
              .reset_index('i', drop=True))
    return tracks


def trim_track(track, t_start, t_end):
    """
    Temporally trim a track.

    Parameters
    ----------
    track : LineString
        Moving source trajectory.
    t_start : float
        Specify the start time (POSIX timestamp in seconds).
    t_end : float
        Specify the end time (POSIX timestamp in seconds).

    Returns
    -------
    LineString
        Trimed moving source trajectory.

    """
    xyt = np.asarray(track.coords)
    mask = (xyt[:, -1] > t_start) & (xyt[:, -1] < t_end)
    return np.nan if (mask.sum() < 2) else LineString(xyt[mask])


def select_tracks(tracks, station, radius, cpa):
    """
    Select tracks passing close to some interest point.

    Parameters
    ----------
    tracks : GeoSerie
        Tracks from which the selection must be done.
    station : Obspy Station
        Station of interest.
    radius : float
        Tracks are intersected in with a disk with radius is given but that
        value (in metres).
    cpa : float
        Tracks passing farther than this are rejected (in metres).

    Returns
    -------
    df : GeoDataFrame
        The selected tracks in the local azimuthal equidistant projection
        as a GeoDataFrame containing tracks and related statistics.

    """
    # insure that original data is preserved
    tracks = tracks.copy()
    # trim data before and after station operating time
    t_start = station.start_date.timestamp
    t_end = station.end_date.timestamp
    tracks = tracks.apply(lambda track: trim_track(track, t_start, t_end))
    tracks = tracks[~tracks.isna()]
    # project in a local CRS
    crs = ccrs.AzimuthalEquidistant(station.longitude, station.latitude)
    tracks = tracks.apply(
        lambda track: LineString(crs.transform_points(
            ccrs.PlateCarree(), *np.array(track.coords).T)))
    # remove tracks which CPA is too far
    centre = Point(0, 0)
    cpa_mask = tracks.apply(centre.distance) <= cpa
    tracks = tracks[cpa_mask]
    # intersect tracks with the interest area
    area = centre.buffer(radius)
    tracks = tracks.apply(area.intersection)
    # if tracks empty return None
    if len(tracks) == 0:
        return None
    # separate eventual MultiLineString in separate LineStrings
    tracks = (tracks
              .apply(pd.Series)  # split MultiLineStrings into LineStrings
              .stack()  # stack them so that each row as a unique LineString
              # only keep the original index which will be use to retrieve MMSI
              .reset_index(level=1, drop=True)
              .apply(lambda track: LineString(
                  sorted(track.coords, key=lambda point: point[-1]))))
    # verify that tracks do not have too far CPA
    cpa_mask = tracks.apply(centre.distance) <= cpa
    tracks = tracks[cpa_mask]
    return tracks


def track2xarr(track):
    """
    Convert a track to a DataArray.

    Parameters
    ----------
    track : LineString
        Moving source trajectory as a LineString.

    Returns
    -------
    xarray.DataArray
        Moving source trajectory as a DataArray.

    """
    x, y, t = np.array(track.coords).T
    data = x + 1j * y
    track = xr.DataArray(data=data, coords={'time': t}, dims='time')
    _, index = np.unique(track['time'], return_index=True)
    track = track.isel(time=index)
    return track
